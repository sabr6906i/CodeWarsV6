import socket
import threading
import time
import numpy as np
import pygame
import os
from engine.weapons.weapons import WEAPONS, get_weapon
from engine.spawners.gun_spawner import GunSpawner, PlayerInventory
from engine.spawners.medkit_spawner import MedkitSpawner
import config

class Server:
    def grenade_effect_active_after_explosion(self, grenade_slot):
        """True when a grenade slot is no longer active but its lingering effect still exists."""
        if 48 <= grenade_slot <= 54 and self.world_data[grenade_slot, 0] == 1:
            return False
        return any(
            effect.get('source_slot') == grenade_slot and effect.get('duration', 0) > 0
            for effect in self.gas_effects.values()
        )
    
    def grenade_damage(self, distance, max_damage, radius, falloff=2):
        if distance >= radius:
            return 0.0
        damage = max_damage * (1 - (distance / radius))
        return damage

    def throw_grenade(self, player_id, grenade_id, throw_angle, throw_power):
        """
        Spawns a grenade with gravity-based arc.
        - player_id: int, the player throwing
        - grenade_id: int, type of grenade (from weapons.py)
        - throw_angle: float, angle in radians
        - throw_power: float, initial speed
        """
        from engine.weapons.weapons import get_grenade
        # Find a free grenade slot (rows 48-54)
        for i in range(48, 55):
            if self.world_data[i, 0] == 0:
                # Get player position
                px, py = self.world_data[player_id, 1], self.world_data[player_id, 2]
                vx = throw_power * np.cos(throw_angle)
                vy = throw_power * np.sin(throw_angle)
                grenade = get_grenade(grenade_id)
                if grenade is None:
                    return False
                # Fill world_data for grenade
                self.world_data[i, 0] = 1  # active
                self.world_data[i, 1] = px
                self.world_data[i, 2] = py
                self.world_data[i, 3] = throw_angle  # store angle if needed
                self.world_data[i, 4] = vx  # vx
                self.world_data[i, 5] = vy  # vy
                self.world_data[i, 6] = grenade.blast_radius
                self.world_data[i, 7] = grenade.damage
                self.world_data[i, 8] = grenade.effect_time
                self.world_data[i, 9] = player_id
                self.world_data[i, 10] = grenade_id
                # Store fuse timer in a dict for each grenade slot
                if not hasattr(self, 'grenade_fuse_timers'):
                    self.grenade_fuse_timers = {}
                # proxy grenades should arm after 2 seconds regardless of their config
                if grenade.is_proxy:
                    self.grenade_fuse_timers[i] = 2.0
                    # ensure we have an armed-state tracker
                    if not hasattr(self, 'proxy_armed'):
                        self.proxy_armed = set()
                else:
                    self.grenade_fuse_timers[i] = grenade.fuse_time
                return True
        return False  # No free slot
    
    def __init__(self):
        PORT = config.SERVER_PORT
        if not self._start_server(PORT):
            return

        self.setup_game()
        self.player_lock = threading.Lock()

        thread = threading.Thread(target=self.add_players, daemon=True)
        thread.start()

        self.run_game()

    def _start_server(self, PORT):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        SERVER_IP = getattr(config, "SERVER_BIND_HOST", None)
        if not SERVER_IP:
            HOST_NAME = socket.gethostname()
            SERVER_IP = socket.gethostbyname(HOST_NAME)

        try:
            self.server_socket.bind((SERVER_IP, PORT))
        except socket.error as e:
            print(str(e))
            print("[SERVER] Server could not start")
            return False

        self.server_socket.listen()

        print(f"[SERVER] Server Started with bind ip {SERVER_IP}")
        return True

    def setup_game(self):
        self.match_ended = False
        self.match_start_time = None
        self.match_duration = config.MATCH_DURATION
        self.player_names = [""] * 8
        # Grenade cooldown per player (seconds)
        self.player_grenade_cooldown = np.zeros(8, dtype=np.float64)
        self.player_respawn_cooldown = np.zeros(8, dtype=np.float64)
        self.player_count = 0
        # [kills, deaths] per player. K/D delta is computed as kills - deaths.
        self.player_stats = np.zeros((8, 2), dtype=np.int32)
        # world_data columns (per row):
        # 0:is_alive, 1:x, 2:y, 3:theta, 4:v, 5:omega_or_traveled, 6:fuel, 7:health_or_damage, 8:score, 9:current_ammo, 10:total_ammo_or_weapon_id
        # for tanks: column 6 = fuel, 7 = health, 8 = score, 9 = current_ammo, 10 = total_ammo
        # for bullets: column 5 = distance traveled, 7 = snapshot damage, 9 = owner id, 10 = weapon id
        self.world_data = np.zeros((55, 11), dtype=np.float64)
        self.player_inputs = np.zeros((8, 14), dtype=np.int32)  # inputs: W,A,D,UP,DOWN,LEFT,RIGHT,SPACE,R,S,G,C,P,KNEEL
        self.grenade_data = np.zeros((10, 4), dtype=np.float64)  # separate array for grenades (slots 48-54 in world_data)
        self.grenade_data[:8, 0] = 1  # default selected grenade type: frag
        self.grenade_data[:8, 1] = config.FRAG_GRENADE_COUNT
        self.grenade_data[:8, 2] = config.PROXY_GRENADE_COUNT
        self.grenade_data[:8, 3] = config.GAS_GREANADE_COUNT

        # Load constants from config
        TANK_V = config.TANK_SPEED
        TANK_OMEGA = config.TANK_ROTATION_SPEED
        BULLET_V = config.BULLET_SPEED

        self.world_data[:8, 4] = TANK_V
        self.world_data[:8, 5] = TANK_OMEGA
        self.world_data[8:, 4] = BULLET_V
        # per-player vertical velocity for gravity (tanks only)
        self.player_vy = np.zeros(8, dtype=np.float64)
        self.player_stuck_frames = np.zeros(8, dtype=np.int32)
        # screen and tank constants for ground handling
        self.SCREEN_W = 800
        self.SCREEN_H = 600
        self.TANK_RADIUS = config.TANK_VISUAL_RADIUS
        self.COLLISION_RADIUS = config.TANK_COLLISION_RADIUS
        self.PLAYER_HITBOX_W = float(getattr(config, 'PLAYER_HITBOX_WIDTH', 40.0))
        self.PLAYER_HITBOX_H = float(getattr(config, 'PLAYER_HITBOX_HEIGHT', 40.0))
        self.PLAYER_HALF_W = self.PLAYER_HITBOX_W * 0.5
        self.PLAYER_HALF_H = self.PLAYER_HITBOX_H * 0.5
        self.KNEEL_HEIGHT_DELTA = float(getattr(config, 'KNEEL_HEIGHT_DELTA', 5.0))
        self.KNEEL_HALF_H = max(2.0, self.PLAYER_HALF_H - (self.KNEEL_HEIGHT_DELTA * 0.5))
        self.KNEEL_SPEED_MULTIPLIER = float(getattr(config, 'KNEEL_SPEED_MULTIPLIER', 0.6))
        self.player_is_kneeling = np.zeros(8, dtype=bool)
        self.player_half_h = np.full(8, self.PLAYER_HALF_H, dtype=np.float64)
        self.GROUND_Y = self.SCREEN_H - self.PLAYER_HALF_H
        self.GRAVITY = config.GRAVITY
        
        # Jetpack system
        self.player_fuel = np.full(8, config.MAX_FUEL, dtype=np.float64)
        self.MAX_FUEL = config.MAX_FUEL
        self.FUEL_CONSUMPTION = config.FUEL_CONSUMPTION
        self.FUEL_RECHARGE = config.FUEL_RECHARGE
        self.JETPACK_THRUST = config.JETPACK_THRUST
        self.KNEEL_JUMP_IMPULSE = float(getattr(config, 'KNEEL_JUMP_IMPULSE', 3.0))
        self.KNEEL_GROUND_TOLERANCE = float(getattr(config, 'KNEEL_GROUND_TOLERANCE', 1.5))
        
        # Use column 6 (info) to store fuel for tanks
        self.world_data[:8, 6] = self.player_fuel

        # Health and score per tank
        self.MAX_HEALTH = config.MAX_HEALTH
        self.HEALTH_REGEN_ENABLED = bool(getattr(config, 'HEALTH_REGEN_ENABLED', False))
        self.HEALTH_REGEN_PER_SECOND = float(getattr(config, 'HEALTH_REGEN_PER_SECOND', 0.0))
        self.RESPAWN_DELAY = getattr(config, 'RESPAWN_DELAY', 5.0)
        self.world_data[:8, 7] = self.MAX_HEALTH  # health
        self.world_data[:8, 8] = 0.0  # score

        # Weapon system - player inventories with dual gun support
        secondary_default = getattr(config, 'DEFAULT_SECONDARY_WEAPON', 2)
        self.player_inventories = [
            PlayerInventory(
                starting_weapon_id=config.get_random_starting_weapon(),
                secondary_weapon_id=secondary_default,
            )
            for _ in range(8)
        ]
        
        # Fire rate cooldown per player (in seconds, tracks time since last shot)
        self.player_fire_cooldown = np.zeros(8, dtype=np.float64)
        self.saw_charge_end_time = np.zeros(8, dtype=np.float64)
        # Reload cooldown per player (in seconds, tracks time remaining for reload)
        self.player_reload_cooldown = np.zeros(8, dtype=np.float64)
        # Track previous frame's input state for edge detection
        self.previous_inputs = np.zeros((8, 14), dtype=np.int32)
        self.last_frame_time = time.time()
        
        # Collision map system - grid-based obstacles
        self.GRID_SIZE = config.GRID_SIZE
        
        # Load map from file - this will set GRID_W, GRID_H, and collision_map
        self.current_map_name = config.DEFAULT_MAP
        self.load_map(self.current_map_name)
        
        # Gas grenade effect tracking - stores active gas zones
        # Format: {effect_id: {'x': x, 'y': y, 'radius': radius, 'damage': damage, 'duration': remaining_time}}
        self.gas_effects = {}
        self.gas_effect_counter = 0  # Unique ID for each gas effect
        
        # Initialize gun spawner system
        self.gun_spawner = GunSpawner()
        self.gun_spawner.initialize_map(self.current_map_name)
        
        # Initialize medkit spawner system
        self.medkit_spawner = MedkitSpawner()
        self.medkit_spawner.set_collision_map(self.collision_map, self.GRID_SIZE, self.GRID_W, self.GRID_H)
        self.medkit_spawner.initialize_map(self.current_map_name)
        
        # Update screen dimensions based on loaded map
        self.SCREEN_W = self.GRID_W * self.GRID_SIZE
        self.SCREEN_H = self.GRID_H * self.GRID_SIZE
        self.GROUND_Y = self.SCREEN_H - self.PLAYER_HALF_H

        # Build safe spawn positions from map geometry so players never spawn outside terrain.
        self._rebuild_spawn_candidates()
        
        # Convert map to bytes for transmission
        self.collision_map_bytes = self.collision_map.tobytes()

    def _record_player_death(self, victim_idx, killer_idx=None):
        """Register one death and optional kill attribution, then schedule respawn."""
        if self.world_data[victim_idx, 0] != 1:
            return False

        self.player_stats[victim_idx, 1] += 1

        if killer_idx is not None and 0 <= killer_idx < 8 and killer_idx != victim_idx:
            self.player_stats[killer_idx, 0] += 1
            self.world_data[killer_idx, 8] = float(self.player_stats[killer_idx, 0])

        self.world_data[victim_idx, 7] = 0
        self.respawn(victim_idx, delay=self.RESPAWN_DELAY)
        return True

    def _apply_damage_to_player(self, victim_idx, damage, killer_idx=None):
        """Apply damage and attribute kill to attacker only if this is the lethal hit."""
        if self.world_data[victim_idx, 0] != 1:
            return False
        if damage <= 0:
            return False

        self.world_data[victim_idx, 7] -= float(damage)
        if self.world_data[victim_idx, 7] <= 0:
            return self._record_player_death(victim_idx, killer_idx)
        return False

    def _apply_health_regeneration(self, delta_time):
        """Apply slow linear health regeneration to alive players."""
        if not self.HEALTH_REGEN_ENABLED:
            return
        if self.HEALTH_REGEN_PER_SECOND <= 0.0:
            return

        regen_amount = self.HEALTH_REGEN_PER_SECOND * float(delta_time)
        if regen_amount <= 0.0:
            return

        for idx in range(8):
            if self.world_data[idx, 0] != 1:
                continue

            health = self.world_data[idx, 7]
            if health <= 0.0 or health >= self.MAX_HEALTH:
                continue

            self.world_data[idx, 7] = min(self.MAX_HEALTH, health + regen_amount)

    def _build_leaderboard_array(self):
        """Return top-5 leaderboard rows: [player_idx, kills, deaths, kills_minus_deaths]."""
        rows = []
        for idx in range(8):
            kills = int(self.player_stats[idx, 0])
            deaths = int(self.player_stats[idx, 1])
            kd_delta = kills - deaths
            name = self.player_names[idx].strip() if idx < len(self.player_names) else ""
            include = bool(name) or self.world_data[idx, 0] != 0 or kills > 0 or deaths > 0
            if include:
                rows.append((idx, kills, deaths, kd_delta))

        rows.sort(key=lambda r: (-r[1], -r[3], r[0]))

        board = np.full((5, 4), -1, dtype=np.int32)
        for i, row in enumerate(rows[:5]):
            board[i] = np.array(row, dtype=np.int32)
        return board

    def _rebuild_spawn_candidates(self):
        """Collect valid standable points from map collision tiles."""
        self.spawn_candidates = []

        min_x = self.PLAYER_HALF_W
        max_x = self.SCREEN_W - self.PLAYER_HALF_W
        min_y = self.PLAYER_HALF_H
        max_y = self.SCREEN_H - self.PLAYER_HALF_H

        for gy in range(1, self.GRID_H):
            for gx in range(self.GRID_W):
                # 0 = obstacle/floor tile, 1 = passable tile
                if self.collision_map[gy, gx] != 0:
                    continue
                if self.collision_map[gy - 1, gx] != 1:
                    continue

                x = gx * self.GRID_SIZE + (self.GRID_SIZE // 2)
                y = gy * self.GRID_SIZE - self.PLAYER_HALF_H

                if x < min_x or x > max_x or y < min_y or y > max_y:
                    continue

                if not self.is_player_colliding_with_obstacle(x, y):
                    self.spawn_candidates.append((float(x), float(y)))

    def _get_safe_spawn_position(self):
        """Return a safe spawn point inside map bounds and away from solid tiles."""
        if getattr(self, 'spawn_candidates', None):
            idx = np.random.randint(0, len(self.spawn_candidates))
            return self.spawn_candidates[idx]

        min_x = int(np.ceil(self.PLAYER_HALF_W))
        max_x = int(np.floor(self.SCREEN_W - self.PLAYER_HALF_W))

        if max_x <= min_x:
            return float(self.SCREEN_W // 2), float(max(self.PLAYER_HALF_H, self.SCREEN_H - self.PLAYER_HALF_H))

        for _ in range(64):
            spawn_x = float(np.random.randint(min_x, max_x + 1))
            ground_y = self.find_ground_below(spawn_x, 0)
            spawn_y = float(self.GROUND_Y if ground_y is None else ground_y)
            spawn_y = float(np.clip(spawn_y, self.PLAYER_HALF_H, self.SCREEN_H - self.PLAYER_HALF_H))
            if not self.is_player_colliding_with_obstacle(spawn_x, spawn_y):
                return spawn_x, spawn_y

        return float(self.SCREEN_W // 2), float(np.clip(self.GROUND_Y, self.PLAYER_HALF_H, self.SCREEN_H - self.PLAYER_HALF_H))

    def load_map(self, map_name):
        """Load map from maps/ folder or create default if not found"""
        map_path = os.path.join("maps", f"{map_name}.npy")
        
        if os.path.exists(map_path):
            try:
                self.collision_map = np.load(map_path)
                # Extract dimensions from loaded map
                self.GRID_H, self.GRID_W = self.collision_map.shape
                print(f"[SERVER] Loaded map: {map_name} (dimensions: {self.GRID_W}x{self.GRID_H})")
                return True
            except Exception as e:
                print(f"[SERVER] Error loading map {map_name}: {e}")
        
        # Create default map if file not found
        print(f"[SERVER] Map '{map_name}' not found, creating default map")
        # Use default dimensions if not already set
        if not hasattr(self, 'GRID_W'):
            self.GRID_W = 80
            self.GRID_H = 60
        
        self.collision_map = np.ones((self.GRID_H, self.GRID_W), dtype=np.int32)
        
        # Simple default layout
        self.collision_map[-1, :] = 0  # Ground floor
        if self.GRID_H > 20:
            self.collision_map[20, 5:min(15, self.GRID_W)] = 0  # Left platform
        if self.GRID_H > 15:
            self.collision_map[15, 25:min(35, self.GRID_W)] = 0  # Right platform
        if self.GRID_H > 10:
            self.collision_map[10, 10:min(20, self.GRID_W)] = 0  # Top platform
        
        return False
    
    def is_colliding_with_obstacle(self, x, y, radius):
        """Check if a circle at (x, y) with given radius collides with any obstacle"""
        # Check all grid cells that the circle overlaps
        min_grid_x = max(0, int((x - radius) / self.GRID_SIZE))
        max_grid_x = min(self.GRID_W - 1, int((x + radius) / self.GRID_SIZE))
        min_grid_y = max(0, int((y - radius) / self.GRID_SIZE))
        max_grid_y = min(self.GRID_H - 1, int((y + radius) / self.GRID_SIZE))
        
        for gy in range(min_grid_y, max_grid_y + 1):
            for gx in range(min_grid_x, max_grid_x + 1):
                if self.collision_map[gy, gx] == 0:  # obstacle
                    # Check if circle intersects this grid cell
                    cell_x = gx * self.GRID_SIZE
                    cell_y = gy * self.GRID_SIZE
                    # Find closest point in rectangle to circle center
                    closest_x = max(cell_x, min(x, cell_x + self.GRID_SIZE))
                    closest_y = max(cell_y, min(y, cell_y + self.GRID_SIZE))
                    # Check distance
                    dist = np.sqrt((x - closest_x)**2 + (y - closest_y)**2)
                    if dist < radius:
                        return True
        return False

    def is_rect_colliding_with_obstacle(self, cx, cy, half_w, half_h):
        """Check if axis-aligned rectangle centered at (cx, cy) overlaps any obstacle tile."""
        left = cx - half_w
        right = cx + half_w
        top = cy - half_h
        bottom = cy + half_h

        min_grid_x = max(0, int(np.floor(left / self.GRID_SIZE)))
        max_grid_x = min(self.GRID_W - 1, int(np.floor(right / self.GRID_SIZE)))
        min_grid_y = max(0, int(np.floor(top / self.GRID_SIZE)))
        max_grid_y = min(self.GRID_H - 1, int(np.floor(bottom / self.GRID_SIZE)))

        for gy in range(min_grid_y, max_grid_y + 1):
            for gx in range(min_grid_x, max_grid_x + 1):
                if self.collision_map[gy, gx] != 0:
                    continue
                cell_left = gx * self.GRID_SIZE
                cell_top = gy * self.GRID_SIZE
                cell_right = cell_left + self.GRID_SIZE
                cell_bottom = cell_top + self.GRID_SIZE
                if right > cell_left and left < cell_right and bottom > cell_top and top < cell_bottom:
                    return True
        return False

    def _get_player_half_h(self, player_idx):
        if player_idx is None:
            return self.PLAYER_HALF_H
        return float(self.player_half_h[player_idx])

    def _get_target_half_h(self, kneeling):
        return self.KNEEL_HALF_H if kneeling else self.PLAYER_HALF_H

    def is_player_colliding_with_obstacle(self, x, y, player_idx=None):
        return self.is_rect_colliding_with_obstacle(x, y, self.PLAYER_HALF_W, self._get_player_half_h(player_idx))

    def _push_player_out_of_obstacle(self, x, y, player_idx=None, push_x=0.0, push_y=0.0):
        """Try to move a player rectangle out of terrain using short directional nudges."""
        if not self.is_player_colliding_with_obstacle(x, y, player_idx):
            return x, y

        directions = []
        pref_len = np.hypot(push_x, push_y)
        if pref_len > 1e-6:
            directions.append((push_x / pref_len, push_y / pref_len))
        directions.extend([
            (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
            (0.707, 0.707), (0.707, -0.707), (-0.707, 0.707), (-0.707, -0.707)
        ])

        for distance in range(1, self.GRID_SIZE + 6):
            for dx, dy in directions:
                test_x = x + dx * distance
                test_y = y + dy * distance
                if not self.is_player_colliding_with_obstacle(test_x, test_y, player_idx):
                    return test_x, test_y
        return x, y

    def _distance_point_to_player_hitbox(self, px, py, player_idx):
        """Distance from point to nearest point on player's rectangle hitbox."""
        cx = self.world_data[player_idx, 1]
        cy = self.world_data[player_idx, 2]
        half_h = self._get_player_half_h(player_idx)
        left = cx - self.PLAYER_HALF_W
        right = cx + self.PLAYER_HALF_W
        top = cy - half_h
        bottom = cy + half_h
        nearest_x = max(left, min(px, right))
        nearest_y = max(top, min(py, bottom))
        return float(np.hypot(px - nearest_x, py - nearest_y))

    def _point_hits_player_hitbox(self, px, py, player_idx, padding=0.0):
        cx = self.world_data[player_idx, 1]
        cy = self.world_data[player_idx, 2]
        half_h = self._get_player_half_h(player_idx)
        return (
            (cx - self.PLAYER_HALF_W - padding) <= px <= (cx + self.PLAYER_HALF_W + padding)
            and (cy - half_h - padding) <= py <= (cy + half_h + padding)
        )

    def _segment_hits_player_hitbox(self, x0, y0, x1, y1, player_idx, padding=0.0, max_step=2.0):
        """Swept segment hit test against player rectangle to prevent bullet tunneling."""
        cx = self.world_data[player_idx, 1]
        cy = self.world_data[player_idx, 2]
        half_h = self._get_player_half_h(player_idx)
        left = cx - self.PLAYER_HALF_W - padding
        right = cx + self.PLAYER_HALF_W + padding
        top = cy - half_h - padding
        bottom = cy + half_h + padding

        seg_min_x = min(x0, x1)
        seg_max_x = max(x0, x1)
        seg_min_y = min(y0, y1)
        seg_max_y = max(y0, y1)
        if seg_max_x < left or seg_min_x > right or seg_max_y < top or seg_min_y > bottom:
            return False

        dx = x1 - x0
        dy = y1 - y0
        length = float(np.hypot(dx, dy))
        if length <= 1e-8:
            return self._point_hits_player_hitbox(x0, y0, player_idx, padding=padding)

        steps = max(1, int(np.ceil(length / max_step)))
        for i in range(steps + 1):
            t = i / steps
            px = x0 + dx * t
            py = y0 + dy * t
            if left <= px <= right and top <= py <= bottom:
                return True
        return False

    def _push_out_of_obstacle(self, x, y, radius, push_x=0.0, push_y=0.0):
        """Try to move a grenade out of solid geometry using small radial offsets."""
        if not self.is_colliding_with_obstacle(x, y, radius):
            return x, y

        directions = []
        pref_len = np.hypot(push_x, push_y)
        if pref_len > 1e-6:
            directions.append((push_x / pref_len, push_y / pref_len))
        directions.extend([
            (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
            (0.707, 0.707), (0.707, -0.707), (-0.707, 0.707), (-0.707, -0.707)
        ])

        for distance in range(1, self.GRID_SIZE + 3):
            for dx, dy in directions:
                test_x = x + dx * distance
                test_y = y + dy * distance
                if not self.is_colliding_with_obstacle(test_x, test_y, radius):
                    return test_x, test_y

        return x, y

    def _update_bouncy_grenade(self, grenade_slot, radius=4.0):
        """Stepwise movement with bounce, rolling friction, and anti-stuck resolution."""
        x = self.world_data[grenade_slot, 1]
        y = self.world_data[grenade_slot, 2]
        vx = self.world_data[grenade_slot, 4]
        vy = self.world_data[grenade_slot, 5] + self.GRAVITY

        wall_bounce = 0.10
        floor_bounce = 0.10
        roll_friction = 0.95

        speed = max(abs(vx), abs(vy), 1.0)
        steps = min(8, max(1, int(np.ceil(speed / 4.0))))
        dx = vx / steps
        dy = vy / steps

        hit_horizontal = False
        hit_vertical = False

        for _ in range(steps):
            next_x = x + dx
            if self.is_colliding_with_obstacle(next_x, y, radius):
                hit_horizontal = True
                vx = -vx * wall_bounce
                dx = vx / steps
                x, y = self._push_out_of_obstacle(x, y, radius, push_x=np.sign(vx), push_y=0.0)
            else:
                x = next_x

            next_y = y + dy
            if self.is_colliding_with_obstacle(x, next_y, radius):
                hit_vertical = True
                falling = vy > 0.0
                vy = -vy * (floor_bounce if falling else wall_bounce)
                if falling:
                    vx *= roll_friction
                dy = vy / steps
                x, y = self._push_out_of_obstacle(x, y, radius, push_x=0.0, push_y=-np.sign(vy if vy != 0 else 1.0))
            else:
                y = next_y

        on_ground = self.is_colliding_with_obstacle(x, y + 1.5, radius)
        if on_ground:
            vx *= 0.97
            if abs(vx) < 0.08:
                vx = 0.0
            if abs(vy) < 0.35:
                vy = 0.0

        if not hasattr(self, 'grenade_stuck_frames'):
            self.grenade_stuck_frames = {}

        if (hit_horizontal or hit_vertical) and abs(vx) < 0.12 and abs(vy) < 0.12 and self.is_colliding_with_obstacle(x, y, radius):
            self.grenade_stuck_frames[grenade_slot] = self.grenade_stuck_frames.get(grenade_slot, 0) + 1
        else:
            self.grenade_stuck_frames[grenade_slot] = 0

        if self.grenade_stuck_frames.get(grenade_slot, 0) > 3:
            # Nudge upward and sideways so grenades never stay embedded in corners.
            nudge = -1.0 if (grenade_slot % 2 == 0) else 1.0
            x, y = self._push_out_of_obstacle(x + nudge * 1.5, y - 2.0, radius, push_x=nudge, push_y=-1.0)
            vx = nudge * max(0.8, abs(vx) + 0.5)
            vy = -1.6
            self.grenade_stuck_frames[grenade_slot] = 0

        self.world_data[grenade_slot, 1] = x
        self.world_data[grenade_slot, 2] = y
        self.world_data[grenade_slot, 4] = vx
        self.world_data[grenade_slot, 5] = vy

    def _update_non_bouncy_grenade(self, grenade_slot, radius=4.0):
        """Stepwise movement without bounce; grenade stops on first collision."""
        x = self.world_data[grenade_slot, 1]
        y = self.world_data[grenade_slot, 2]
        vx = self.world_data[grenade_slot, 4]
        vy = self.world_data[grenade_slot, 5]

        steps = max(1, int(np.ceil(max(abs(vx), abs(vy), 1.0))))
        dx = vx / steps
        dy = vy / steps

        for _ in range(steps):
            next_x = x + dx
            next_y = y + dy
            if self.is_colliding_with_obstacle(next_x, next_y, radius):
                # Keep grenade just outside geometry and fully stop it.
                x, y = self._push_out_of_obstacle(x, y, radius, push_x=-np.sign(dx), push_y=-np.sign(dy if dy != 0 else 1.0))
                vx = 0.0
                vy = 0.0
                break
            x = next_x
            y = next_y

        x = float(np.clip(x, radius, self.SCREEN_W - radius))
        y = float(np.clip(y, radius, self.SCREEN_H - radius))

        self.world_data[grenade_slot, 1] = x
        self.world_data[grenade_slot, 2] = y
        self.world_data[grenade_slot, 4] = vx
        self.world_data[grenade_slot, 5] = vy
    
    def find_ground_below(self, x, y, player_idx=None):
        """Find standable center-y below a player rectangle centered at (x, y)."""
        half_h = self._get_player_half_h(player_idx)
        left_x = x - self.PLAYER_HALF_W
        right_x = x + self.PLAYER_HALF_W
        min_grid_x = max(0, int(np.floor(left_x / self.GRID_SIZE)))
        max_grid_x = min(self.GRID_W - 1, int(np.floor(right_x / self.GRID_SIZE)))

        if min_grid_x > max_grid_x:
            return None

        feet_y = y + half_h
        start_grid_y = int(np.floor(feet_y / self.GRID_SIZE))
        if start_grid_y < 0:
            start_grid_y = -1

        for gy in range(start_grid_y + 1, self.GRID_H):
            if np.any(self.collision_map[gy, min_grid_x:max_grid_x + 1] == 0):
                return gy * self.GRID_SIZE - half_h

        return None
    
    def _get_barrel_distance(self, weapon_id):
        """Get distance from player center to gun barrel tip for bullet spawning"""
        barrel_distances = {
            1: 35, 2: 35, 6: 36,  # Pistols
            7: 34, 8: 33, 9: 34,  # SMGs
            0: 38, 4: 39, 12: 38, 13: 39,  # Assault Rifles
            3: 42, 5: 45,  # Snipers
            10: 37, 11: 41, 14: 40,  # Special weapons
        }
        return barrel_distances.get(weapon_id, 37)
    
    def _get_bullet_spawn_offset(self, weapon_id):
        """Get x,y offset adjustments for bullet spawn (in pixels)"""
        offsets = {
            8: (6, 4),  # UZI: 6px forward, 4px perpendicular down
        }
        return offsets.get(weapon_id, (0, 0))
    
    def get_extended_game_state(self):
        """Package world_data, gun spawns, medkit spawns, player inventories, gas effects, and grenade data for client"""
        # Gun spawn data: [[x, y, weapon_id, is_active], ...]
        spawn_data = self.gun_spawner.get_spawn_data_for_client()
        
        # Medkit spawn data: [[x, y, is_active], ...]
        medkit_data = self.medkit_spawner.get_spawn_data_for_client()
        
        # Player inventory data: [[gun1_id, gun2_id, current_slot], ...] for 8 players
        inventory_data = np.zeros((8, 3), dtype=np.int32)
        for i in range(8):
            gun_ids = self.player_inventories[i].get_gun_ids()
            inventory_data[i, 0] = gun_ids[0]
            inventory_data[i, 1] = gun_ids[1]
            inventory_data[i, 2] = self.player_inventories[i].current_slot
            
            # Update ammo in world_data for current gun
            current_gun = self.player_inventories[i].get_current_gun()
            if current_gun is not None:
                self.world_data[i, 9] = current_gun.current_ammo
                self.world_data[i, 10] = current_gun.total_ammo
            else:
                self.world_data[i, 9] = 0
                self.world_data[i, 10] = 0
        
        # Gas effects data: [[x, y, radius, duration], ...] for all active gas zones
        gas_data = np.array([
            [effect['x'], effect['y'], effect['radius'], effect['duration']]
            for effect in self.gas_effects.values()
        ], dtype=np.float64) if self.gas_effects else np.zeros((0, 4), dtype=np.float64)
        
        # Grenade data per player: [selected_type, frag_count, proxy_count, gas_count]
        grenade_data = self.grenade_data[:8].copy()
        
        return self.world_data, spawn_data, medkit_data, inventory_data, gas_data, grenade_data

    def _update_player_kneel_states(self):
        """Apply kneel input and resize active player hitboxes by 10px in height."""
        for i in range(8):
            if self.world_data[i, 0] == 0:
                self.player_is_kneeling[i] = False
                self.player_half_h[i] = self.PLAYER_HALF_H
                continue

            wants_kneel = self.player_inputs[i, 13] == 1
            if wants_kneel == self.player_is_kneeling[i]:
                continue

            x = self.world_data[i, 1]
            y = self.world_data[i, 2]
            old_half_h = self.player_half_h[i]
            new_half_h = self._get_target_half_h(wants_kneel)

            # Keep feet level when toggling stance by adjusting center y.
            feet_y = y + old_half_h
            new_y = feet_y - new_half_h

            if self.is_rect_colliding_with_obstacle(x, new_y, self.PLAYER_HALF_W, new_half_h):
                # Prevent standing up inside a ceiling.
                if not wants_kneel:
                    continue
                new_y = y

            self.player_is_kneeling[i] = wants_kneel
            self.player_half_h[i] = new_half_h
            self.world_data[i, 2] = new_y

    def run_game(self):
        MAX_BULLET_DIST = config.MAX_BULLET_DISTANCE
        SAW_WEAPON_ID = config.SAW_WEAPON_ID
        SAW_FIRE_DELAY = getattr(config, 'SAW_FIRE_DELAY', 2.0)
        SAW_LIFETIME = config.SAW_LIFETIME
        SAW_EXPLOSION_RADIUS = config.SAW_EXPLOSION_RADIUS
        SAW_EXPLOSION_DAMAGE = config.SAW_EXPLOSION_DAMAGE

        if not hasattr(self, 'saw_bullet_timers'):
            self.saw_bullet_timers = {}

        clock = pygame.time.Clock()
        while True:
            clock.tick(config.SERVER_FPS)
            # Calculate delta_time for cooldowns
            current_time = time.time()
            delta_time = current_time - self.last_frame_time
            self.last_frame_time = current_time

            # Match timer
            if self.match_start_time is None:
                # Timer hasn't started yet - skip timer logic
                self.time_remaining = self.match_duration
            else:
                elapsed = current_time - self.match_start_time
                self.time_remaining = max(0.0, self.match_duration - elapsed)
                if self.time_remaining <= 0:
                    self.time_remaining = 0
                    if not self.match_ended:
                        leaderboard_data = self._build_leaderboard_array()
                        print("[SERVER] Match ended")
                        print("Leaderboard:", leaderboard_data)
                        self.match_ended = True
                    # freeze gameplay
                    continue
            # Decrement grenade cooldowns
            self.player_grenade_cooldown = np.maximum(0, self.player_grenade_cooldown - delta_time)
            
            # Update fire cooldowns
            self.player_fire_cooldown = np.maximum(0, self.player_fire_cooldown - delta_time)
            
            # Update respawn cooldowns and respawn players when ready
            for idx in range(8):
                if self.player_respawn_cooldown[idx] > 0:
                    self.player_respawn_cooldown[idx] -= delta_time
                    if self.player_respawn_cooldown[idx] <= 0:
                        self.player_respawn_cooldown[idx] = 0
                        self.respawn(idx, delay=0)  # instant respawn after timer expires

            self._apply_health_regeneration(delta_time)
            
            self._update_player_kneel_states()

            # Tank movement: left/right using A/D keys (index 1=A, 2=D).
            # Kneeling reduces movement speed.
            movement_speed = self.world_data[:8, 4].copy()
            movement_speed[self.player_is_kneeling] *= self.KNEEL_SPEED_MULTIPLIER
            horizontal_move = (self.player_inputs[:, 2] - self.player_inputs[:, 1]) * movement_speed
            # --- Grenade physics and fuse update ---
            if not hasattr(self, 'grenade_fuse_timers'):
                self.grenade_fuse_timers = {}
            for g in range(48, 55):
                if self.world_data[g, 0] == 1:
                    grenade_type = int(self.world_data[g, 10])
                    blast_radius = self.world_data[g, 6]
                    damage = self.world_data[g, 7]
                    if grenade_type == 2:
                        self._update_non_bouncy_grenade(g, radius=4.0)
                    else:
                        self._update_bouncy_grenade(g, radius=4.0)
                    gx, gy = self.world_data[g, 1], self.world_data[g, 2]

                    if grenade_type == 1:      # normal timed grenade with bounce physics
                        if g in self.grenade_fuse_timers:
                            self.grenade_fuse_timers[g] -= 1.0 / config.SERVER_FPS

                        if g in self.grenade_fuse_timers and self.grenade_fuse_timers[g] <= 0:
                            # explode
                            owner = int(self.world_data[g, 9])
                            for t in range(8):
                                if self.world_data[t, 0] == 1:
                                    distance = self._distance_point_to_player_hitbox(gx, gy, t)
                                    if distance < blast_radius:
                                        dealt = self.grenade_damage(distance, max_damage=damage, radius=blast_radius)
                                        self._apply_damage_to_player(t, dealt, killer_idx=owner)
                            self.world_data[g, 0] = 0
                            del self.grenade_fuse_timers[g]
                            if hasattr(self, 'grenade_stuck_frames'):
                                self.grenade_stuck_frames.pop(g, None)

                    elif grenade_type == 2:  # proxy grenade – arms after delay then explodes on contact
                        # decrement whichever timer is active (arming or lifetime)
                        if g in self.grenade_fuse_timers:
                            self.grenade_fuse_timers[g] -= 1.0 / config.SERVER_FPS
                        # ensure armed-state tracking exists
                        if not hasattr(self, 'proxy_armed'):
                            self.proxy_armed = set()

                        if g in self.proxy_armed:
                            # already armed – timer now represents remaining life
                            if self.grenade_fuse_timers.get(g, 0) <= 0:
                                # lifetime expired without contact, just remove
                                self.world_data[g, 0] = 0
                                self.proxy_armed.discard(g)
                                self.grenade_fuse_timers.pop(g, None)
                                if hasattr(self, 'grenade_stuck_frames'):
                                    self.grenade_stuck_frames.pop(g, None)
                            else:
                                # check for player contact and detonate if seen
                                detonated = False
                                for t in range(8):
                                    if self.world_data[t, 0] == 1:
                                        if self._distance_point_to_player_hitbox(gx, gy, t) < blast_radius / 3.0:
                                            detonated = True
                                            break
                                if detonated:
                                    owner = int(self.world_data[g, 9])
                                    for t in range(8):
                                        if self.world_data[t, 0] == 1:
                                            distance = self._distance_point_to_player_hitbox(gx, gy, t)
                                            if distance < blast_radius:
                                                dealt = self.grenade_damage(distance, max_damage=damage, radius=blast_radius)
                                                self._apply_damage_to_player(t, dealt, killer_idx=owner)
                                    self.world_data[g, 0] = 0
                                    self.proxy_armed.discard(g)
                                    self.grenade_fuse_timers.pop(g, None)
                                    if hasattr(self, 'grenade_stuck_frames'):
                                        self.grenade_stuck_frames.pop(g, None)
                        else:
                            # still in arming phase
                            if g in self.grenade_fuse_timers and self.grenade_fuse_timers[g] <= 0:
                                # move into armed state with 45‑second lifetime
                                self.proxy_armed.add(g)
                                self.grenade_fuse_timers[g] = 45.0
                    elif grenade_type == 3:  # gas grenade - creates persistent damage zone
                        if g in self.grenade_fuse_timers:
                            self.grenade_fuse_timers[g] -= 1.0 / config.SERVER_FPS
                            if self.grenade_fuse_timers[g] <= 0:
                                # Create persistent gas effect zone
                                effect_id = self.gas_effect_counter
                                self.gas_effect_counter += 1
                                self.gas_effects[effect_id] = {
                                    'x': gx,
                                    'y': gy,
                                    'radius': blast_radius,
                                    'damage': damage,
                                    'duration': 12.0,
                                    'owner_id': int(self.world_data[g, 9]),
                                    'source_slot': g
                                }
                                print(f"[SERVER] Gas effect created at ({gx:.1f}, {gy:.1f}) with radius {blast_radius}, damage {damage}/frame")
                                self.world_data[g, 0] = 0
                                del self.grenade_fuse_timers[g]
                                if hasattr(self, 'grenade_stuck_frames'):
                                    self.grenade_stuck_frames.pop(g, None)
            
            # --- Process active gas effects ---
            effects_to_remove = []
            for effect_id, effect in self.gas_effects.items():
                # Decrement gas duration
                effect['duration'] -= 1.0 / config.SERVER_FPS
                
                # Apply damage to players in the gas zone
                for t in range(8):
                    if self.world_data[t, 0] == 1:
                        distance = self._distance_point_to_player_hitbox(effect['x'], effect['y'], t)
                        if distance < effect['radius']:
                            dealt = (effect['damage'] / max(distance, 1.0))
                            killer = int(effect.get('owner_id', -1))
                            if self._apply_damage_to_player(t, dealt, killer_idx=killer):
                                print(f"[SERVER] Player {t} killed by gas grenade")
                
                # Remove effect if duration expired
                if effect['duration'] <= 0:
                    effects_to_remove.append(effect_id)
            
            # Clean up expired gas effects
            for effect_id in effects_to_remove:
                del self.gas_effects[effect_id]
            
            # Check horizontal collisions before moving - allow movement away from obstacles
            for i in range(8):
                if self.world_data[i, 0] == 0:
                    continue
                old_x = self.world_data[i, 1]
                new_x = old_x + horizontal_move[i]
                
                # Check if currently colliding
                currently_colliding = self.is_player_colliding_with_obstacle(old_x, self.world_data[i, 2], i)
                will_collide = self.is_player_colliding_with_obstacle(new_x, self.world_data[i, 2], i)
                
                # Only accept horizontal movement when destination is non-colliding.
                if not will_collide:
                    self.world_data[i, 1] = new_x
            
            # Aim control using all 4 arrow keys for continuous rotation
            # UP=3, DOWN=4, LEFT=5, RIGHT=6
            AIM_ROTATION_SPEED = config.AIM_ROTATION_SPEED
            
            # Handle gun switching (S key = input[9]) - only on key press (rising edge)
            for i in range(8):
                if self.world_data[i, 0] == 0:
                    continue
                # Only switch if S key is pressed now but wasn't pressed last frame
                if self.player_inputs[i, 9] == 1 and self.previous_inputs[i, 9] == 0:
                    self.player_inventories[i].switch_gun()
            
            # Handle reload (R key = input[8])
            for i in range(8):
                if self.world_data[i, 0] == 0:
                    continue
                if self.player_inputs[i, 8] == 1:  # R key pressed
                    weapon = self.player_inventories[i].get_current_gun()
                    if weapon is None:
                        continue
                    # Only start reload if not already reloading and if reload is needed
                    if self.player_reload_cooldown[i] <= 0 and weapon.current_ammo < weapon.magazine_capacity and weapon.total_ammo > 0:
                        self.player_reload_cooldown[i] = weapon.reload_time
            
            for i in range(8):
                if self.world_data[i, 0] == 0:
                    continue
                
                # Continuous rotation based on arrow keys
                if self.player_inputs[i, 5] == 1:  # LEFT arrow
                    self.world_data[i, 3] -= AIM_ROTATION_SPEED
                if self.player_inputs[i, 6] == 1:  # RIGHT arrow
                    self.world_data[i, 3] += AIM_ROTATION_SPEED
                if self.player_inputs[i, 3] == 1:  # UP arrow
                    self.world_data[i, 3] -= AIM_ROTATION_SPEED
                if self.player_inputs[i, 4] == 1:  # DOWN arrow
                    self.world_data[i, 3] += AIM_ROTATION_SPEED

            # Jetpack system (W key = keyboard_input[0])
            jetpack_active = self.player_inputs[:, 0].astype(bool)
            for i in range(8):
                if self.world_data[i, 0] == 0:  # skip inactive players
                    continue

                if self.player_is_kneeling[i]:
                    # While kneeling, disable flying. Only allow a small jump on W press.
                    if jetpack_active[i] and self.previous_inputs[i, 0] == 0:
                        ground_y = self.find_ground_below(self.world_data[i, 1], self.world_data[i, 2], i)
                        on_ground = (
                            ground_y is not None
                            and abs(self.world_data[i, 2] - ground_y) <= self.KNEEL_GROUND_TOLERANCE
                            and self.player_vy[i] >= 0
                        )
                        if on_ground:
                            self.player_vy[i] = -self.KNEEL_JUMP_IMPULSE
                    # Recharge fuel while kneeling.
                    self.player_fuel[i] = min(self.MAX_FUEL, self.player_fuel[i] + self.FUEL_RECHARGE)
                elif jetpack_active[i] and self.player_fuel[i] > 0:
                    # Apply upward thrust
                    self.player_vy[i] -= self.JETPACK_THRUST
                    # Consume fuel
                    self.player_fuel[i] = max(0, self.player_fuel[i] - self.FUEL_CONSUMPTION)
                else:
                    # Recharge fuel when not using jetpack
                    self.player_fuel[i] = min(self.MAX_FUEL, self.player_fuel[i] + self.FUEL_RECHARGE)
                # Update fuel in world_data
                self.world_data[i, 6] = self.player_fuel[i]

            # Apply gravity and vertical movement with obstacle collision
            for i in range(8):
                if self.world_data[i, 0] == 0:
                    continue
                
                # Apply gravity
                self.player_vy[i] += self.GRAVITY
                old_y = self.world_data[i, 2]
                new_y = old_y + self.player_vy[i]
                
                # Check collision with obstacles below
                if self.player_vy[i] > 0:  # falling down
                    ground_y = self.find_ground_below(self.world_data[i, 1], old_y, i)
                    if ground_y is not None and new_y >= ground_y:
                        self.world_data[i, 2] = ground_y
                        self.player_vy[i] = 0
                    else:
                        self.world_data[i, 2] = new_y
                else:  # moving up
                    # Check collision when moving up
                    currently_colliding = self.is_player_colliding_with_obstacle(self.world_data[i, 1], old_y, i)
                    will_collide = self.is_player_colliding_with_obstacle(self.world_data[i, 1], new_y, i)
                    
                    if not will_collide:
                        # Only allow upward movement into free space.
                        self.world_data[i, 2] = new_y
                    else:
                        self.player_vy[i] = 0
            
            # Clamp players to the world bounds using their active hitbox height.
            for i in range(8):
                if self.world_data[i, 0] == 0:
                    continue
                half_h = self._get_player_half_h(i)
                self.world_data[i, 1] = np.clip(self.world_data[i, 1], self.PLAYER_HALF_W, self.SCREEN_W - self.PLAYER_HALF_W)
                self.world_data[i, 2] = max(self.world_data[i, 2], half_h)

            # Players that fall out below the map die and immediately respawn.
            for i in range(8):
                if self.world_data[i, 0] == 0:
                    continue
                fall_kill_y = self.SCREEN_H + self._get_player_half_h(i)
                if self.world_data[i, 2] > fall_kill_y:
                    self._record_player_death(i, killer_idx=i)

            # Resolve any tank that is still embedded in terrain.
            for i in range(8):
                if self.world_data[i, 0] == 0:
                    continue

                x = self.world_data[i, 1]
                y = self.world_data[i, 2]
                if not self.is_player_colliding_with_obstacle(x, y, i):
                    self.player_stuck_frames[i] = 0
                    continue

                # Try a local push-out first to preserve nearby position.
                push_x = horizontal_move[i]
                push_y = -1.0 if self.player_vy[i] >= 0 else self.player_vy[i]
                x, y = self._push_player_out_of_obstacle(x, y, player_idx=i, push_x=push_x, push_y=push_y)
                x = float(np.clip(x, self.PLAYER_HALF_W, self.SCREEN_W - self.PLAYER_HALF_W))
                y = float(np.clip(y, self._get_player_half_h(i), self.SCREEN_H - self._get_player_half_h(i)))
                self.world_data[i, 1] = x
                self.world_data[i, 2] = y

                if self.is_player_colliding_with_obstacle(x, y, i):
                    self.player_stuck_frames[i] += 1
                    if self.player_stuck_frames[i] > 10:
                        spawn_x, spawn_y = self._get_safe_spawn_position()
                        self.world_data[i, 1] = spawn_x
                        self.world_data[i, 2] = spawn_y
                        self.player_vy[i] = 0
                        self.player_stuck_frames[i] = 0
                else:
                    self.player_stuck_frames[i] = 0

            # Store bullet old positions before movement for interpolated collision detection
            bullet_old_positions = np.zeros((40, 2), dtype=np.float64)  # 40 bullets, (x, y)
            for b in range(8, 48):
                if self.world_data[b, 0] == 1:
                    bullet_old_positions[b-8, 0] = self.world_data[b, 1]
                    bullet_old_positions[b-8, 1] = self.world_data[b, 2]

            # Move bullets and check collisions
            for b in range(8, 48):
                if self.world_data[b, 0] == 1:  # if bullet is active
                    bullet_weapon_id = int(self.world_data[b, 10])
                    old_x, old_y = self.world_data[b, 1], self.world_data[b, 2]
                    self.world_data[b, 1] += np.cos(self.world_data[b, 3]) * self.world_data[b, 4]
                    self.world_data[b, 2] += np.sin(self.world_data[b, 3]) * self.world_data[b, 4]
                    self.world_data[b, 5] += self.world_data[b, 4]

                    # Check screen boundary reflection first
                    new_x, new_y = self.world_data[b, 1], self.world_data[b, 2]
                    reflected = False

                    # Left boundary (x <= 0)
                    if new_x <= 0:
                        self.world_data[b, 3] = np.pi - self.world_data[b, 3]  # Reflect horizontally
                        self.world_data[b, 1] = -new_x  # Bounce back
                        reflected = True

                    # Right boundary (x >= SCREEN_W)
                    elif new_x >= self.SCREEN_W:
                        self.world_data[b, 3] = np.pi - self.world_data[b, 3]  # Reflect horizontally
                        self.world_data[b, 1] = 2 * self.SCREEN_W - new_x  # Bounce back
                        reflected = True

                    # Top boundary (y <= 0)
                    if new_y <= 0:
                        self.world_data[b, 3] = -self.world_data[b, 3]  # Reflect vertically
                        self.world_data[b, 2] = -new_y  # Bounce back
                        reflected = True

                    # Bottom boundary (y >= SCREEN_H)
                    elif new_y >= self.SCREEN_H:
                        self.world_data[b, 3] = -self.world_data[b, 3]  # Reflect vertically
                        self.world_data[b, 2] = 2 * self.SCREEN_H - new_y  # Bounce back
                        reflected = True

                    # If reflected, update position with new angle
                    if reflected:
                        self.world_data[b, 1] = old_x + np.cos(self.world_data[b, 3]) * self.world_data[b, 4]
                        self.world_data[b, 2] = old_y + np.sin(self.world_data[b, 3]) * self.world_data[b, 4]

                    if self.world_data[b, 5] > self.world_data[b, 4]:
                        new_x, new_y = self.world_data[b, 1], self.world_data[b, 2]

                        # sample along path to detect obstacle hit
                        steps = 5
                        hit = False
                        for i in range(steps + 1):
                            t = i / steps
                            check_x = old_x + (new_x - old_x) * t
                            check_y = old_y + (new_y - old_y) * t
                            if self.is_colliding_with_obstacle(check_x, check_y, 2):
                                hit = True
                                break
                        if hit:
                            if bullet_weapon_id == SAW_WEAPON_ID:
                                # Mirror-like reflection: preserve speed, reflect angle by hit axis.
                                hit_x = self.is_colliding_with_obstacle(new_x, old_y, 2)
                                hit_y = self.is_colliding_with_obstacle(old_x, new_y, 2)

                                if hit_x and not hit_y:
                                    self.world_data[b, 3] = np.pi - self.world_data[b, 3]
                                elif hit_y and not hit_x:
                                    self.world_data[b, 3] = -self.world_data[b, 3]
                                else:
                                    self.world_data[b, 3] += np.pi

                                self.world_data[b, 1] = old_x + np.cos(self.world_data[b, 3]) * self.world_data[b, 4]
                                self.world_data[b, 2] = old_y + np.sin(self.world_data[b, 3]) * self.world_data[b, 4]
                            else:
                                self.world_data[b, 0] = 0

                    if bullet_weapon_id == SAW_WEAPON_ID:
                        if b not in self.saw_bullet_timers:
                            self.saw_bullet_timers[b] = SAW_LIFETIME
                        self.saw_bullet_timers[b] -= 1.0 / config.SERVER_FPS
                        if self.saw_bullet_timers[b] <= 0:
                            sx, sy = self.world_data[b, 1], self.world_data[b, 2]
                            owner = int(self.world_data[b, 9])
                            for t in range(8):
                                if self.world_data[t, 0] == 0:
                                    continue
                                distance = self._distance_point_to_player_hitbox(sx, sy, t)
                                if distance < SAW_EXPLOSION_RADIUS:
                                    dealt = self.grenade_damage(
                                        distance,
                                        max_damage=SAW_EXPLOSION_DAMAGE,
                                        radius=SAW_EXPLOSION_RADIUS
                                    )
                                    self._apply_damage_to_player(t, dealt, killer_idx=owner)
                            self.world_data[b, 0] = 0
                            self.saw_bullet_timers.pop(b, None)
                    else:
                        self.saw_bullet_timers.pop(b, None)
                    if bullet_weapon_id == config.ROCKET_LAUNCHER_ID:
                        #explodes on any collision player or wall, so check walls first then player collisions
                        sx, sy = self.world_data[b, 1], self.world_data[b, 2]
                        owner = int(self.world_data[b, 9])
                        
                        # Check for screen boundary collision
                        screen_hit = False
                        if sx <= 0 or sx >= self.SCREEN_W or sy <= 0 or sy >= self.SCREEN_H:
                            screen_hit = True
                        
                        # Check for wall collision along path
                        steps = 5
                        wall_hit = False
                        player_hit = False
                        for i in range(steps + 1):
                            sample_t = i / steps
                            check_x = old_x + (new_x - old_x) * sample_t
                            check_y = old_y + (new_y - old_y) * sample_t
                            if self.is_colliding_with_obstacle(check_x, check_y, 2):
                                wall_hit = True
                                sx, sy = check_x, check_y
                                break

                            # Trigger on collision with other players only (owner contact should not trigger).
                            for player_idx in range(8):
                                if self.world_data[player_idx, 0] == 0:
                                    continue
                                if player_idx == owner:
                                    continue
                                if self._point_hits_player_hitbox(check_x, check_y, player_idx, padding=2.0):
                                    player_hit = True
                                    sx, sy = check_x, check_y
                                    break
                            if player_hit:
                                break
                        # Apply AoE explosion damage
                        if wall_hit or player_hit or screen_hit:
                            for t in range(8):
                                if self.world_data[t, 0] == 0:
                                    continue
                                distance = self._distance_point_to_player_hitbox(sx, sy, t)
                                if distance < config.ROCKET_EXPLOSION_RADIUS:
                                    dealt = self.grenade_damage(
                                        distance,
                                        max_damage=config.ROCKET_EXPLOSION_DAMAGE,
                                        radius=config.ROCKET_EXPLOSION_RADIUS
                                    )
                                    self._apply_damage_to_player(t, dealt, killer_idx=owner)
                        
                        # Deactivate rocket if it hit something
                        if wall_hit or player_hit or screen_hit:
                            self.world_data[b, 0] = 0

            # Deactivate bullets based on the firing weapon's effective range.
            for b in range(8, 48):
                if self.world_data[b, 0] != 1:
                    continue
                weapon_id = int(self.world_data[b, 10])
                effective_range = MAX_BULLET_DIST
                if weapon_id in WEAPONS:
                    effective_range = float(WEAPONS[weapon_id].effective_range)
                if self.world_data[b, 5] > effective_range:
                    self.world_data[b, 0] = 0

            # Cleanup timers for any saw bullets that were deactivated this frame.
            for b in list(self.saw_bullet_timers.keys()):
                if self.world_data[b, 0] == 0:
                    self.saw_bullet_timers.pop(b, None)

            # Update gun spawner
            self.gun_spawner.update(delta_time)
            
            # Check for gun pickups
            # Gun pickup using P key (input index 12)
            # Gun pickup using P key (index 12) - only on press, not while held
            for i in range(8):
                if self.world_data[i, 0] == 1:
                    if self.player_inputs[i, 12] == 1 and self.previous_inputs[i, 12] == 0:

                        player_x = self.world_data[i, 1]
                        player_y = self.world_data[i, 2]

                        spawn_index = self.gun_spawner.get_nearby_gun(player_x, player_y)

                        if spawn_index is not None:
                            weapon_id = self.gun_spawner.pickup_gun(spawn_index)

                            if weapon_id is not None:
                                self.player_inventories[i].pickup_gun(weapon_id)
                                      
            # Update medkit spawner
            self.medkit_spawner.update(delta_time)
            
            # Check for medkit pickups
            for i in range(8):
                if self.world_data[i, 0] == 1:  # Player is alive
                    # Do not consume medkits when already at full health.
                    if self.world_data[i, 7] >= self.MAX_HEALTH:
                        continue

                    player_x = self.world_data[i, 1]
                    player_y = self.world_data[i, 2]
                    picked_medkit = self.medkit_spawner.check_pickup(player_x, player_y)
                    if picked_medkit:
                        # Heal player to full health
                        self.world_data[i, 7] = self.MAX_HEALTH
                                
            # Update reload cooldowns and complete reloads when finished
            for i in range(8):
                if self.player_reload_cooldown[i] > 0:
                    self.player_reload_cooldown[i] -= delta_time
                    if self.player_reload_cooldown[i] <= 0:
                        # Reload is complete
                        self.player_reload_cooldown[i] = 0
                        weapon = self.player_inventories[i].get_current_gun()
                        if weapon is not None:
                            weapon.reload()

            # create bullets (space = index 7)
            for idx in range(8):
                # Cancel SAW charge when player is dead.
                if self.world_data[idx, 0] != 1:
                    self.saw_charge_end_time[idx] = 0.0
                    continue
                current_weapon = self.player_inventories[idx].get_current_gun()
                # Cancel SAW charge if player switched away from SAW.
                if current_weapon is None or current_weapon.gun_id != SAW_WEAPON_ID:
                    self.saw_charge_end_time[idx] = 0.0
                    continue

                # Tap-to-charge: start SAW charge only on shoot key rising edge.
                if (
                    self.player_inputs[idx, 7] == 1
                    and self.previous_inputs[idx, 7] == 0
                    and self.saw_charge_end_time[idx] <= 0.0
                    and self.player_reload_cooldown[idx] <= 0
                    and self.player_fire_cooldown[idx] <= 0
                    and current_weapon.can_shoot()
                ):
                    self.saw_charge_end_time[idx] = current_time + SAW_FIRE_DELAY

            shooting_id = np.where(self.player_inputs[:, 7] == 1)[0]
            saw_ready_id = np.where(
                (self.saw_charge_end_time > 0.0)
                & (self.saw_charge_end_time <= current_time)
            )[0]
            shooting_id = np.unique(np.concatenate((shooting_id, saw_ready_id)))
            for idx in shooting_id:
                if self.world_data[idx, 0] != 1:
                    continue
                weapon = self.player_inventories[idx].get_current_gun()
                
                if weapon is None:
                    continue
                
                # Check if currently reloading
                if self.player_reload_cooldown[idx] > 0:
                    if weapon.gun_id == SAW_WEAPON_ID:
                        self.saw_charge_end_time[idx] = 0.0
                    continue
                
                # Check fire cooldown and ammo
                if self.player_fire_cooldown[idx] > 0:
                    continue

                if weapon.gun_id == SAW_WEAPON_ID:
                    if self.saw_charge_end_time[idx] <= 0.0:
                        self.saw_charge_end_time[idx] = current_time + SAW_FIRE_DELAY
                        continue
                    if current_time < self.saw_charge_end_time[idx]:
                        continue
                
                if not weapon.can_shoot():
                    self.saw_charge_end_time[idx] = 0.0
                    # Auto reload if out of ammo
                    if weapon.total_ammo > 0:
                        self.player_reload_cooldown[idx] = weapon.reload_time
                    continue
                
                # Shoot weapon
                weapon.shoot()
                    
                id = idx * 5 + 8
                
                # Spawn bullets based on rpf (rounds per fire)
                bullets_spawned = 0
                spawned_bullet_slots = []
                for i in range(weapon.rpf):
                    free_slots = np.where(self.world_data[id:id+5, 0] == 0)[0]
                    if len(free_slots) > 0:
                        bullet_index = free_slots[0]
                        
                        # Calculate bullet angle with spread
                        bullet_angle = weapon.get_bullet_angle_with_spread(self.world_data[idx, 3])
                        
                        # Get proper barrel distance for this weapon
                        barrel_dist = self._get_barrel_distance(weapon.gun_id)
                        offset_x, offset_y = self._get_bullet_spawn_offset(weapon.gun_id)
                        
                        # Calculate spawn position at gun barrel with offset
                        spawn_x = self.world_data[idx, 1] + barrel_dist * np.cos(bullet_angle)
                        spawn_y = self.world_data[idx, 2] + barrel_dist * np.sin(bullet_angle)
                        
                        # Apply perpendicular offset if needed
                        if offset_x != 0 or offset_y != 0:
                            perp_angle = bullet_angle + np.pi / 2
                            spawn_x += offset_x * np.cos(bullet_angle) + offset_y * np.cos(perp_angle)
                            spawn_y += offset_x * np.sin(bullet_angle) + offset_y * np.sin(perp_angle)
                        
                        self.world_data[id+bullet_index, 0] = 1
                        self.world_data[id+bullet_index, 1] = spawn_x
                        self.world_data[id+bullet_index, 2] = spawn_y
                        self.world_data[id+bullet_index, 3] = bullet_angle
                        # Set bullet speed from weapon
                        self.world_data[id+bullet_index, 4] = weapon.bullet_speed
                        # reset traveled distance
                        self.world_data[id+bullet_index, 5] = 0
                        # set bullet damage from weapon
                        self.world_data[id+bullet_index, 7] = weapon.damage
                        # set bullet owner for scoring
                        self.world_data[id+bullet_index, 9] = idx
                        # store weapon id so SAW/Rocket logic can identify bullet type
                        self.world_data[id+bullet_index, 10] = weapon.gun_id
                        spawned_bullet_slots.append(id + bullet_index)
                        bullets_spawned += 1
                
                if bullets_spawned > 0:
                    self.saw_charge_end_time[idx] = 0.0
                    # Set cooldown based on weapon's rate of fire
                    self.player_fire_cooldown[idx] = weapon.rate_of_fire
                    # Immediately sync ammo to world_data after shooting
                    self.world_data[idx, 9] = weapon.current_ammo
                    self.world_data[idx, 10] = weapon.total_ammo
                    
                    # Update bullet_old_positions for newly spawned bullets to prevent false collision checks
                    for bullet_slot in spawned_bullet_slots:
                        if bullet_slot < 48 and self.world_data[bullet_slot, 0] == 1:
                            # Start sweep at shooter center so close-range targets between muzzle and bullet don't get skipped.
                            bullet_old_positions[bullet_slot-8, 0] = self.world_data[idx, 1]
                            bullet_old_positions[bullet_slot-8, 1] = self.world_data[idx, 2]

            # detect collisions
            # BULLET -> TANK collisions: apply damage, credit score, respawn on death
            for b in range(8, 48):
                if self.world_data[b, 0] == 0:
                    continue
                bullet_weapon_id = int(self.world_data[b, 10])
                # bullet position
                bx, by = self.world_data[b, 1], self.world_data[b, 2]
                old_bx = bullet_old_positions[b - 8, 0]
                old_by = bullet_old_positions[b - 8, 1]
                
                if bullet_weapon_id in WEAPONS:
                    damage = WEAPONS[bullet_weapon_id].damage
                else:
                    damage = self.world_data[b, 7]
                owner = int(self.world_data[b, 9])
                
                bullet_hit = False
                for t in range(8):
                    if self.world_data[t, 0] == 0:
                        continue
                    if t == owner and bullet_weapon_id != SAW_WEAPON_ID:
                        continue
                    if (
                        bullet_weapon_id == SAW_WEAPON_ID
                        and t == owner
                        and self.world_data[b, 5] < config.SAW_SELF_HIT_ARM_DISTANCE
                    ):
                        # Prevent instant self-kill right at launch.
                        continue
                    if self._segment_hits_player_hitbox(old_bx, old_by, bx, by, t, padding=2.0):
                        if bullet_weapon_id == SAW_WEAPON_ID:
                            # Saw projectile pierces and kills everything it touches.
                            self._record_player_death(t, killer_idx=owner)
                            continue
                        else:
                            # Normal hit - apply damage once and consume bullet.
                            self._apply_damage_to_player(t, damage, killer_idx=owner)
                            bullet_hit = True
                            break
                
                # remove bullet after processing all potential hits
                if bullet_hit:
                    self.world_data[b, 0] = 0
            
            # Sync all player ammo to world_data at end of frame (after all shooting/reloading)
            for i in range(8):
                if self.world_data[i, 0] == 1:
                    current_gun = self.player_inventories[i].get_current_gun()
                    if current_gun is not None:
                        self.world_data[i, 9] = current_gun.current_ammo
                        self.world_data[i, 10] = current_gun.total_ammo
                    else:
                        # Player has no gun - set ammo to 0
                        self.world_data[i, 9] = 0
                        self.world_data[i, 10] = 0
            
            # Handle grenade type cycling (C key = input 11) on rising edge
            for idx in range(8):
                if self.world_data[idx, 0] == 0:
                    continue
                if self.player_inputs[idx, 11] == 1 and self.previous_inputs[idx, 11] == 0:
                    if self.grenade_data[idx, 0] == 3:
                        self.grenade_data[idx, 0] = 1
                    else:
                        self.grenade_data[idx, 0] += 1
            
            # Handle grenade throws (G key = input 10)
            for idx in range(8):
                if self.world_data[idx, 0] == 0:
                    continue
                # Check for grenade throw input (G key = index 10)
                if self.player_inputs[idx, 10] == 1 and self.player_grenade_cooldown[idx] <= 0:
                    throw_angle = self.world_data[idx, 3]
                    throw_power = 15  # Adjust as needed
                    grenade_id = int(self.grenade_data[idx, 0])  # Get grenade type from grenade_data array
                    if self.grenade_data[idx, grenade_id] <= 0:
                        continue  # No grenades of this type available
                    self.grenade_data[idx, grenade_id] -= 1
                    self.throw_grenade(idx, grenade_id, throw_angle, throw_power)
                    self.player_grenade_cooldown[idx] = 2.0  # 2 seconds cooldown

            # Store current inputs for next frame's edge detection
            self.previous_inputs = self.player_inputs.copy()

    def respawn(self, tank_index, delay=0.0):
        # spawn at a random x and find ground below
        # Non-blocking respawn: set timer instead of sleeping
        if delay > 0:
            self.player_respawn_cooldown[tank_index] = delay
            self.world_data[tank_index, 0] = 0  # mark as dead during respawn delay
            return

        spawn_x, spawn_y = self._get_safe_spawn_position()
        self.world_data[tank_index, 1] = spawn_x
        self.world_data[tank_index, 2] = spawn_y
        self.world_data[tank_index, 0] = 1  # mark as alive
        secondary_default = getattr(config, 'DEFAULT_SECONDARY_WEAPON', 2)
        self.player_inventories[tank_index] = PlayerInventory(
            starting_weapon_id=config.get_random_starting_weapon(),
            secondary_weapon_id=secondary_default,
        )
        if hasattr(self, 'player_is_kneeling'):
            self.player_is_kneeling[tank_index] = False
        if hasattr(self, 'player_half_h'):
            self.player_half_h[tank_index] = self.PLAYER_HALF_H
        # reset vertical velocity
        if hasattr(self, 'player_vy'):
            self.player_vy[tank_index] = 0
        if hasattr(self, 'player_stuck_frames'):
            self.player_stuck_frames[tank_index] = 0
        # reset fuel to full
        if hasattr(self, 'player_fuel'):
            self.player_fuel[tank_index] = self.MAX_FUEL
            self.world_data[tank_index, 6] = self.MAX_FUEL
        # reset health on respawn
        if hasattr(self, 'MAX_HEALTH'):
            self.world_data[tank_index, 7] = self.MAX_HEALTH
        # reset grenade selection and counts on respawn
        if hasattr(self, 'grenade_data'):
            self.grenade_data[tank_index, 0] = 1
            self.grenade_data[tank_index, 1] = config.FRAG_GRENADE_COUNT
            self.grenade_data[tank_index, 2] = config.PROXY_GRENADE_COUNT
            self.grenade_data[tank_index, 3] = config.GAS_GREANADE_COUNT

    def add_players(self):
        while True:
            conn, address = self.server_socket.accept()

            with self.player_lock:
                available_ids = np.where(self.world_data[:8, 0] == 0)[0]
                if len(available_ids) == 0:
                    print("server full")
                    continue

                player_id = available_ids[0]
                self.world_data[player_id, 0] = -1  # temporarily reserve slot

            thread = threading.Thread(
                target=self.player_handler,
                args=(conn, player_id),
                daemon=True
            )
            thread.start()
            self.player_count += 1
            

    def player_handler(self, conn, player_id):
        try:
            data = conn.recv(16)
            name = data.decode("utf-8").strip("\x00")
            self.player_names[player_id] = name
            print("[LOG]", name, "connected to the server.")

            conn.send(int(player_id).to_bytes(4, byteorder='little'))

            # Send collision map dimensions and data
            map_info = np.array([self.GRID_W, self.GRID_H, self.GRID_SIZE], dtype=np.int32)
            conn.send(map_info.tobytes())
            conn.send(self.collision_map_bytes)

            self.world_data[player_id, 0] = 1
            self.respawn(player_id, delay=0)  # instant spawn when joining game

            # Start match timer when first player connects
            if self.match_start_time is None:
                self.match_start_time = time.time()
                print("[SERVER] Match timer started - first player connected")

            while True:
                data = conn.recv(16)  # allow for up to 16 bytes (14 bools = 14 bytes, with headroom)
                if not data:
                    break

                player_input = np.frombuffer(data, dtype=bool)
                # Robustly pad or truncate to 14 inputs.
                if len(player_input) < 14:
                    padded_input = np.zeros(14, dtype=bool)
                    padded_input[:len(player_input)] = player_input
                    player_input = padded_input
                elif len(player_input) > 14:
                    player_input = player_input[:14]
                self.player_inputs[player_id] = player_input.astype(int)

                # Prepare extended game state
                world_data, spawn_data, medkit_data, inventory_data, gas_data, grenade_data = self.get_extended_game_state()
                leaderboard_data = self._build_leaderboard_array()

                # Send all data with explicit length header for robust client parsing
                world_bytes = world_data.tobytes()
                spawn_bytes = spawn_data.tobytes()
                medkit_bytes = medkit_data.tobytes()
                gas_bytes = gas_data.tobytes()
                grenade_bytes = grenade_data.tobytes()
                inventory_bytes = inventory_data.tobytes()
                leaderboard_bytes = leaderboard_data.tobytes()
                header_bytes = np.array(
                    [len(spawn_bytes), len(medkit_bytes), len(gas_bytes), len(grenade_bytes), len(leaderboard_bytes), 8],
                    dtype=np.int32
                ).tobytes()
                names_bytes = ("|".join(self.player_names)).encode()
                names_bytes = names_bytes.ljust(128, b'\x00')
                timer_bytes = np.array([self.time_remaining], dtype=np.float64).tobytes()

                # Packet layout:
                # [world_data float64 fixed size]
                # [header int32*5 -> spawn_len, medkit_len, gas_len, grenade_len, leaderboard_len]
                # [spawn_data bytes]
                # [medkit_data bytes]
                # [gas_data bytes]
                # [grenade_data bytes]
                # [inventory_data int32 fixed size]
                # [names bytes fixed size]
                # [leaderboard_data int32 variable size]
                conn.sendall(
                    world_bytes
                    + header_bytes
                    + spawn_bytes
                    + medkit_bytes
                    + gas_bytes
                    + grenade_bytes
                    + inventory_bytes
                    + names_bytes
                    + leaderboard_bytes
                    + timer_bytes
                )
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass
        finally:
            self.world_data[player_id, 0] = 0
            if hasattr(self, 'player_vy'):
                self.player_vy[player_id] = 0
            self.player_names[player_id] = ""
            try:
                conn.close()
            except OSError:
                pass


            
if __name__ == "__main__":
    Server()
    print("program concluded")
