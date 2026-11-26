import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

from simulation import (
    Robot, Person, Cyclist, Litter, Car, PublicPlace, Tree, Building,
    build_lidar_obstacles,
)


class RobotEnvWaypoints(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    WIDTH, HEIGHT = 800, 600
    BIKE_PATH_Y = 150
    PIXELS_PER_METER = 20
    METERS_PER_PIXEL = 1.0 / PIXELS_PER_METER

    WAYPOINTS = [(100, 100), (700, 100), (700, 500),
                 (100, 500), (400, 300), (100, 500)]

    MAX_LINEAR_SPEED = 0.5
    MAX_ANGULAR_SPEED = 2.0

    COLLECT_RADIUS = 30
    COLLISION_RADIUS = 12
    ROBOT_RADIUS = 13.0
    FENCE_MARGIN = 12.0

    WAYPOINT_RADIUS = 30
    LITTER_MAGNET_RADIUS = 100

    # Награды
    R_STEP = -0.2
    R_PROGRESS = 0.2
    R_WAYPOINT = 80.0
    R_ROUTE_DONE = 100.0
    R_LITTER = 150.0
    R_ALL_LITTER = 500.0
    R_COLLISION_STATIC = -30.0
    R_COLLISION_DYNAMIC = -50.0
    R_BOUNDARY = -10.0

    DEFAULT_MAX_STEPS = 8000

    def __init__(self, render_mode=None, max_steps=None, num_waypoints=None,
                 stage=1, fsm_enabled=True, reward_kwargs=None, env_kwargs=None):
        super().__init__()
        self.render_mode = render_mode
        self._prev_fsm_state = None
        self._prev_movement_mode = None

        # Базовая конфигурация
        cfg = {
            'wp': 6,
            'wp_radius': 40,
            'num_cars': 1,
            'num_places': 1,
            'num_trees': 1,
            'num_people': 0,
            'num_cyclists': 0,
            'num_buildings': 0,
            'litters_range': (1, 2),
            'max_steps': 10000,
            'fsm_enabled': fsm_enabled,
        }

        # Автоматическая настройка по stage
        if stage == 2:
            cfg.update({
                'num_cars': 3,
                'num_places': 3,
                'num_trees': 5,
                'num_people': 3,
                'num_cyclists': 1,
                'num_buildings': 3,
                'litters_range': (4, 6),
            })


        if env_kwargs:
            cfg.update(env_kwargs)

        self.cfg = cfg
        self.fsm_enabled = fsm_enabled

        self.max_steps = max_steps if max_steps is not None else cfg.get('max_steps', 2000)
        n_wp = num_waypoints if num_waypoints is not None else cfg.get('wp', 6)
        self.route = list(self.WAYPOINTS[:n_wp])
        self.waypoint_radius = cfg.get('wp_radius', 40)

        self.stage = stage
        self.NUM_PEOPLE = cfg.get('num_people', 0)
        self.NUM_CYCLISTS = cfg.get('num_cyclists', 0)
        self.NUM_LITTERS = 2
        self.NUM_CARS = cfg.get('num_cars', 1)
        self.NUM_PLACES = cfg.get('num_places', 1)
        self.NUM_TREES = cfg.get('num_trees', 1)
        self.NUM_BUILDINGS = cfg.get('num_buildings', 0)
        self.litters_range = cfg.get('litters_range', (1, 2))

        self.OBS_SIZE = 85

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(self.OBS_SIZE,), dtype=np.float32)

        self.robot = None
        self.people, self.cyclists, self.litters = [], [], []
        self.cars, self.places, self.trees = [], [], []
        self.wp_idx = 0
        self.phase = 'route'
        self.step_count = 0
        self.prev_target_dist = None
        self.prev_litter_dist = None
        self.prev_wp_dist = None
        self.initial_litters = 1

        self.screen = None
        self.clock = None
        self.font = None

        self.stats = {
            'litter_collected': 0,
            'collisions_static': 0,
            'collisions_dynamic': 0,
            'distance_traveled': 0.0,
            'episodes_completed': 0,
            'total_reward': 0.0,
            'waypoints_completed': 0,
            'circles_completed': 0,
        }
        self._prev_robot_x = 0.0
        self._prev_robot_y = 0.0

        self.REAL_MAX_SPEED = 0.5
        self.LINEAR_ACCELERATION = 0.15
        self.ANGULAR_ACCELERATION = 0.5
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0

        self.FRICTION = 0.85
        self.buildings = []
        self.obstacles = []

        self.turning_to_target = False
        self.target_angle = 0.0
        self.turn_steps = 0
        self.max_turn_steps = 60

        self.collision_cd = 0
        self.boundary_cd = 0

        self._stuck_x = self.WIDTH // 2
        self._stuck_y = self.HEIGHT // 2
        self._stuck_steps = 0

        self.emergency_ticks = 0
        self._last_recovery_count = 0
        self.debug_log = True

        if self.fsm_enabled:
            from hybrid_fsm import HybridFSMController
            self.fsm = HybridFSMController(self)
        else:
            self.fsm = None

        if reward_kwargs is not None:
            for key, value in reward_kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, float(value))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        cfg = self.cfg

        if 'litters_range' in cfg:
            self.NUM_LITTERS = int(self.np_random.integers(
                cfg['litters_range'][0], cfg['litters_range'][1] + 1))
        else:
            self.NUM_LITTERS = cfg.get('litters', 2)

        if hasattr(self, 'logger'):
            self.logger.start_episode()

        self.step_count = 0
        self.wp_idx = 0
        self.phase = 'route'
        self.prev_target_dist = None
        self.prev_litter_dist = None
        self.prev_wp_dist = None
        self._prev_visible_count = 0
        self.collision_cd = 0
        self.boundary_cd = 0
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0

        self.turning_to_target = False
        self.target_angle = 0.0
        self.turn_steps = 0
        self.no_progress_steps = 0

        self.emergency_ticks = 0
        self._last_recovery_count = 0

        for key in self.stats:
            if key != 'episodes_completed':
                self.stats[key] = 0.0 if key == 'distance_traveled' else 0

        # Здания
        self.buildings = []
        building_configs = [
            ((100, 300), (200, 350), (80, 120), (80, 120)),
            ((500, 650), (200, 350), (80, 120), (80, 120)),
            ((250, 500), (360, 420), (80, 120), (40, 60)),
        ]

        for i, (xr, yr, wr, hr) in enumerate(building_configs):
            if i >= self.NUM_BUILDINGS:
                break

            for attempt in range(50):
                x = float(self.np_random.uniform(xr[0], xr[1]))
                y = float(self.np_random.uniform(yr[0], yr[1]))
                w = float(self.np_random.uniform(wr[0], wr[1]))
                h = float(self.np_random.uniform(hr[0], hr[1]))
                if y + h > 480 or y < 180:
                    continue
                if x + w > 680 and y + h > 380:
                    continue
                if x < 40 or x + w > 760 or y < 40 or y + h > 560:
                    continue

                polygon = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                building = Building(polygon)

                valid = True
                for other_b in self.buildings:
                    if math.hypot(building.x - other_b.x, building.y - other_b.y) < 150:
                        valid = False
                        break
                if valid:
                    self.buildings.append(building)
                    break

        # Робот
        robot_x, robot_y = self.WIDTH // 2, self.HEIGHT // 2
        min_safe_distance = 80
        for attempt in range(200):
            test_x = float(self.np_random.uniform(150, 650))
            test_y = float(self.np_random.uniform(150, 450))
            too_close_to_building = False
            for b in self.buildings:
                if b.contains_point(test_x, test_y) or b.distance_to_point(test_x, test_y) < min_safe_distance:
                    too_close_to_building = True
                    break
            if too_close_to_building:
                continue
            margin = 50
            if test_x < margin or test_x > self.WIDTH - margin or test_y < margin or test_y > self.HEIGHT - margin:
                continue
            robot_x, robot_y = test_x, test_y
            break

        self.robot = Robot(robot_x, robot_y)
        self._prev_rx = self.robot.x
        self._prev_ry = self.robot.y

        self._stuck_x = self.robot.x
        self._stuck_y = self.robot.y
        self._stuck_steps = 0

        # Объекты
        self.people, self.cyclists, self.litters = [], [], []
        self.cars, self.places, self.trees = [], [], []
        colors = ['red', 'blue', 'green', 'yellow', 'purple']

        # Случайный старт маршрута из первых 3 точек
        base_route = list(self.WAYPOINTS[:cfg['wp']])
        n = len(base_route)
        start_idx = int(self.np_random.integers(0, min(3, n)))
        self.route = base_route[start_idx:] + base_route[:start_idx]
        self.wp_idx = 0
        self.robot.direction = float(self.np_random.uniform(0, 360))

        for _ in range(self.NUM_PEOPLE):
            x, y = self._free_spot(100, 700, 100, 500, min_dist=30,
                                   allow_paths=True, allow_parking=False)
            self.people.append(Person(x, y, str(self.np_random.choice(colors))))

        for _ in range(self.NUM_CYCLISTS):
            x, y = self._free_spot(50, 750, self.BIKE_PATH_Y - 10,
                                   self.BIKE_PATH_Y + 10, min_dist=40,
                                   allow_paths=True, allow_parking=False)
            direction = 1 if self.np_random.random() < 0.5 else -1
            self.cyclists.append(Cyclist(x, y, direction, str(self.np_random.choice(colors))))

        self._spawn_litter()
        self.initial_litters = max(1, len(self.litters))

        for _ in range(self.NUM_CARS):
            x, y = self._free_spot(715, 745, 415, 545, min_dist=75)
            self.cars.append(Car(x, y))

        place_types = ['скамейка', 'скамейка', 'турник', 'беседка']
        for _ in range(self.NUM_PLACES):
            x, y = self._free_spot(50, 750, 50, 550, min_dist=40,
                                   allow_paths=False, allow_parking=False)
            self.places.append(PublicPlace(x, y, str(self.np_random.choice(place_types))))

        tree_types = ['Ёлка', 'Дуб']
        for _ in range(self.NUM_TREES):
            x, y = self._free_spot(50, 680, 50, 550, min_dist=50,
                                   allow_paths=False, allow_parking=False)
            self.trees.append(Tree(x, y, str(self.np_random.choice(tree_types))))

        if self.route or self.litters:
            self._start_turning_to_target()

        if self.fsm is not None:
            self.fsm.reset()

        self._update_sensors()
        if self.render_mode == 'human':
            self._render_frame()
        return self._get_observation(), {}

    def set_reward_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"Изменён параметр {key} = {value}")

    def _spawn_litter(self):
        for _ in range(self.NUM_LITTERS):
            if self.np_random.random() < 0.8:
                if self.np_random.random() < 0.5:
                    base_x = float(self.np_random.uniform(50, 750))
                    base_y = float(self.np_random.uniform(self.BIKE_PATH_Y - 20, self.BIKE_PATH_Y + 20))
                else:
                    base_x = float(self.np_random.uniform(50, 750))
                    base_y = float(self.np_random.uniform(490, 520))
            else:
                if self.places:
                    place = self.np_random.choice(self.places)
                    base_x = float(self.np_random.uniform(max(50, place.x - 40), min(750, place.x + 40)))
                    base_y = float(self.np_random.uniform(max(50, place.y - 40), min(550, place.y + 40)))
                else:
                    base_x = float(self.np_random.uniform(50, 750))
                    base_y = float(self.np_random.uniform(self.BIKE_PATH_Y - 20, self.BIKE_PATH_Y + 20))
            valid = False
            for _ in range(50):
                test_x = base_x + float(self.np_random.uniform(-15, 15))
                test_y = base_y + float(self.np_random.uniform(-15, 15))
                test_x = max(40, min(760, test_x))
                test_y = max(40, min(560, test_y))
                too_close = False
                for group in (self.people, self.cyclists, self.litters,
                              self.cars, self.places, self.trees):
                    for obj in group:
                        if math.hypot(test_x - obj.x, test_y - obj.y) < 15:
                            too_close = True
                            break
                    if too_close:
                        break
                in_building = False
                for b in self.buildings:
                    if b.contains_point(test_x, test_y) or b.distance_to_point(test_x, test_y) < 15:
                        in_building = True
                        break
                if not too_close and not in_building:
                    x, y = test_x, test_y
                    valid = True
                    break
            if not valid:
                x, y = base_x, base_y
            self.litters.append(Litter(x, y))

    def _shaping_target(self):
        r = self.robot
        visible_litters = [l for l in self.litters if self._is_litter_visible(l)]
        if visible_litters:
            nearest = min(visible_litters, key=lambda l: math.hypot(l.x - r.x, l.y - r.y))
            return (nearest.x, nearest.y)
        if self.phase == 'route' and self.wp_idx < len(self.route):
            return self.route[self.wp_idx]
        if self.litters:
            nearest = min(self.litters, key=lambda l: math.hypot(l.x - r.x, l.y - r.y))
            return (nearest.x, nearest.y)
        return None

    def _get_observation(self):
        r = self.robot
        dir_rad = math.radians(r.direction)

        obs = []
        obs.append(math.sin(dir_rad))
        obs.append(math.cos(dir_rad))
        obs.extend(d / r.lidar_range for d in r.lidar_distances)

        if self.phase == 'route' and self.wp_idx < len(self.route):
            obs.extend(self._to_robot_frame(*self.route[self.wp_idx]))
        else:
            obs.extend([0.0, 0.0])

        visible_litters = [l for l in self.litters if self._is_litter_visible(l)]
        if visible_litters:
            nearest_litter = min(visible_litters, key=lambda l: math.hypot(l.x - r.x, l.y - r.y))
            obs.extend(self._to_robot_frame(nearest_litter.x, nearest_litter.y))
        else:
            obs.extend([0.0, 0.0])

        static_objects = self.cars + self.places + self.trees
        obs.extend(self._get_nearest_visible_by_type(static_objects))

        dynamic_objects = self.people + self.cyclists
        obs.extend(self._get_nearest_visible_by_type(dynamic_objects))

        obs.append(len(self.litters) / max(1, self.initial_litters))
        obs.append(self.wp_idx / max(1, len(self.route)))
        obs.append(1.0 if self.phase == 'route' else 0.0)

        obs = np.array(obs, dtype=np.float32)
        assert obs.shape[0] == self.OBS_SIZE, \
            f"Критическая ошибка: Размер наблюдения {obs.shape[0]} != {self.OBS_SIZE}"
        return obs

    def _try_collect_litter(self):
        r = self.robot
        collected = 0
        for l in list(self.litters):
            if self._is_litter_visible(l) and math.hypot(l.x - r.x, l.y - r.y) <= self.COLLECT_RADIUS:
                self.litters.remove(l)
                collected += 1
        if collected:
            self.prev_target_dist = None
            self.stats['litter_collected'] += collected
        return collected

    def _calculate_target_angle(self):
        target = self._shaping_target()
        if target is None:
            return None
        dx = target[0] - self.robot.x
        dy = target[1] - self.robot.y
        return math.degrees(math.atan2(dy, dx)) % 360

    def _is_object_visible_in_cameras(self, obj_x, obj_y):
        dx = obj_x - self.robot.x
        dy = obj_y - self.robot.y
        dist = math.hypot(dx, dy)
        if dist > self.robot.camera_range:
            return False
        angle_to_obj = math.degrees(math.atan2(dy, dx)) % 360
        for i in range(self.robot.num_cameras):
            cam_dir = (self.robot.direction + i * 360 / self.robot.num_cameras) % 360
            angle_diff = abs(angle_to_obj - cam_dir)
            angle_diff = min(angle_diff, 360 - angle_diff)
            if angle_diff <= self.robot.camera_angle / 2:
                return True
        return False

    def _get_nearest_visible_by_type(self, object_list):
        nearest = None
        min_dist = float('inf')
        for obj in object_list:
            if self._is_object_visible_in_cameras(obj.x, obj.y):
                dist = math.hypot(obj.x - self.robot.x, obj.y - self.robot.y)
                if dist < min_dist:
                    min_dist = dist
                    nearest = obj
        if nearest is not None:
            return self._to_robot_frame(nearest.x, nearest.y)
        return [0.0, 0.0]

    def _is_litter_visible(self, litter):
        r = self.robot
        dx = litter.x - r.x
        dy = litter.y - r.y
        dist = math.hypot(dx, dy)
        if dist > r.camera_range:
            return False
        angle_to_litter = math.degrees(math.atan2(dy, dx)) % 360
        for i in range(r.num_cameras):
            cam_dir = (r.direction + i * 360 / r.num_cameras) % 360
            angle_diff = abs(angle_to_litter - cam_dir)
            angle_diff = min(angle_diff, 360 - angle_diff)
            if angle_diff <= r.camera_angle / 2:
                return True
        return False

    def _start_turning_to_target(self):
        angle = self._calculate_target_angle()
        if angle is not None:
            self.turning_to_target = True
            self.target_angle = angle
            self.turn_steps = 0

    def _free_spot(self, x_lo, x_hi, y_lo, y_hi, min_dist=20,
                   allow_paths=False, allow_parking=False, max_attempts=200):
        taken = [(self.robot.x, self.robot.y)] + list(self.route)
        for group in (self.people, self.cyclists, self.litters,
                      self.cars, self.places, self.trees):
            for obj in group:
                taken.append((obj.x, obj.y))

        x, y = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2
        for _ in range(max_attempts):
            x = float(self.np_random.uniform(x_lo, x_hi))
            y = float(self.np_random.uniform(y_lo, y_hi))
            if not all(math.hypot(x - tx, y - ty) >= min_dist for tx, ty in taken):
                continue
            inside_or_close_to_building = False
            for b in self.buildings:
                if b.contains_point(x, y) or b.distance_to_point(x, y) < min_dist:
                    inside_or_close_to_building = True
                    break
            if inside_or_close_to_building:
                continue
            on_bike_path = abs(y - self.BIKE_PATH_Y) < 10
            on_lower_path = 490 <= y <= 610
            if not allow_paths and (on_bike_path or on_lower_path):
                continue
            in_parking = (700 <= x <= 800 and 400 <= y <= 550)
            if not allow_parking and in_parking:
                continue
            margin = 30
            if x < margin or x > self.WIDTH - margin or y < margin or y > self.HEIGHT - margin:
                continue
            return x, y
        return float(x), float(y)

    def _free_spot_around(self, cx, cy, radius, min_dist):
        taken = [(self.robot.x, self.robot.y)]
        for group in (self.people, self.cyclists, self.litters,
                      self.cars, self.places, self.trees):
            for obj in group:
                taken.append((obj.x, obj.y))
        for _ in range(100):
            angle = self.np_random.uniform(0, 2 * math.pi)
            dist = self.np_random.uniform(min_dist, radius)
            x = cx + dist * math.cos(angle)
            y = cy + dist * math.sin(angle)
            if 50 <= x <= 750 and 50 <= y <= 550:
                if all(math.hypot(x - tx, y - ty) >= min_dist for tx, ty in taken):
                    return float(x), float(y)
        return self._free_spot_along_path()

    def _free_spot_along_path(self):
        for _ in range(50):
            if self.np_random.random() < 0.5:
                x = float(self.np_random.uniform(100, 700))
                y = self.BIKE_PATH_Y + float(self.np_random.uniform(-20, 20))
            else:
                x = float(self.np_random.uniform(100, 700))
                y = float(self.np_random.uniform(490, 510))
            if 50 <= x <= 750 and 50 <= y <= 550:
                return x, y
        return float(self.np_random.uniform(50, 750)), float(self.np_random.uniform(50, 550))

    def _point_in_polygon(self, x, y, polygon):
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _blocking_collision(self):
        r = self.robot
        # Забор
        if (r.x <= self.FENCE_MARGIN or r.x >= self.WIDTH - self.FENCE_MARGIN
                or r.y <= self.FENCE_MARGIN or r.y >= self.HEIGHT - self.FENCE_MARGIN):
            return 'fence'
        # Здания
        for b in self.buildings:
            if b.contains_point(r.x, r.y) or b.distance_to_point(r.x, r.y) < self.ROBOT_RADIUS:
                return 'static'
        # Статические объекты
        for o in self.cars + self.places + self.trees:
            if math.hypot(r.x - o.x, r.y - o.y) < self.COLLISION_RADIUS + 6:
                return 'static'
        return None

    def _dynamic_hit(self):
        r = self.robot
        return any(math.hypot(r.x - o.x, r.y - o.y) < self.COLLISION_RADIUS
                   for o in self.people + self.cyclists)

    def step(self, action):
        self.step_count += 1
        r = self.robot

        static_obstacles = self.cars + self.places + self.trees

        robot_obstacle = None
        if self.fsm is not None:
            is_frozen = self.fsm.current_state == self.fsm.STATE_FREEZE
            is_emergency = self.fsm.movement.current_mode == self.fsm.movement.MODE_EMERGENCY
            is_recovery = self.fsm.current_state == self.fsm.STATE_RECOVERY
            if is_frozen or is_emergency or is_recovery:
                robot_obstacle = self.robot
            if is_emergency:
                self.emergency_ticks += 1

        for p in self.people:
            p.move(obstacles=static_obstacles, buildings=self.buildings,
                   places=self.places, robot=robot_obstacle)
        for c in self.cyclists:
            c.move(self.BIKE_PATH_Y, buildings=self.buildings, robot=self.robot)

        # Обновляем сенсоры ДО принятия решения
        self._update_sensors()

        if self.fsm is not None and self.fsm_enabled:
            linear_vel_m_s, angular_vel_rad_s = self.fsm.get_action(
                self._get_observation(), action
            )
        else:
            linear_vel_m_s = (float(action[0]) + 1.0) / 2.0 * self.MAX_LINEAR_SPEED
            angular_vel_rad_s = float(action[1]) * self.MAX_ANGULAR_SPEED

        # Запоминаем позицию ДО движения
        prev_x, prev_y = r.x, r.y

        # Применяем скорость
        linear_vel_pix_per_tick = (linear_vel_m_s * self.PIXELS_PER_METER) / 30.0
        angular_vel_deg_per_tick = math.degrees(angular_vel_rad_s) / 30.0
        r.apply_velocity(linear_vel_pix_per_tick, angular_vel_deg_per_tick)

        # Проверяем блокирующую коллизию
        block = self._blocking_collision()
        if block:
            # Откатываем позицию
            r.x, r.y = prev_x, prev_y
            if self.fsm is not None:
                self.fsm.movement.hard_stop()

        # Обновляем сенсоры после движения
        self._update_sensors()

        # Награды
        terminated = False
        reward = self.R_STEP

        # Детектор застревания
        self._stuck_steps += 1
        if self._stuck_steps >= 60:
            moved = math.hypot(r.x - self._stuck_x, r.y - self._stuck_y)
            if moved < 10.0:
                reward -= 10.0
            self._stuck_x, self._stuck_y = r.x, r.y
            self._stuck_steps = 0

        # Сбор мусора
        collected = self._try_collect_litter()
        if collected:
            reward += self.R_LITTER * collected
            self.prev_target_dist = None
            self.prev_litter_dist = None

        # прогресс к цели
        target = self._shaping_target()
        if target is not None:
            dist = math.hypot(r.x - target[0], r.y - target[1])
            if self.prev_target_dist is not None:
                reward += self.R_PROGRESS * (self.prev_target_dist - dist)
            self.prev_target_dist = dist

        # прогресс к ближайшему мусору
        if self.litters and self.phase == 'route':
            nearest_litter = min(self.litters,
                                 key=lambda l: math.hypot(l.x - r.x, l.y - r.y))
            dist_to_litter = math.hypot(r.x - nearest_litter.x, r.y - nearest_litter.y)
            if dist_to_litter < 80:
                if self.prev_litter_dist is not None:
                    reward += 0.4 * (self.prev_litter_dist - dist_to_litter)
                self.prev_litter_dist = dist_to_litter
            else:
                self.prev_litter_dist = None

        # Достижение waypoint
        if self.phase == 'route' and self.wp_idx < len(self.route):
            wp_x, wp_y = self.route[self.wp_idx]
            if math.hypot(r.x - wp_x, r.y - wp_y) <= self.waypoint_radius:
                reward += self.R_WAYPOINT
                self.wp_idx += 1
                self.prev_target_dist = None
                if self.wp_idx >= len(self.route):
                    reward += self.R_ROUTE_DONE
                    self.phase = 'cleanup'

        # Завершение эпизода
        if self.phase == 'cleanup' and not self.litters:
            reward += self.R_ALL_LITTER
            terminated = True

        # Столкновения
        if self.collision_cd > 0:
            self.collision_cd -= 1

        hit = block if block else ('dynamic' if self._dynamic_hit() else None)
        if hit and self.collision_cd == 0:
            if hit == 'dynamic':
                reward += self.R_COLLISION_DYNAMIC
                self.stats['collisions_dynamic'] += 1
                self.collision_cd = 40
            else:
                reward += self.R_COLLISION_STATIC
                self.stats['collisions_static'] += 1
                self.collision_cd = 80
                if self.fsm is not None:
                    self.fsm.trigger_recovery()
            if self.fsm is not None:
                self.fsm.trigger_recovery()

        # Штраф за близость к границам
        margin = 45.0
        dist_to_edge = min(r.x, self.WIDTH - r.x, r.y, self.HEIGHT - r.y)
        if dist_to_edge < margin:
            reward -= (margin - dist_to_edge) * 0.15

        if self.boundary_cd > 0:
            self.boundary_cd -= 1
        if dist_to_edge <= 12 and self.boundary_cd == 0:
            reward += self.R_BOUNDARY
            self.boundary_cd = 50

        truncated = self.step_count >= self.max_steps

        info = {
            'phase': self.phase,
            'wp_idx': self.wp_idx,
            'litters_left': len(self.litters),
            'success': terminated and self.phase == 'cleanup' and not self.litters,
        }
        if self.fsm is not None:
            info['fsm_state'] = self.fsm.current_state
            info['movement_mode'] = self.fsm.movement.current_mode
            info['linear_vel'] = self.fsm.movement.current_linear_vel
            info['angular_vel'] = self.fsm.movement.current_angular_vel
            info['recoveries'] = getattr(self.fsm, 'recovery_count', 0)
            info['emergency_ticks'] = self.emergency_ticks
            planner = getattr(self.fsm, 'planner', None)
            path = getattr(self.fsm, 'path', None)
            info['path_len'] = getattr(planner, 'last_path_len', 0.0) if planner else 0.0
            info['path_points'] = len(path) if path else 0
            if self.debug_log and info['recoveries'] != self._last_recovery_count:
                self._last_recovery_count = info['recoveries']
                print(f"[{self.step_count}] RECOVERY #{info['recoveries']} "
                      f"state={self.fsm.current_state} wp={self.wp_idx} "
                      f"pos=({r.x:.0f},{r.y:.0f})")

        if self.render_mode == 'human':
            self._render_frame()

        return self._get_observation(), reward, terminated, truncated, info

    def _update_sensors(self):
        obstacles = build_lidar_obstacles(
            self.people, self.cyclists, self.litters,
            self.cars, self.places, self.trees)
        self.robot.update_lidar(obstacles, buildings=self.buildings)

    def _current_target(self):
        if self.phase == 'route' and self.wp_idx < len(self.route):
            return self.route[self.wp_idx]
        if self.litters:
            nearest = min(self.litters,
                          key=lambda l: math.hypot(l.x - self.robot.x, l.y - self.robot.y))
            return (nearest.x, nearest.y)
        return None

    def _check_collision(self):
        for obj in (self.people + self.cyclists + self.cars +
                    self.places + self.trees):
            if math.hypot(self.robot.x - obj.x, self.robot.y - obj.y) < self.COLLISION_RADIUS:
                return True
        return False

    def _ray_intersects_building(self, x1, y1, x2, y2):
        for building in self.buildings:
            n = len(building.polygon)
            for i in range(n):
                bx1, by1 = building.polygon[i]
                bx2, by2 = building.polygon[(i + 1) % n]
                if self._segments_intersect(x1, y1, x2, y2, bx1, by1, bx2, by2):
                    return True
        return False

    def _segments_intersect(self, x1, y1, x2, y2, x3, y3, x4, y4):
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return False
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        return 0 <= t <= 1 and 0 <= u <= 1

    def _is_object_visible(self, obj_x, obj_y, obj_radius=6):
        r = self.robot
        dx = obj_x - r.x
        dy = obj_y - r.y
        dist = math.hypot(dx, dy)
        if dist > r.camera_range:
            return False
        angle_to_obj = math.degrees(math.atan2(dy, dx)) % 360
        in_camera_fov = False
        for i in range(r.num_cameras):
            cam_dir = (r.direction + i * 360 / r.num_cameras) % 360
            angle_diff = abs(angle_to_obj - cam_dir)
            angle_diff = min(angle_diff, 360 - angle_diff)
            if angle_diff <= r.camera_angle / 2:
                in_camera_fov = True
                break
        if not in_camera_fov:
            return False
        if self._ray_intersects_building(r.x, r.y, obj_x, obj_y):
            return False
        all_obstacles = self.cars + self.places + self.trees
        for obstacle in all_obstacles:
            obs_dist = math.hypot(obstacle.x - r.x, obstacle.y - r.y)
            if obs_dist < dist:
                if self._point_on_ray(r.x, r.y, obj_x, obj_y,
                                      obstacle.x, obstacle.y, obstacle_radius=15):
                    return False
        return True

    def _point_on_ray(self, x1, y1, x2, y2, px, py, obstacle_radius=15):
        dx = x2 - x1
        dy = y2 - y1
        line_len_sq = dx * dx + dy * dy
        if line_len_sq < 1e-6:
            return False
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / line_len_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        dist_to_line = math.hypot(px - proj_x, py - proj_y)
        return dist_to_line < obstacle_radius

    def _to_robot_frame(self, tx, ty):
        r = self.robot
        dx, dy = tx - r.x, ty - r.y
        dir_rad = math.radians(r.direction)
        local_x = dx * math.cos(dir_rad) + dy * math.sin(dir_rad)
        local_y = -dx * math.sin(dir_rad) + dy * math.cos(dir_rad)
        norm = math.hypot(self.WIDTH, self.HEIGHT)
        return [float(np.clip(local_x / norm, -1.0, 1.0)),
                float(np.clip(local_y / norm, -1.0, 1.0))]

    def render(self):
        if self.render_mode == 'human':
            self._render_frame()

    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('arial', 16)

        self.screen.fill((48, 194, 107))
        pygame.draw.line(self.screen, (150, 150, 150),
                         (0, self.BIKE_PATH_Y), (self.WIDTH, self.BIKE_PATH_Y), 12)
        pygame.draw.rect(self.screen, (202, 204, 206), (0, 500, 500, 16))
        pygame.draw.rect(self.screen, (202, 204, 206), (500, 500, 16, 100))
        pygame.draw.rect(self.screen, (126, 137, 148), (700, 400, 100, 150))
        pygame.draw.rect(self.screen, (155, 94, 64), (0, 0, self.WIDTH, self.HEIGHT), 6)

        for i, (wx, wy) in enumerate(self.route):
            if self.phase == 'route' and i == self.wp_idx:
                pygame.draw.circle(self.screen, (255, 0, 0), (wx, wy), 10, 2)
            elif self.phase == 'route' and i > self.wp_idx:
                pygame.draw.circle(self.screen, (0, 0, 255), (wx, wy), 7, 1)

        for building in self.buildings:
            building.draw(self.screen)
        for obj in self.litters:
            obj.draw(self.screen, self.robot, False)
        for obj in self.cars:
            obj.draw(self.screen, self.robot, False)
        for obj in self.places:
            obj.draw(self.screen, self.robot, False)
        for p in self.people:
            p.draw(self.screen, self.robot, False)
        for c in self.cyclists:
            c.draw(self.screen, self.robot, False)

        self.robot.draw(self.screen)

        sorted_trees = sorted(self.trees, key=lambda t: t.y)
        for obj in sorted_trees:
            obj.draw(self.screen, self.robot, False)

        # маршрут A*
        path = getattr(self.fsm, 'path', None) if self.fsm else None
        if path and len(path) >= 2:
            pts = [(int(x), int(y)) for x, y in path]
            surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            pygame.draw.lines(surface, (0, 255, 255, 100), False, pts, 2)
            for p in pts:
                pygame.draw.circle(surface, (0, 200, 255, 100), p, 3)
            self.screen.blit(surface, (0, 0))

        # HUD
        mode_names = {1: 'CRUISE', 2: 'SLOWDOWN', 3: 'EMERGENCY'}
        mode_str = mode_names.get(self.fsm.movement.current_mode, '?') if self.fsm else '?'
        state_str = self.fsm.current_state if self.fsm else '-'
        rec = getattr(self.fsm, 'recovery_count', 0) if self.fsm else 0
        hud1 = (f"Фаза: {self.phase} | Точка: {self.wp_idx}/{len(self.route)} | "
                f"Мусор: {len(self.litters)} | Шаг: {self.step_count}/{self.max_steps}")
        hud2 = (f"FSM: {state_str} | Режим: {mode_str} | "
                f"Recovery: {rec}")
        self.screen.blit(self.font.render(hud1, True, (255, 255, 255)), (12, 12))
        self.screen.blit(self.font.render(hud2, True, (255, 255, 0)), (12, 32))
        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])

    def close(self):
        if hasattr(self, 'logger'):
            self.logger.close()
        if self.screen is not None:
            pygame.quit()
            self.screen = None


if __name__ == '__main__':
    env = RobotEnvWaypoints(render_mode='human', stage=2)
    obs, _ = env.reset(seed=42)
    print(f"Размер наблюдения: {obs.shape}")
    print("ESC/Q — выход | R — новый эпизод | P — пауза | L — лог вкл/выкл")

    running = True
    paused = False
    episode = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_l:
                    env.debug_log = not env.debug_log

        if paused:
            pygame.time.wait(30)
            continue

        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

        if terminated or truncated:
            episode += 1
            rec = getattr(env.fsm, 'recovery_count', 0) if env.fsm else 0
            print(f"Эпизод {episode}: wp={env.wp_idx}/{len(env.route)}, "
                  f"мусор={len(env.litters)}, recovery={rec}, "
                  f"EMG={env.emergency_ticks}, шагов={env.step_count}, "
                  f"коллизии стат/дин={env.stats['collisions_static']}/"
                  f"{env.stats['collisions_dynamic']}, "
                  f"путь={env.stats['distance_traveled']:.0f}px, "
                  f"reward={env.stats['total_reward']:.1f} ---")
            obs, _ = env.reset()

    env.close()


def make_env(stage=1, env_kwargs=None, reward_kwargs=None, fsm_enabled=True):
    return RobotEnvWaypoints(render_mode=None, stage=stage,
                             env_kwargs=env_kwargs, reward_kwargs=reward_kwargs,
                             fsm_enabled=fsm_enabled)
