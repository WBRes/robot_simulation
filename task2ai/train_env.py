import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

from simulation import (
    Robot, Person, Cyclist, Litter, Car, PublicPlace, Tree,
    build_lidar_obstacles,
)


class RobotEnvWaypoints(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    WIDTH, HEIGHT = 800, 600
    BIKE_PATH_Y = 150

    WAYPOINTS = [(100, 100), (700, 100), (700, 500),
                 (100, 500), (400, 300), (100, 500)]

    STAGES = {
        1: dict(people=0, cyclists=0, litters=2, cars=2, places=1, trees=2,
                wp=3, wp_radius=40, max_steps=1500),
        2: dict(people=4, cyclists=2, litters=6, cars=4, places=4, trees=5,
                wp=6, wp_radius=30, max_steps=2200),
    }
    R_COLLISION_STATIC = -20.0
    R_COLLISION_DYNAMIC = -6.0

    MAX_LINEAR_SPEED = 3.0
    MAX_ANGULAR_SPEED = 6.0

    COLLECT_RADIUS = 15
    COLLISION_RADIUS = 16

    DEFAULT_MAX_STEPS = 1500

    R_STEP = -0.05
    R_PROGRESS = 0.1
    R_WAYPOINT = 30.0
    R_ROUTE_DONE = 50.0
    R_LITTER = 100.0
    R_ALL_LITTER = 100.0
    R_COLLISION = -30.0
    R_BOUNDARY = -10.0
    WAYPOINT_RADIUS = 30
    LITTER_MAGNET_RADIUS = 120
    R_COLLISION_STATIC = -20.0
    R_COLLISION_DYNAMIC = -6.0

    OBS_SIZE = 33

    def __init__(self, render_mode=None, max_steps=None, num_waypoints=None, stage=1):
        super().__init__()
        self.render_mode = render_mode
        cfg = self.STAGES[stage]
        self.max_steps = max_steps if max_steps is not None else cfg.get('max_steps', self.DEFAULT_MAX_STEPS)
        n_wp = num_waypoints if num_waypoints is not None else cfg['wp']
        self.route = list(self.WAYPOINTS[:n_wp])
        self.waypoint_radius = cfg['wp_radius']

        self.stage = stage
        self.NUM_PEOPLE = cfg['people']
        self.NUM_CYCLISTS = cfg['cyclists']
        self.NUM_LITTERS = cfg['litters']
        self.NUM_CARS = cfg['cars']
        self.NUM_PLACES = cfg['places']
        self.NUM_TREES = cfg['trees']

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
        self.initial_litters = 1

        self.screen = None
        self.clock = None
        self.font = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.wp_idx = 0
        self.phase = 'route'
        self.prev_target_dist = None
        self.prev_litter_dist = None
        self.collision_cd = 0
        self.prev_litter_dist = None
        self.boundary_cd = 0
        self.robot = Robot(self.WIDTH // 2, self.HEIGHT // 2)
        self._stuck_x, self._stuck_y = self.robot.x, self.robot.y
        self._stuck_steps = 0
        self.people, self.cyclists, self.litters = [], [], []
        self.cars, self.places, self.trees = [], [], []
        colors = ['red', 'blue', 'green', 'yellow', 'purple']
        cfg = self.STAGES[self.stage]
        if cfg.get('random_route', False):
            n_wp = int(self.np_random.integers(2, 6))
            self.route = []
            while len(self.route) < n_wp:
                x = float(self.np_random.uniform(60, self.WIDTH - 60))
                y = float(self.np_random.uniform(60, self.HEIGHT - 60))
                if (all(math.hypot(x - wx, y - wy) > 120 for wx, wy in self.route)
                        and math.hypot(x - self.robot.x, y - self.robot.y) > 120):
                    self.route.append((x, y))
        else:
            self.route = list(self.WAYPOINTS[:len(self.route)])
        self.robot.direction = float(self.np_random.uniform(0, 360))
        for _ in range(self.NUM_PEOPLE):
            x, y = self._free_spot(100, 700, 100, 500, min_dist=30)
            self.people.append(Person(x, y, str(self.np_random.choice(colors))))
        for _ in range(self.NUM_CYCLISTS):
            x = float(self.np_random.uniform(50, 750))
            direction = 1 if self.np_random.random() < 0.5 else -1
            self.cyclists.append(Cyclist(x, self.BIKE_PATH_Y, direction,
                                         str(self.np_random.choice(colors))))
        for _ in range(self.NUM_LITTERS):
            x, y = self._free_spot(50, 750, 50, 550, min_dist=30)
            self.litters.append(Litter(x, y))
        self.initial_litters = max(1, len(self.litters))
        for _ in range(self.NUM_CARS):
            x, y = self._free_spot(710, 750, 410, 550, min_dist=20)
            self.cars.append(Car(x, y))
        place_types = ['скамейка', 'скамейка', 'турник', 'беседка']
        for _ in range(self.NUM_PLACES):
            x, y = self._free_spot(50, 750, 50, 550, min_dist=30)
            self.places.append(PublicPlace(x, y, str(self.np_random.choice(place_types))))
        tree_types = ['Ёлка', 'Дуб']
        for _ in range(self.NUM_TREES):
            x, y = self._free_spot(50, 680, 50, 550, min_dist=50)
            self.trees.append(Tree(x, y, str(self.np_random.choice(tree_types))))
        self._update_sensors()
        if self.render_mode == 'human':
            self._render_frame()
        return self._get_observation(), {}

    def _shaping_target(self):
        if self.litters:
            nearest = min(self.litters,
                          key=lambda l: math.hypot(l.x - self.robot.x,
                                                   l.y - self.robot.y))
            d = math.hypot(nearest.x - self.robot.x, nearest.y - self.robot.y)
            if d < self.LITTER_MAGNET_RADIUS:
                return (nearest.x, nearest.y)
        if self.phase == 'route' and self.wp_idx < len(self.route):
            return self.route[self.wp_idx]
        if self.litters:
            return (nearest.x, nearest.y)
        return None

    def _free_spot(self, x_lo, x_hi, y_lo, y_hi, min_dist):
        taken = [(self.robot.x, self.robot.y)] + list(self.route)
        for group in (self.people, self.cyclists, self.litters,
                      self.cars, self.places, self.trees):
            for obj in group:
                taken.append((obj.x, obj.y))
        x = float(self.np_random.uniform(x_lo, x_hi))
        y = float(self.np_random.uniform(y_lo, y_hi))
        for _ in range(100):
            if all(math.hypot(x - tx, y - ty) >= min_dist for tx, ty in taken):
                return x, y
            x = float(self.np_random.uniform(x_lo, x_hi))
            y = float(self.np_random.uniform(y_lo, y_hi))
        return x, y
    def step(self, action):
        self.step_count += 1
        r = self.robot

        linear_vel = (float(action[0]) + 1.0) / 2.0 * self.MAX_LINEAR_SPEED
        angular_vel = float(action[1]) * self.MAX_ANGULAR_SPEED

        r.apply_velocity(linear_vel, angular_vel)

        for p in self.people:
            p.move()
        for c in self.cyclists:
            c.move(self.BIKE_PATH_Y)

        self._update_sensors()
        terminated = False
        reward = self.R_STEP

        self._stuck_steps += 1
        if self._stuck_steps >= 60:
            moved = math.hypot(r.x - self._stuck_x, r.y - self._stuck_y)
            if moved < 10.0:
                reward -= 10.0
            self._stuck_x, self._stuck_y = r.x, r.y
            self._stuck_steps = 0

        collected = self._try_collect_litter()
        if collected:
            reward += self.R_LITTER * collected
            self.prev_target_dist = None
            self.prev_litter_dist = None

        target = self._shaping_target()
        if target is not None:
            dist = math.hypot(r.x - target[0], r.y - target[1])
            if self.prev_target_dist is not None:
                reward += self.R_PROGRESS * (self.prev_target_dist - dist)
            self.prev_target_dist = dist

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

        if self.phase == 'route' and self.wp_idx < len(self.route):
            wp_x, wp_y = self.route[self.wp_idx]
            if math.hypot(r.x - wp_x, r.y - wp_y) <= self.waypoint_radius:
                reward += self.R_WAYPOINT
                self.wp_idx += 1
                self.prev_target_dist = None
                if self.wp_idx >= len(self.route):
                    reward += self.R_ROUTE_DONE
                    self.phase = 'cleanup'

        if self.phase == 'cleanup' and not self.litters:
            reward += self.R_ALL_LITTER
            terminated = True

        if self.collision_cd > 0:
            self.collision_cd -= 1
        if self.collision_cd == 0:
            hit_static = any(math.hypot(r.x - o.x, r.y - o.y) < self.COLLISION_RADIUS
                             for o in (self.cars + self.places + self.trees))
            hit_dyn = any(math.hypot(r.x - o.x, r.y - o.y) < self.COLLISION_RADIUS
                          for o in (self.people + self.cyclists))
            if hit_static:
                reward += self.R_COLLISION_STATIC
                self.collision_cd = 50
            elif hit_dyn:
                reward += self.R_COLLISION_DYNAMIC
                self.collision_cd = 25

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
        if self.render_mode == 'human':
            self._render_frame()
        return self._get_observation(), reward, terminated, truncated, info


    def _update_sensors(self):
        obstacles = build_lidar_obstacles(
            self.people, self.cyclists, self.litters,
            self.cars, self.places, self.trees)
        self.robot.update_lidar(obstacles)


    def _try_collect_litter(self):
        collected = 0
        for litter in self.litters[:]:
            if math.hypot(self.robot.x - litter.x,
                          self.robot.y - litter.y) <= self.COLLECT_RADIUS:
                self.litters.remove(litter)
                collected += 1
        if collected:
            self.prev_target_dist = None
        return collected

    def _current_target(self):
        if self.phase == 'route' and self.wp_idx < len(self.route):
            return self.route[self.wp_idx]
        if self.litters:
            nearest = min(self.litters,
                          key=lambda l: math.hypot(l.x - self.robot.x,
                                                   l.y - self.robot.y))
            return (nearest.x, nearest.y)
        return None

    def _check_collision(self):
        for obj in (self.people + self.cyclists + self.cars +
                    self.places + self.trees):
            if math.hypot(self.robot.x - obj.x,
                          self.robot.y - obj.y) < self.COLLISION_RADIUS:
                return True
        return False


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

        if self.litters:
            nearest = min(self.litters,
                          key=lambda l: math.hypot(l.x - r.x, l.y - r.y))
            obs.extend(self._to_robot_frame(nearest.x, nearest.y))
        else:
            obs.extend([0.0, 0.0])

        obs.append(len(self.litters) / self.initial_litters)
        obs.append(self.wp_idx / len(self.route))
        obs.append(1.0 if self.phase == 'cleanup' else 0.0)

        obs = np.array(obs, dtype=np.float32)
        assert obs.shape[0] == self.OBS_SIZE, \
            f"Размер наблюдения {obs.shape[0]} != {self.OBS_SIZE}"
        return obs

    def _to_robot_frame(self, tx, ty):
        r = self.robot
        dx, dy = tx - r.x, ty - r.y
        dir_rad = math.radians(r.direction)
        local_x = dx * math.cos(dir_rad) + dy * math.sin(dir_rad)   # вперёд
        local_y = -dx * math.sin(dir_rad) + dy * math.cos(dir_rad)  # вправо
        norm = math.hypot(self.WIDTH, self.HEIGHT)  # диагональ мира, ~1000
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
                         (0, self.BIKE_PATH_Y), (self.WIDTH, self.BIKE_PATH_Y), 4)
        pygame.draw.rect(self.screen, (202, 204, 206), (0, 500, 500, 8))
        pygame.draw.rect(self.screen, (202, 204, 206), (500, 500, 8, 100))
        pygame.draw.rect(self.screen, (126, 137, 148), (700, 400, 100, 150))
        pygame.draw.rect(self.screen, (155, 94, 64),
                         (0, 0, self.WIDTH, self.HEIGHT), 6)

        for i, (wx, wy) in enumerate(self.route):
            if self.phase == 'route' and i == self.wp_idx:
                pygame.draw.circle(self.screen, (255, 0, 0), (wx, wy), 10, 2)
            elif self.phase == 'route' and i > self.wp_idx:
                pygame.draw.circle(self.screen, (0, 0, 255), (wx, wy), 7, 1)

        for obj in self.litters:
            obj.draw(self.screen, True)
        for obj in self.cars:
            obj.draw(self.screen, True)
        for obj in self.places:
            obj.draw(self.screen, True)
        for obj in self.trees:
            obj.draw(self.screen, True)
        for p in self.people:
            p.draw(self.screen, self.robot, True)
        for c in self.cyclists:
            c.draw(self.screen, self.robot, True)

        self.robot.draw(self.screen)

        hud = (f"Фаза: {self.phase} | Точка: {self.wp_idx}/{len(self.route)} | "
               f"Мусор: {len(self.litters)} | Шаг: {self.step_count}/{self.max_steps}")
        self.screen.blit(self.font.render(hud, True, (255, 255, 255)), (12, 12))

        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

if __name__ == '__main__':
    env = RobotEnvWaypoints(render_mode='human')
    obs, _ = env.reset()
    print(f"Размер наблюдения: {obs.shape}")
    print("ESC или Q - выход")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Эпизод закончен: {info}")
            obs, _ = env.reset()

    env.close()

def make_env(stage=1):
    return RobotEnvWaypoints(render_mode=None, max_steps=1500, stage=stage)