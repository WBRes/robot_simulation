import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

from simulation import (
    Robot, Person, Cyclist, Litter, Car, PublicPlace, Tree, Building,
    build_lidar_obstacles,
)

# Класс логики конечного автомата
class HybridFSMController:
    STATE_RL = "RL_CONTROL"
    STATE_TURN = "TURN_TO_TARGET"
    STATE_TURN_TO_LITTER = "TURN_TO_LITTER"
    STATE_RECOVERY = "RECOVERY_STUCK"
    STATE_FREEZE = "FREEZE_DYNAMIC"

    RECOVERY_REVERSE_AWAY = "reverse_from_obstacle"
    RECOVERY_ALIGN = "align_perpendicular"
    RECOVERY_REVERSE_ALONG = "reverse_along_wall"
    RECOVERY_FINAL_TURN = "final_turn"

    FREEZE_DISTANCE = 50.0
    FREEZE_COOLDOWN = 50
    MAX_FREEZE_TIME = 200
    FREEZE_POST_COOLDOWN = 100
    FREEZE_ANGLE_CONE = 120

    LITTER_TURN_DURATION = 30
    NO_PROGRESS_LIMIT = 1500

    ROBOT_LENGTH = 34.0
    WALL_CLEARANCE = 50.0
    REVERSE_SPEED = -0.6
    ALIGN_TURN_SPEED = 1.0

    def __init__(self, env):
        self.env = env
        self.current_state = self.STATE_RL
        self.state_timer = 0
        self.target_angle = 0.0
        self.freeze_cooldown_timer = 0
        self.litter_turn_timer = 0
        self.no_progress_steps = 0
        self.last_wp_idx = 0
        self.last_litter_count = 0
        self.last_target_dist = float('inf')
        self.no_progress_warned = False

        self.recovery_substate = None
        self.recovery_obstacle_pos = None
        self.recovery_phase_timer = 0
        self.recovery_frozen_during = None
        self.recovery_target_side = None
        self.recovery_equidistant_mode = False
        self.recovery_target_equidistant = None

        self.frozen_dynamic_target = None
        self.frozen_dynamic_timer = 0

        self.freeze_post_cooldown_timer = 0

    def get_action(self, obs, rl_action):
        self.state_timer += 1
        r = self.env.robot

        target = self.env._shaping_target()
        current_dist = math.hypot(r.x - target[0], r.y - target[1]) if target else float('inf')
        progress_made = (
                self.env.wp_idx != self.last_wp_idx or
                len(self.env.litters) < self.last_litter_count or
                current_dist < self.last_target_dist - 5.0
        )
        if progress_made:
            self.no_progress_steps = 0
            self.last_wp_idx = self.env.wp_idx
            self.last_litter_count = len(self.env.litters)
            self.last_target_dist = current_dist
            self.no_progress_warned = False
        else:
            self.no_progress_steps += 1

        if (self.no_progress_steps >= self.NO_PROGRESS_LIMIT and
                self.current_state == self.STATE_RL):
            if not self.no_progress_warned:
                print(f"FSM: {self.no_progress_steps} тиков без прогресса, принудительный поворот к цели")
                self.no_progress_warned = True
            self._transition_to_nearest_target()

        dangerous_obj_nearby = False

        if self.freeze_post_cooldown_timer > 0:
            self.freeze_post_cooldown_timer -= 1
        else:
            # Обычная проверка: объект в радиусе и в переднем конусе
            for obj in self.env.people + self.env.cyclists:
                dist = math.hypot(r.x - obj.x, r.y - obj.y)
                if dist <= self.FREEZE_DISTANCE:
                    # Вычисляем угол до объекта относительно направления робота
                    to_obj_x = obj.x - r.x
                    to_obj_y = obj.y - r.y
                    angle_to_obj = math.degrees(math.atan2(to_obj_y, to_obj_x)) % 360
                    angle_diff = (angle_to_obj - r.direction + 180) % 360 - 180

                    if abs(angle_diff) < self.FREEZE_ANGLE_CONE:
                        dangerous_obj_nearby = True
                        self.frozen_dynamic_target = (obj.x, obj.y)
                        self.frozen_dynamic_timer = 0
                        break

        if not dangerous_obj_nearby and self.frozen_dynamic_target is not None:
            self.frozen_dynamic_timer += 1
            for obj in self.env.people + self.env.cyclists:
                dist = math.hypot(r.x - obj.x, r.y - obj.y)
                if dist <= self.FREEZE_DISTANCE:
                    to_obj_x = obj.x - r.x
                    to_obj_y = obj.y - r.y
                    angle_to_obj = math.degrees(math.atan2(to_obj_y, to_obj_x)) % 360
                    angle_diff = (angle_to_obj - r.direction + 180) % 360 - 180

                    if abs(angle_diff) < self.FREEZE_ANGLE_CONE:
                        dangerous_obj_nearby = True
                        self.frozen_dynamic_target = (obj.x, obj.y)
                        self.frozen_dynamic_timer = 0
                        break

        visible_litters = [l for l in self.env.litters if self.env._is_litter_visible(l)]
        has_visible_litter = len(visible_litters) > 0

        if self.current_state == self.STATE_RL:
            if dangerous_obj_nearby:
                self._transition(self.STATE_FREEZE)
                self.freeze_cooldown_timer = 0
            elif has_visible_litter:
                nearest = min(visible_litters, key=lambda l: math.hypot(l.x - r.x, l.y - r.y))
                dx = nearest.x - r.x
                dy = nearest.y - r.y
                self.target_angle = math.degrees(math.atan2(dy, dx)) % 360
                self._transition(self.STATE_TURN_TO_LITTER)
                self.litter_turn_timer = 0
            elif self.env.wp_idx < len(self.env.route):
                self._transition(self.STATE_TURN)
                wp_x, wp_y = self.env.route[self.env.wp_idx]
                self.target_angle = self._get_safe_target_angle(wp_x, wp_y)

        elif self.current_state == self.STATE_TURN_TO_LITTER:
            self.litter_turn_timer += 1
            angle_diff = (self.target_angle - r.direction + 180) % 360 - 180
            if dangerous_obj_nearby:
                self.recovery_frozen_during = self.STATE_TURN_TO_LITTER
                self._transition(self.STATE_FREEZE)
                self.freeze_cooldown_timer = 0
            elif (
                    abs(angle_diff) < 5.0 or self.litter_turn_timer >= self.LITTER_TURN_DURATION or not has_visible_litter):
                self._transition_to_nearest_target()

        elif self.current_state == self.STATE_TURN:
            angle_diff = (self.target_angle - r.direction + 180) % 360 - 180
            if dangerous_obj_nearby:
                self.recovery_frozen_during = self.STATE_TURN
                self._transition(self.STATE_FREEZE)
                self.freeze_cooldown_timer = 0
            elif abs(angle_diff) < 3.0 or self.state_timer >= self.env.max_turn_steps:
                self.env.turning_to_target = False
                self._transition(self.STATE_RL)

        elif self.current_state == self.STATE_RECOVERY:
            if dangerous_obj_nearby:
                self.recovery_frozen_during = self.recovery_substate
                self._transition(self.STATE_FREEZE)
                self.freeze_cooldown_timer = 0
            else:
                self._execute_recovery_phase()

        elif self.current_state == self.STATE_FREEZE:

            # 1. Принудительный выход если слишком долго стоим
            if self.freeze_cooldown_timer >= self.MAX_FREEZE_TIME:
                print(f"FSM: Принудительный выход из FREEZE после {self.MAX_FREEZE_TIME} тиков")
                self.frozen_dynamic_target = None
                self.frozen_dynamic_timer = 0
                self.no_progress_steps = 0
                self.no_progress_warned = False
                # Включаем иммунитет
                self.freeze_post_cooldown_timer = self.FREEZE_POST_COOLDOWN
                self._transition_to_nearest_target()

            # 2. Объект ушёл, ждём FREEZE_COOLDOWN тиков
            elif not dangerous_obj_nearby:
                self.freeze_cooldown_timer += 1
                if self.freeze_cooldown_timer >= self.FREEZE_COOLDOWN:
                    self.frozen_dynamic_target = None
                    self.frozen_dynamic_timer = 0
                    self.no_progress_steps = 0
                    self.no_progress_warned = False
                    # Включаем иммунитет
                    self.freeze_post_cooldown_timer = self.FREEZE_POST_COOLDOWN
                    self._transition_to_nearest_target()

            # 3. Объект всё ещё рядом — сбрасываем таймер (перезапуск FREEZE)
            else:
                self.freeze_cooldown_timer = 0

        if self.current_state == self.STATE_RL:
            return rl_action
        elif self.current_state == self.STATE_TURN_TO_LITTER:
            angle_diff = (self.target_angle - r.direction + 180) % 360 - 180
            turn_speed = self.env.MAX_ANGULAR_SPEED if angle_diff > 0 else -self.env.MAX_ANGULAR_SPEED
            norm_turn = np.clip(turn_speed / self.env.MAX_ANGULAR_SPEED, -1.0, 1.0)
            return np.array([0.4, norm_turn], dtype=np.float32)
        elif self.current_state == self.STATE_TURN:
            angle_diff = (self.target_angle - r.direction + 180) % 360 - 180
            turn_speed = self.env.MAX_ANGULAR_SPEED if angle_diff > 0 else -self.env.MAX_ANGULAR_SPEED
            norm_turn = np.clip(turn_speed / self.env.MAX_ANGULAR_SPEED, -1.0, 1.0)
            return np.array([0.4, norm_turn], dtype=np.float32)
        elif self.current_state == self.STATE_RECOVERY:
            return self._execute_recovery_action()
        elif self.current_state == self.STATE_FREEZE:
            # В FREEZE робот стоит на месте
            return np.array([0.0, 0.0], dtype=np.float32)

        return rl_action

    def _start_recovery(self, obstacle_pos=None):
        """Запуск recovery при столкновении"""
        self.current_state = self.STATE_RECOVERY
        self.state_timer = 0
        self.recovery_phase_timer = 0
        self.recovery_frozen_during = None
        self.recovery_equidistant_mode = False
        self.recovery_target_equidistant = None

        if obstacle_pos is None:
            obstacle_pos = self._find_nearest_obstacle_pos()
        self.recovery_obstacle_pos = obstacle_pos

        self.recovery_substate = self.RECOVERY_REVERSE_AWAY
        print(f"FSM: Начало recovery от препятствия {obstacle_pos}")

    def _get_safe_target_angle(self, target_x, target_y):
        r = self.env.robot

        dx = target_x - r.x
        dy = target_y - r.y
        direct_angle = math.degrees(math.atan2(dy, dx)) % 360
        dist_to_target = math.hypot(dx, dy)

        if dist_to_target < 50:
            return direct_angle

        # Ищем препятствие на пути к цели
        obstacle_dist = float('inf')
        obstacle_found = False
        obstacle_side = 0

        for i, (dist, obj_type) in enumerate(zip(r.lidar_distances, r.lidar_types)):
            if obj_type in ['car', 'tree', 'place', 'building', 'fence']:
                ray_local_angle = (i * 360 / r.num_rays) % 360
                ray_world_angle = (r.direction + ray_local_angle) % 360

                angle_diff = (ray_world_angle - direct_angle + 180) % 360 - 180

                if abs(angle_diff) < 40 and dist < 70:
                    if dist < obstacle_dist:
                        obstacle_dist = dist
                        obstacle_found = True

                        to_target_x = math.cos(math.radians(direct_angle))
                        to_target_y = math.sin(math.radians(direct_angle))
                        to_obs_x = math.cos(math.radians(ray_world_angle))
                        to_obs_y = math.sin(math.radians(ray_world_angle))
                        cross = to_target_x * to_obs_y - to_target_y * to_obs_x
                        obstacle_side = 1 if cross > 0 else -1

        if not obstacle_found:
            return direct_angle
        safe_angle = (direct_angle - obstacle_side * 60) % 360

        return safe_angle


    def _find_nearest_obstacle_pos(self):
        """Позиция ближайшего препятствия через лидар"""
        r = self.env.robot
        min_dist = float('inf')
        best_angle = 0
        for i, (dist, obj_type) in enumerate(zip(r.lidar_distances, r.lidar_types)):
            if obj_type in ['car', 'tree', 'place', 'building', 'fence'] and dist < min_dist:
                min_dist = dist
                best_angle = i
        ray_angle = math.radians(r.direction + best_angle * 360 / r.num_rays)
        ox = r.x + min_dist * math.cos(ray_angle)
        oy = r.y + min_dist * math.sin(ray_angle)
        return (ox, oy)

    def _obstacle_still_ahead(self, cone_half_angle=30):
        """Есть ли препятствие впереди робота"""
        r = self.env.robot
        for i, (dist, obj_type) in enumerate(zip(r.lidar_distances, r.lidar_types)):
            if obj_type in ['car', 'tree', 'place', 'building', 'fence']:
                ray_local_angle = (i * 360 / r.num_rays + 180) % 360 - 180
                if abs(ray_local_angle) <= cone_half_angle and dist < self.WALL_CLEARANCE * 1.5:
                    return True, dist
        return False, float('inf')

    def _obstacle_behind(self, cone_half_angle=60):
        """Есть ли препятствие СЗАДИ робота"""
        r = self.env.robot
        for i, (dist, obj_type) in enumerate(zip(r.lidar_distances, r.lidar_types)):
            if obj_type in ['car', 'tree', 'place', 'building', 'fence']:
                ray_local_angle = (i * 360 / r.num_rays + 180) % 360 - 180
                if abs(abs(ray_local_angle) - 180) <= cone_half_angle and dist < self.WALL_CLEARANCE * 2.0:
                    return True, dist
        return False, float('inf')

    def _execute_recovery_phase(self):
        """Логика перехода между под-фазами"""
        r = self.env.robot
        self.recovery_phase_timer += 1

        if self.recovery_substate == self.RECOVERY_REVERSE_AWAY:
            has_obstacle, dist = self._obstacle_still_ahead(cone_half_angle=60)
            if not has_obstacle or self.recovery_phase_timer > 60:
                self.recovery_substate = self.RECOVERY_ALIGN
                self.recovery_phase_timer = 0

        elif self.recovery_substate == self.RECOVERY_ALIGN:
            if self.recovery_obstacle_pos is None:
                self.recovery_obstacle_pos = self._find_nearest_obstacle_pos()
            ox, oy = self.recovery_obstacle_pos
            angle_to_obstacle = math.degrees(math.atan2(oy - r.y, ox - r.x)) % 360
            angle_diff = (angle_to_obstacle - r.direction + 180) % 360 - 180
            if abs(angle_diff) < 5.0 or self.recovery_phase_timer > 90:
                has_back, back_dist = self._obstacle_behind(cone_half_angle=60)
                has_front, front_dist = self._obstacle_still_ahead(cone_half_angle=60)
                if has_back and back_dist < self.WALL_CLEARANCE * 1.5:
                    self.recovery_equidistant_mode = True
                    target_dist = (front_dist + back_dist) / 2.0
                    self.recovery_target_equidistant = target_dist
                    print(f"FSM: Обнаружено препятствие сзади на {back_dist:.0f} пикс. "
                          f"Выравниваем до {target_dist:.0f} пикс.")
                self.recovery_substate = self.RECOVERY_REVERSE_ALONG
                self.recovery_phase_timer = 0

        elif self.recovery_substate == self.RECOVERY_REVERSE_ALONG:
            has_front, front_dist = self._obstacle_still_ahead(cone_half_angle=60)
            has_back, back_dist = self._obstacle_behind(cone_half_angle=60)

            if self.recovery_equidistant_mode and self.recovery_target_equidistant is not None:
                if front_dist >= self.recovery_target_equidistant - 3.0:
                    print(f"FSM: Расстояния выравнены: перед={front_dist:.0f}, зад={back_dist:.0f}")
                    self.recovery_substate = self.RECOVERY_FINAL_TURN
                    self.recovery_phase_timer = 0
                elif self.recovery_phase_timer > 150:
                    self.recovery_substate = self.RECOVERY_FINAL_TURN
                    self.recovery_phase_timer = 0
            else:
                if not has_front and self.recovery_phase_timer > 30:
                    self.recovery_substate = self.RECOVERY_FINAL_TURN
                    self.recovery_phase_timer = 0
                elif self.recovery_phase_timer > 120:
                    self.recovery_substate = self.RECOVERY_FINAL_TURN
                    self.recovery_phase_timer = 0

        elif self.recovery_substate == self.RECOVERY_FINAL_TURN:
            if self.recovery_target_side is None:
                self.recovery_target_side = self._determine_target_side()
            target = self.env._shaping_target()
            if target is None:
                self._finish_recovery()
                return

            tx, ty = target
            angle_to_target = self._get_safe_target_angle(tx, ty)

            if self.recovery_target_side == 'behind':
                angle_diff = (angle_to_target - r.direction + 180) % 360 - 180
                if abs(angle_diff) < 5.0 or self.recovery_phase_timer > 90:
                    self._finish_recovery()
            else:
                desired_turn = -90.0 if self.recovery_target_side == 'right' else 90.0
                desired_direction = (r.direction + desired_turn) % 360
                angle_diff = (desired_direction - r.direction + 180) % 360 - 180
                if abs(angle_diff) < 5.0 or self.recovery_phase_timer > 60:
                    self._finish_recovery()

    def _finish_recovery(self):
        """Завершение recovery"""
        print(f"FSM: Recovery завершён за {self.state_timer} тиков")
        self.recovery_substate = None
        self.recovery_obstacle_pos = None
        self.recovery_target_side = None
        self.recovery_equidistant_mode = False
        self.recovery_target_equidistant = None
        self._transition(self.STATE_RL)

    def _determine_target_side(self):
        """С какой стороны от препятствия находится цель"""
        r = self.env.robot
        target = self.env._shaping_target()
        if target is None:
            return 'behind'

        tx, ty = target
        if self.recovery_obstacle_pos is not None:
            ox, oy = self.recovery_obstacle_pos
        else:
            ox, oy = self._find_nearest_obstacle_pos()

        forward_x = ox - r.x
        forward_y = oy - r.y
        forward_len = math.hypot(forward_x, forward_y)
        if forward_len < 1e-6:
            return 'behind'
        forward_x /= forward_len
        forward_y /= forward_len

        to_target_x = tx - r.x
        to_target_y = ty - r.y
        dot = forward_x * to_target_x + forward_y * to_target_y
        cross = forward_x * to_target_y - forward_y * to_target_x
        target_len = math.hypot(to_target_x, to_target_y)
        if target_len < 1e-6:
            return 'behind'

        cos_angle = dot / target_len
        if cos_angle < -0.5:
            return 'behind'
        elif cross > 0:
            return 'left'
        else:
            return 'right'

    def _execute_recovery_action(self):
        """Действие для текущей под-фазы recovery"""
        r = self.env.robot

        if self.recovery_substate == self.RECOVERY_REVERSE_AWAY:
            return np.array([self.REVERSE_SPEED, 0.3], dtype=np.float32)

        elif self.recovery_substate == self.RECOVERY_ALIGN:
            if self.recovery_obstacle_pos is None:
                self.recovery_obstacle_pos = self._find_nearest_obstacle_pos()
            ox, oy = self.recovery_obstacle_pos
            angle_to_obstacle = math.degrees(math.atan2(oy - r.y, ox - r.x)) % 360
            angle_diff = (angle_to_obstacle - r.direction + 180) % 360 - 180
            turn_speed = self.ALIGN_TURN_SPEED * self.env.MAX_ANGULAR_SPEED if angle_diff > 0 else -self.ALIGN_TURN_SPEED * self.env.MAX_ANGULAR_SPEED
            norm_turn = np.clip(turn_speed / self.env.MAX_ANGULAR_SPEED, -1.0, 1.0)
            return np.array([-0.3, norm_turn], dtype=np.float32)

        elif self.recovery_substate == self.RECOVERY_REVERSE_ALONG:
            return np.array([self.REVERSE_SPEED, 0.0], dtype=np.float32)

        elif self.recovery_substate == self.RECOVERY_FINAL_TURN:
            target = self.env._shaping_target()
            if target is None:
                return np.array([0.0, 0.0], dtype=np.float32)

            tx, ty = target
            if self.recovery_target_side == 'behind':
                angle_to_target = self._get_safe_target_angle(tx, ty)
                angle_diff = (angle_to_target - r.direction + 180) % 360 - 180
            else:
                desired_turn = -90.0 if self.recovery_target_side == 'right' else 90.0
                desired_direction = (r.direction + desired_turn) % 360
                angle_diff = (desired_direction - r.direction + 180) % 360 - 180

            turn_speed = self.ALIGN_TURN_SPEED * self.env.MAX_ANGULAR_SPEED if angle_diff > 0 else -self.ALIGN_TURN_SPEED * self.env.MAX_ANGULAR_SPEED
            norm_turn = np.clip(turn_speed / self.env.MAX_ANGULAR_SPEED, -1.0, 1.0)
            return np.array([0.2, norm_turn], dtype=np.float32)

        return np.array([-0.5, 0.8], dtype=np.float32)

    def _transition_to_nearest_target(self):
        r = self.env.robot
        visible_litters = [l for l in self.env.litters if self.env._is_litter_visible(l)]
        if visible_litters:
            nearest = min(visible_litters, key=lambda l: math.hypot(l.x - r.x, l.y - r.y))
            self.target_angle = self._get_safe_target_angle(nearest.x, nearest.y)
            self._transition(self.STATE_TURN_TO_LITTER)
            self.litter_turn_timer = 0
        elif self.env.wp_idx < len(self.env.route):
            wp_x, wp_y = self.env.route[self.env.wp_idx]
            self.target_angle = self._get_safe_target_angle(wp_x, wp_y)
            self._transition(self.STATE_TURN)
        else:
            self._transition(self.STATE_RL)

    def _transition(self, new_state):
        if self.current_state != new_state:
            self.current_state = new_state
            self.state_timer = 0

    def reset(self):
        self.current_state = self.STATE_RL
        self.state_timer = 0
        self.target_angle = 0.0
        self.freeze_cooldown_timer = 0
        self.litter_turn_timer = 0
        self.no_progress_steps = 0
        self.last_wp_idx = 0
        self.last_litter_count = 0
        self.last_target_dist = float('inf')
        self.no_progress_warned = False
        self.recovery_substate = None
        self.recovery_obstacle_pos = None
        self.recovery_phase_timer = 0
        self.recovery_frozen_during = None
        self.recovery_target_side = None
        self.recovery_equidistant_mode = False
        self.recovery_target_equidistant = None
        self.frozen_dynamic_target = None
        self.frozen_dynamic_timer = 0
        self.freeze_post_cooldown_timer = 0

    def get_state(self):
        return self.current_state


class RobotEnvWaypoints(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    WIDTH, HEIGHT = 800, 600
    BIKE_PATH_Y = 150
    PIXELS_PER_METER = 20  # 20 пикселей - 1 метр
    METERS_PER_PIXEL = 1.0 / PIXELS_PER_METER

    WAYPOINTS = [(100, 100), (700, 100), (700, 500),
                 (100, 500), (400, 300), (100, 500)]

    MAX_LINEAR_SPEED = 0.5
    MAX_ANGULAR_SPEED = 2.0

    COLLECT_RADIUS = 30
    COLLISION_RADIUS = 12

    WAYPOINT_RADIUS = 30
    LITTER_MAGNET_RADIUS = 100

    # Дефолтные награды (будут перезаписаны)
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

    def __init__(self, render_mode=None, max_steps=None, num_waypoints=None, stage=1,
                 env_kwargs=None, reward_kwargs=None):
        super().__init__()
        self.render_mode = render_mode

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
            'max_steps': 2000
        }

        if env_kwargs:
            cfg.update(env_kwargs)

        self.cfg = cfg

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
        self.MAX_ANGULAR_SPEED = 2.0
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

        self.fsm = HybridFSMController(self)

        if reward_kwargs:
            for key, value in reward_kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, float(value))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        cfg = self.cfg

        if 'litters_range' in cfg:
            self.NUM_LITTERS = int(self.np_random.integers(cfg['litters_range'][0], cfg['litters_range'][1] + 1))
        else:
            self.NUM_LITTERS = cfg.get('litters', 2)
        self.step_count = 0
        self.wp_idx = 0
        self.phase = 'route'
        self.prev_target_dist = None
        self.prev_litter_dist = None
        self.prev_wp_dist = None
        self._prev_visible_count = 0
        self.collision_cd = 0
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0

        self.turning_to_target = False
        self.target_angle = 0.0
        self.turn_steps = 0
        self.no_progress_steps = 0

        self.fsm.reset()
        self.fsm.last_wp_idx = self.wp_idx
        self.fsm.last_litter_count = len(self.litters)

        for key in self.stats:
            if key != 'episodes_completed':
                self.stats[key] = 0.0 if key == 'distance_traveled' else 0

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
        self._stuck_x, self._stuck_y = self.robot.x, self.robot.y
        self._stuck_steps = 0

        self.people, self.cyclists, self.litters = [], [], []
        self.cars, self.places, self.trees = [], [], []
        colors = ['red', 'blue', 'green', 'yellow', 'purple']

        self.route = list(self.WAYPOINTS[:cfg['wp']])
        self.robot.direction = float(self.np_random.uniform(0, 360))

        for _ in range(self.NUM_PEOPLE):
            x, y = self._free_spot(100, 700, 100, 500, min_dist=30, allow_paths=True, allow_parking=False)
            self.people.append(Person(x, y, str(self.np_random.choice(colors))))

        for _ in range(self.NUM_CYCLISTS):
            x, y = self._free_spot(50, 750, self.BIKE_PATH_Y - 10, self.BIKE_PATH_Y + 10, min_dist=40, allow_paths=True,
                                   allow_parking=False)
            direction = 1 if self.np_random.random() < 0.5 else -1
            self.cyclists.append(Cyclist(x, y, direction, str(self.np_random.choice(colors))))

        self._spawn_litter()
        self.initial_litters = max(1, len(self.litters))

        for _ in range(self.NUM_CARS):
            x, y = self._free_spot(715, 745, 415, 545, min_dist=35)
            self.cars.append(Car(x, y))

        place_types = ['скамейка', 'скамейка', 'турник', 'беседка']
        for _ in range(self.NUM_PLACES):
            x, y = self._free_spot(50, 750, 50, 550, min_dist=40, allow_paths=False, allow_parking=False)
            self.places.append(PublicPlace(x, y, str(self.np_random.choice(place_types))))

        tree_types = ['Ёлка', 'Дуб']
        for _ in range(self.NUM_TREES):
            x, y = self._free_spot(50, 680, 50, 550, min_dist=50, allow_paths=False, allow_parking=False)
            self.trees.append(Tree(x, y, str(self.np_random.choice(tree_types))))

        if self.route or self.litters:
            self._start_turning_to_target()

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
                for group in (self.people, self.cyclists, self.litters, self.cars, self.places, self.trees):
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

        # 1. Направление робота (2)
        obs.append(math.sin(dir_rad))
        obs.append(math.cos(dir_rad))

        # 2. Лидар (72)
        obs.extend(d / r.lidar_range for d in r.lidar_distances)

        # 3. Текущий waypoint (2)
        if self.phase == 'route' and self.wp_idx < len(self.route):
            obs.extend(self._to_robot_frame(*self.route[self.wp_idx]))
        else:
            obs.extend([0.0, 0.0])

        # 4. Ближайший видимый мусор (2)
        visible_litters = [l for l in self.litters if self._is_litter_visible(l)]
        if visible_litters:
            nearest_litter = min(visible_litters, key=lambda l: math.hypot(l.x - r.x, l.y - r.y))
            obs.extend(self._to_robot_frame(nearest_litter.x, nearest_litter.y))
        else:
            obs.extend([0.0, 0.0])

        # 5. Ближайшая видимая статика (машины, места, деревья) (2)
        static_objects = self.cars + self.places + self.trees
        obs.extend(self._get_nearest_visible_by_type(static_objects))

        # 6. Ближайшая видимая динамика (люди, велосипедисты) (2)
        dynamic_objects = self.people + self.cyclists
        obs.extend(self._get_nearest_visible_by_type(dynamic_objects))

        # 7. Доп. информация (3)
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
        target_angle = math.degrees(math.atan2(dy, dx)) % 360
        return target_angle

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
        return float((x_lo + x_hi) / 2), float((y_lo + y_hi) / 2)

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

    def step(self, action):
        self.step_count += 1
        r = self.robot

        for p in self.people:
            p.move(self.cars + self.places + self.trees, self.buildings, self.places)
        for c in self.cyclists:
            c.move(self.BIKE_PATH_Y, self.buildings)

        final_action = self.fsm.get_action(self._get_observation(), action)

        if self.fsm.current_state in (self.fsm.STATE_RECOVERY, self.fsm.STATE_FREEZE):
            target_linear = float(final_action[0]) * self.REAL_MAX_SPEED
        else:
            target_linear = (float(final_action[0]) + 1.0) / 2.0 * self.REAL_MAX_SPEED

        target_angular = float(final_action[1]) * self.MAX_ANGULAR_SPEED

        # Базовый штраф за время (в FREEZE тоже применяется, чтобы не было дисбаланса)
        reward = self.R_STEP

        terminated = False

        if target_linear > self.current_linear_vel:
            self.current_linear_vel = min(self.current_linear_vel + self.LINEAR_ACCELERATION, target_linear)
        else:
            self.current_linear_vel = max(self.current_linear_vel - self.LINEAR_ACCELERATION, target_linear)
        if abs(target_linear) < 0.01:
            self.current_linear_vel *= self.FRICTION

        angular_diff = target_angular - self.current_angular_vel
        if abs(angular_diff) > self.ANGULAR_ACCELERATION:
            self.current_angular_vel += self.ANGULAR_ACCELERATION * (1 if angular_diff > 0 else -1)
        else:
            self.current_angular_vel = target_angular
        if abs(target_angular) < 0.01:
            self.current_angular_vel *= self.FRICTION

        r.direction = (r.direction + self.current_angular_vel) % 360

        rad = math.radians(r.direction)
        next_x = r.x + self.current_linear_vel * math.cos(rad)
        next_y = r.y + self.current_linear_vel * math.sin(rad)

        hit_hard = False

        fence_margin = 15
        if next_x < fence_margin or next_x > self.WIDTH - fence_margin or \
                next_y < fence_margin or next_y > self.HEIGHT - fence_margin:
            hit_hard = True

        if not hit_hard:
            for obj in (self.cars + self.places + self.trees):
                if math.hypot(next_x - obj.x, next_y - obj.y) < self.COLLISION_RADIUS:
                    hit_hard = True
                    break
        if not hit_hard:
            for building in self.buildings:
                if building.distance_to_point(next_x, next_y) < self.COLLISION_RADIUS:
                    hit_hard = True
                    break

        if hit_hard:
            if self.fsm.current_state != self.fsm.STATE_FREEZE:
                if self.collision_cd == 0:
                    reward += self.R_COLLISION_STATIC
                    self.collision_cd = 50
                    self.stats['collisions_static'] += 1
                    self.fsm.current_state = self.fsm.STATE_RECOVERY
                    self.fsm.state_timer = 0
        else:
            r.x = next_x
            r.y = next_y

        if self.fsm.current_state == self.fsm.STATE_RL:
            min_dist = float('inf')
            for dist, obj_type in zip(r.lidar_distances, r.lidar_types):
                if obj_type is not None and dist < min_dist:
                    min_dist = dist
            if min_dist < 40.0:
                proximity_penalty = -((40.0 - min_dist) / 40.0) ** 2 * 1.0
                proximity_penalty = max(proximity_penalty, -1.0)
                reward += proximity_penalty

        hit_dynamic = False
        for obj in (self.people + self.cyclists):
            dist = math.hypot(r.x - obj.x, r.y - obj.y)
            if dist < self.COLLISION_RADIUS:
                hit_dynamic = True
                break

        if hit_dynamic and self.collision_cd == 0:
            if self.fsm.current_state != self.fsm.STATE_FREEZE:
                reward += self.R_COLLISION_DYNAMIC
                self.stats['collisions_dynamic'] += 1
                self.fsm.current_state = self.fsm.STATE_RECOVERY
                self.fsm.state_timer = 0
            self.collision_cd = 25

        self._update_sensors()

        if self.fsm.current_state == self.fsm.STATE_RL:
            if self.step_count % 50 == 0:
                moved_recently = math.hypot(r.x - self._stuck_x, r.y - self._stuck_y)

                if moved_recently < 30.0:
                    min_static_dist = float('inf')
                    for dist, obj_type in zip(r.lidar_distances, r.lidar_types):
                        if obj_type in ['car', 'tree', 'place', 'building', 'fence']:
                            if dist < min_static_dist:
                                min_static_dist = dist

                    if min_static_dist < 20.0:
                        print(f"FSM: Робот застрял! Движение: {moved_recently:.1f} пикс. за 50 шагов, "
                              f"ближайшее препятствие: {min_static_dist:.1f} пикс")
                        self.fsm._start_recovery()

                self._stuck_x, self._stuck_y = r.x, r.y

        collected = self._try_collect_litter()
        if collected:
            reward += self.R_LITTER * collected
            self.prev_target_dist = None
            self.prev_litter_dist = None
            if self.litters or self.wp_idx < len(self.route):
                self._start_turning_to_target()
                self.fsm.last_litter_count = len(self.litters)
                self.fsm.no_progress_steps = 0

        visible_litters = [l for l in self.litters if self._is_litter_visible(l)]
        if visible_litters:
            nearest_litter = min(visible_litters, key=lambda l: math.hypot(l.x - r.x, l.y - r.y))
            dist_to_litter = math.hypot(r.x - nearest_litter.x, r.y - nearest_litter.y)
            if dist_to_litter < 150:
                if self.prev_litter_dist is not None:
                    reward += 0.5 * (self.prev_litter_dist - dist_to_litter)
                self.prev_litter_dist = dist_to_litter
            else:
                self.prev_litter_dist = None

        if self.wp_idx < len(self.route):
            wp_x, wp_y = self.route[self.wp_idx]
            dist_to_wp = math.hypot(r.x - wp_x, r.y - wp_y)
            if self.prev_wp_dist is not None:
                delta = self.prev_wp_dist - dist_to_wp
                if delta > 0:
                    reward += 0.2 * delta
                else:
                    reward += 0.05 * delta
            self.prev_wp_dist = dist_to_wp
        else:
            self.prev_wp_dist = None

        if not visible_litters:
            target = self._shaping_target()
            if target is not None:
                dist = math.hypot(r.x - target[0], r.y - target[1])
                if self.prev_target_dist is not None:
                    reward += self.R_PROGRESS * (self.prev_target_dist - dist)
                self.prev_target_dist = dist
            else:
                self.prev_target_dist = None

        if self.wp_idx < len(self.route):
            wp_x, wp_y = self.route[self.wp_idx]
            if math.hypot(r.x - wp_x, r.y - wp_y) <= self.waypoint_radius:
                reward += self.R_WAYPOINT
                self.wp_idx += 1
                self.prev_target_dist = None
                self.prev_wp_dist = None
                self.stats['waypoints_completed'] += 1

                if self.wp_idx >= len(self.route):
                    self.wp_idx = 0
                    reward += self.R_ROUTE_DONE
                    self.stats['circles_completed'] += 1

                self._start_turning_to_target()
                self.fsm.last_wp_idx = self.wp_idx
                self.fsm.no_progress_steps = 0

        if self.collision_cd > 0:
            self.collision_cd -= 1

        truncated = self.step_count >= self.max_steps

        dist = math.hypot(r.x - self._prev_rx, r.y - self._prev_ry)
        self.stats['distance_traveled'] += dist
        self._prev_rx, self._prev_ry = r.x, r.y
        self.stats['total_reward'] += reward

        info = {
            'phase': self.phase,
            'wp_idx': self.wp_idx,
            'litters_left': len(self.litters),
            'success': False,
            'fsm_state': self.fsm.get_state(),
        }

        if self.render_mode == 'human':
            self._render_frame()

        if terminated or truncated:
            self.stats['episodes_completed'] += 1
            print(f"\n{'=' * 40}")
            print(f" ЭПИЗОД {self.stats['episodes_completed']} ЗАВЕРШЁН")
            print(f"{'=' * 40}")
            print(f" Мусор собрано:         {int(self.stats['litter_collected'])}")
            print(f" Waypoints пройдено:    {int(self.stats['waypoints_completed'])}")
            print(f" Кругов пройдено:       {int(self.stats['circles_completed'])}")
            print(f" Столкновения (стат):   {int(self.stats['collisions_static'])}")
            print(f" Столкновения (дин):    {int(self.stats['collisions_dynamic'])}")
            print(f" Пройдено расстояние:   {self.stats['distance_traveled']:.1f} пикс.")
            print(f" Общая награда:         {self.stats['total_reward']:.1f}")
            print(f"{'=' * 40}\n")

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
                if self._point_on_ray(r.x, r.y, obj_x, obj_y, obstacle.x, obstacle.y, obstacle_radius=15):
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
        pygame.draw.line(self.screen, (150, 150, 150), (0, self.BIKE_PATH_Y), (self.WIDTH, self.BIKE_PATH_Y), 12)
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

        for obj in self.trees:
            obj.draw(self.screen, self.robot, False)

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
    env = RobotEnvWaypoints(render_mode='human', stage=2)
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

def make_env(stage=1, env_kwargs=None, reward_kwargs=None):
    return RobotEnvWaypoints(render_mode=None, stage=stage,
                             env_kwargs=env_kwargs, reward_kwargs=reward_kwargs)
