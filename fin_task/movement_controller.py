import math
from typing import Tuple


class MovementController:
    MODE_CRUISE = 1
    MODE_SLOWDOWN = 2
    MODE_EMERGENCY = 3

    # физика
    MAX_LINEAR_SPEED = 0.7      # м/с
    MIN_DRIVE_SPEED = 0.1       # м/с
    NARROW_SPEED = 0.30         # м/с
    LINEAR_ACCEL = 0.7          # м/с**2
    LINEAR_DECEL = 1.5          # м/с**2
    REVERSE_SPEED = -0.45       # м/с
    MAX_ANGULAR_SPEED = 3.45    # рад/с
    MIN_ANGULAR_SPEED = 0.35    # рад/с
    PIXELS_PER_METER = 20.0
    FPS = 30.0

    # безопасность
    CORRIDOR_HALF_WIDTH = 19.0
    NARROW_HALF_WIDTH = 14.5
    PATH_LOOKAHEAD = 120.0
    STOP_DIST = 20.0
    SAFE_DIST = 35.0
    SAFE_DIST_DYNAMIC = 35.0
    CLEAR_DIST = 45.0
    BUFFER_ZONE = 45.0
    FENCE_KEEP_OUT = 25.0

    TTC_SLOW = 5.0
    TTC_EMERGENCY = 1.0
    TTC_HYSTERESIS = 0.5
    TTC_V_REF_MIN = 0.35
    EMERGENCY_TIMEOUT = 150

    FRONT_CONE = math.radians(50)
    FRONT_CONE_DIRECT = math.radians(15)
    WALL_CONE = math.radians(70)
    REAR_CONE = math.radians(45)

    VFH_THRESHOLD = 2.0
    VFH_SMOOTH_WINDOW = 5
    VFH_DYNAMIC_PENALTY = 4.0
    OBSTACLE_MEMORY_TTL = 60

    # RECOVERY по дуге
    RECOVERY_OMEGA = 0.35            # рад/с
    RECOVERY_TARGET_TURN = math.radians(80)
    RECOVERY_MAX_TICKS = 150
    RECOVERY_MIN_TICKS = 25
    REAR_HALF_WIDTH = 15.0
    REAR_STOP_DIST = 20.0

    CORNER_SLOW_ANGLE = math.radians(35)
    CORNER_SLOW_RANGE = math.radians(60)
    CORNER_MIN_SPEED = 0.25

    def __init__(self, env):
        self.env = env
        robot = getattr(env, 'robot', None)
        self.num_rays = robot.num_rays if robot is not None else 72
        self.lidar_range = robot.lidar_range if robot is not None else 300.0
        self.ray_step = 2 * math.pi / self.num_rays
        self.reset()

    def reset(self):
        self.current_mode = self.MODE_CRUISE
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0
        self.target_heading = 0.0
        self.min_front_dist = self.lidar_range
        self.min_front_dist_static = self.lidar_range
        self.min_front_dist_dynamic = self.lidar_range
        self.min_front_dist_direct = self.lidar_range
        self.ttc_direct = float('inf')
        self.ttc_front = float('inf')
        self.ttc_static = float('inf')
        self.ttc_dynamic = float('inf')
        self.front_obstacle_angle = None
        self.front_obstacle_type = None
        self.needs_recovery = False
        self.emergency_timer = 0
        self.last_side = None
        self.last_side_tick = -9999
        self.last_valley_start = None
        self.last_valley_width = 0
        self.last_valley_tick = -9999
        self.obstacle_memory = []
        self._obs_points = []
        self.path_clearance = self.lidar_range
        self.path_clearance_narrow = self.lidar_range
        self.goal_clearance = self.lidar_range
        self.narrow_ok = False
        self.recovery_finished = True
        self._rec_ticks = 0
        self._rec_turned = 0.0
        self._rec_side = None

    def hard_stop(self):
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0

    def brake(self):
        self._apply_physics(0.0, allow_rotate=False)
        return self.current_linear_vel, self.current_angular_vel

    def get_velocity(self, target_x: float, target_y: float,
                     is_safe_turn: bool = False,
                     is_recovery: bool = False,
                     heading_override: float = None,
                     speed_limit: float = None) -> Tuple[float, float]:
        r = self.env.robot
        dx, dy = target_x - r.x, target_y - r.y
        dist_target = math.hypot(dx, dy)
        target_angle = math.atan2(dy, dx)
        heading = math.radians(r.direction)
        self._scan_front(heading)

        if is_recovery:
            return self.get_recovery_velocity(target_x, target_y)

        if is_safe_turn:
            self.current_mode = self.MODE_EMERGENCY
            self.target_heading = self._norm(target_angle)
            self._apply_physics(0.0)
            return self.current_linear_vel, self.current_angular_vel

        self._update_mode(target_angle)
        speed = self._mode_speed(dist_target)

        err = abs(self._norm(target_angle - heading))
        if err > self.CORNER_SLOW_ANGLE:
            k = 1.0 - (err - self.CORNER_SLOW_ANGLE) / self.CORNER_SLOW_RANGE
            k = max(0.0, min(1.0, k))
            corner_speed = self.CORNER_MIN_SPEED + k * (self.MAX_LINEAR_SPEED - self.CORNER_MIN_SPEED)
            speed = min(speed, corner_speed)

        if speed_limit is not None:
            speed = min(speed, speed_limit)
        self.target_heading = (self._mode_heading(target_angle)
                               if heading_override is None else self._norm(heading_override))
        self._apply_physics(speed)
        return self.current_linear_vel, self.current_angular_vel

    def begin_recovery(self):
        self.recovery_finished = False
        self._rec_ticks = 0
        self._rec_turned = 0.0
        self._rec_side = None
        self.needs_recovery = False
        self.emergency_timer = 0

    def get_recovery_velocity(self, target_x=None, target_y=None):
        r = self.env.robot
        heading = math.radians(r.direction)
        self._scan_front(heading)
        self.current_mode = self.MODE_EMERGENCY
        self._rec_ticks += 1

        if self._rec_side is None:
            back = self._norm(heading + math.pi)
            cl_left = self._path_clearance(self._norm(back - 0.6), self.REAR_HALF_WIDTH, 90.0)
            cl_right = self._norm(back + 0.6)
            cl_right = self._path_clearance(cl_right, self.REAR_HALF_WIDTH, 90.0)
            self._rec_side = -1.0 if cl_left >= cl_right else 1.0

        rear = self._path_clearance(heading, self.REAR_HALF_WIDTH, 90.0, backward=True)
        rear_side = self._path_clearance(
            self._norm(heading + math.pi + self._rec_side * 0.5), self.REAR_HALF_WIDTH, 70.0)
        blocked = min(rear, rear_side) < self.REAR_STOP_DIST

        if blocked:
            speed = 0.0
            omega = self._rec_side * self.MAX_ANGULAR_SPEED * 0.5
        else:
            speed = self.REVERSE_SPEED
            omega = self._rec_side * self.RECOVERY_OMEGA

        dt = 1.0 / self.FPS
        if speed > self.current_linear_vel:
            self.current_linear_vel = min(self.current_linear_vel + self.LINEAR_ACCEL * dt, speed)
        else:
            self.current_linear_vel = max(self.current_linear_vel - self.LINEAR_DECEL * dt, speed)
        if abs(self.current_linear_vel) < 0.02 and abs(speed) < 0.02:
            self.current_linear_vel = 0.0

        self.current_angular_vel = omega
        self.target_heading = self._norm(heading + omega)
        self._rec_turned += abs(omega) * dt

        if (self._rec_turned >= self.RECOVERY_TARGET_TURN
                or self._rec_ticks >= self.RECOVERY_MAX_TICKS
                or (blocked and self._rec_ticks >= self.RECOVERY_MIN_TICKS)):
            self.recovery_finished = True
        return self.current_linear_vel, self.current_angular_vel

    def _v_ref(self) -> float:
        return max(abs(self.current_linear_vel), self.TTC_V_REF_MIN)

    def _scan_front(self, heading: float):
        r = self.env.robot
        self.min_front_dist = self.lidar_range
        self.min_front_dist_direct = self.lidar_range
        self.min_front_dist_static = self.lidar_range
        self.min_front_dist_dynamic = self.lidar_range
        self.front_obstacle_angle = None
        self.front_obstacle_type = None
        self._obs_points = []
        for i in range(self.num_rays):
            t = r.lidar_types[i]
            if t is None or t == 'litter':
                continue
            d = r.lidar_distances[i]
            a = heading + i * self.ray_step
            self._obs_points.append((d * math.cos(a), d * math.sin(a), t))
            angle_diff = abs(self._norm(a - heading))
            if angle_diff > self.FRONT_CONE:
                continue
            if d < self.min_front_dist:
                self.min_front_dist = d
                self.front_obstacle_angle = self._norm(a)
                self.front_obstacle_type = t
            if angle_diff <= self.FRONT_CONE_DIRECT:
                self.min_front_dist_direct = min(self.min_front_dist_direct, d)
            if t in ('person', 'cyclist'):
                self.min_front_dist_dynamic = min(self.min_front_dist_dynamic, d)
            else:
                self.min_front_dist_static = min(self.min_front_dist_static, d)
        v_ref = self._v_ref() * self.PIXELS_PER_METER
        self.ttc_front = self.min_front_dist / v_ref
        self.ttc_direct = self.min_front_dist_direct / v_ref
        self.ttc_static = self.min_front_dist_static / v_ref
        self.ttc_dynamic = self.min_front_dist_dynamic / v_ref

    def _path_clearance(self, angle, half_width=None, max_dist=None, backward=False):
        hw = self.CORRIDOR_HALF_WIDTH if half_width is None else half_width
        md = self.PATH_LOOKAHEAD if max_dist is None else max_dist
        ux, uy = math.cos(angle), math.sin(angle)
        sign = -1.0 if backward else 1.0
        best = md
        for (px, py, t) in self._obs_points:
            s = sign * (px * ux + py * uy)
            if s <= 0.0 or s >= best:
                continue
            if abs(-px * uy + py * ux) < hw:
                best = s
        return best

    def _dist_at(self, angle: float) -> float:
        r = self.env.robot
        idx = int(round((self._norm(angle) - math.radians(r.direction)) / self.ray_step)) % self.num_rays
        best = self.lidar_range
        for k in (-1, 0, 1):
            j = (idx + k) % self.num_rays
            if r.lidar_types[j] is not None and r.lidar_types[j] != 'litter':
                best = min(best, r.lidar_distances[j])
        return best

    def _rear_blocked(self) -> bool:
        heading = math.radians(self.env.robot.direction)
        return self._path_clearance(heading, self.REAR_HALF_WIDTH, 90.0,
                                    backward=True) < self.REAR_STOP_DIST

    def _update_mode(self, target_angle: float):
        heading = math.radians(self.env.robot.direction)
        v = abs(self.current_linear_vel)
        stop_dist = max(self.STOP_DIST, v * self.PIXELS_PER_METER * self.TTC_EMERGENCY + 6.0)
        self.path_clearance = self._path_clearance(heading)
        self.path_clearance_narrow = self._path_clearance(heading, self.NARROW_HALF_WIDTH)
        self.goal_clearance = self._path_clearance(target_angle)
        err = abs(self._norm(target_angle - heading))
        tight = self.path_clearance < self.SAFE_DIST
        self.narrow_ok = tight and self.path_clearance_narrow >= stop_dist and err < math.radians(25)
        danger = self.path_clearance < stop_dist and not self.narrow_ok
        slow = self.path_clearance < self.CLEAR_DIST or self.ttc_front < self.TTC_SLOW

        if self.current_mode == self.MODE_EMERGENCY:
            self.emergency_timer += 1
            if not danger:
                self.current_mode = self.MODE_SLOWDOWN if slow else self.MODE_CRUISE
                self.emergency_timer = 0
                self.needs_recovery = False
            elif self.emergency_timer > self.EMERGENCY_TIMEOUT:
                self.emergency_timer = 0
                self.needs_recovery = True
            return
        if danger:
            self.current_mode = self.MODE_EMERGENCY
            self.emergency_timer = 0
        elif slow:
            self.current_mode = self.MODE_SLOWDOWN
        else:
            self.current_mode = self.MODE_CRUISE
            self.needs_recovery = False

    def _mode_speed(self, dist_target: float) -> float:
        if self.current_mode == self.MODE_EMERGENCY:
            return 0.0
        if self.current_mode == self.MODE_SLOWDOWN:
            k = (self.path_clearance - self.STOP_DIST) / max(1.0, self.CLEAR_DIST - self.STOP_DIST)
            k = max(0.0, min(1.0, k))
            speed = self.NARROW_SPEED + k * (self.MAX_LINEAR_SPEED - self.NARROW_SPEED)
            if self.narrow_ok:
                speed = self.NARROW_SPEED
        else:
            speed = self.MAX_LINEAR_SPEED
        if dist_target < self.BUFFER_ZONE:
            speed = min(speed, max(self.NARROW_SPEED,
                                   self.MAX_LINEAR_SPEED * dist_target / self.BUFFER_ZONE))
        return speed

    def _mode_heading(self, target_angle: float) -> float:
        if self.current_mode == self.MODE_EMERGENCY:
            return self._get_vfh_heading(target_angle)
        if self.narrow_ok:
            return target_angle
        if self.current_mode == self.MODE_SLOWDOWN and self.path_clearance < self.CLEAR_DIST:
            return self._wall_following_heading(target_angle)
        return target_angle

    def _wall_following_heading(self, target_angle: float) -> float:
        r = self.env.robot
        heading = math.radians(r.direction)
        best_d, obs_a = self.lidar_range, None
        for i in range(self.num_rays):
            if r.lidar_types[i] is None or r.lidar_types[i] == 'litter':
                continue
            a = heading + i * self.ray_step
            if abs(self._norm(a - heading)) > self.WALL_CONE:
                continue
            if r.lidar_distances[i] < best_d:
                best_d, obs_a = r.lidar_distances[i], self._norm(a)
        if obs_a is None:
            return target_angle
        p1 = self._norm(obs_a + math.pi / 2)
        p2 = self._norm(obs_a - math.pi / 2)
        d1, d2 = abs(self._norm(p1 - target_angle)), abs(self._norm(p2 - target_angle))
        first, second = (p1, p2) if d1 < d2 else (p2, p1)
        chosen = first if self._dist_at(first) > self.SAFE_DIST else second
        side = 'left' if self._norm(chosen - target_angle) > 0 else 'right'
        self.last_side, self.last_side_tick = side, self.env.step_count
        if best_d < self.SAFE_DIST * 1.2:
            away = self._norm(obs_a + math.pi)
            chosen = self._norm(chosen + 0.35 * self._norm(away - chosen))
        return chosen

    def _get_vfh_heading(self, target_angle: float) -> float:
        r = self.env.robot
        n = self.num_rays
        heading = math.radians(r.direction)
        now = self.env.step_count

        self.obstacle_memory = [(a, d, t) for a, d, t in self.obstacle_memory
                                if now - t < self.OBSTACLE_MEMORY_TTL]
        for i in range(n):
            t = r.lidar_types[i]
            if t is not None and t != 'litter' and r.lidar_distances[i] < self.CLEAR_DIST:
                self.obstacle_memory.append((self._norm(heading + i * self.ray_step),
                                             r.lidar_distances[i], now))

        costs = [0.0] * n
        for i in range(n):
            d = r.lidar_distances[i]
            t = r.lidar_types[i]
            if t is None or t == 'litter':
                costs[i] = 0.0
            elif t == 'fence':
                costs[i] = (self.FENCE_KEEP_OUT / max(d, 1.0)) ** 3
            elif t in ('person', 'cyclist'):
                costs[i] = self.VFH_DYNAMIC_PENALTY * (self.SAFE_DIST_DYNAMIC / max(d, 1.0)) ** 2
            else:
                costs[i] = (self.SAFE_DIST / max(d, 1.0)) ** 2

        for mem_angle, mem_dist, mem_tick in self.obstacle_memory:
            idx = int(round((self._norm(mem_angle) - heading) / self.ray_step)) % n
            decay = 1.0 - (now - mem_tick) / self.OBSTACLE_MEMORY_TTL
            mem_cost = (self.SAFE_DIST / max(mem_dist, 1.0)) ** 2 * decay
            for k in range(-2, 3):
                j = (idx + k) % n
                costs[j] = max(costs[j], mem_cost * 0.5)

        w = self.VFH_SMOOTH_WINDOW
        smoothed = [0.0] * n
        for i in range(n):
            total = 0.0
            for k in range(-w, w + 1):
                total += costs[(i + k) % n]
            smoothed[i] = total / (2 * w + 1)

        free = [smoothed[i] < self.VFH_THRESHOLD for i in range(n)]
        if all(free):
            return self._norm(target_angle)
        starts = [i for i in range(n) if free[i] and not free[(i - 1) % n]]
        if not starts:
            return self._norm(target_angle + math.pi)

        valleys = []
        for s in starts:
            width = 0
            while width < n and free[(s + width) % n]:
                width += 1
            valleys.append((s, width))

        best_valley, best_score = None, float('inf')
        for s, width in valleys:
            center = self._norm(heading + (s + width / 2.0) * self.ray_step)
            diff = abs(self._norm(center - target_angle))
            score = math.degrees(diff) + 20.0 / max(width, 1)
            if self.last_side is not None and now - self.last_side_tick < 90:
                side = 'left' if self._norm(center - target_angle) > 0 else 'right'
                if side == self.last_side:
                    score -= 10.0
            if score < best_score:
                best_score, best_valley = score, (s, width)

        if (self.last_valley_start is not None and now - self.last_valley_tick < 60):
            ls, lw = self.last_valley_start, self.last_valley_width
            if lw > 0 and all(free[(ls + k) % n] for k in range(lw)):
                best_valley = (ls, lw)

        start, width = best_valley
        self.last_valley_start, self.last_valley_width = start, width
        self.last_valley_tick = now
        side = 'left' if self._norm(self._norm(heading + (start + width / 2.0) * self.ray_step)
                                    - target_angle) > 0 else 'right'
        self.last_side, self.last_side_tick = side, now

        best_angle, best_diff = None, float('inf')
        for k in range(width):
            idx = (start + k) % n
            a = self._norm(heading + idx * self.ray_step)
            diff = abs(self._norm(a - target_angle))
            if diff < best_diff:
                best_diff, best_angle = diff, a
        return self._norm(best_angle)

    def _apply_physics(self, target_speed: float, allow_rotate: bool = True):
        dt = 1.0 / self.FPS
        if target_speed > self.current_linear_vel:
            self.current_linear_vel = min(self.current_linear_vel + self.LINEAR_ACCEL * dt, target_speed)
        else:
            self.current_linear_vel = max(self.current_linear_vel - self.LINEAR_DECEL * dt, target_speed)
        if abs(self.current_linear_vel) < 0.02 and abs(target_speed) < 0.02:
            self.current_linear_vel = 0.0
        if not allow_rotate:
            self.current_angular_vel = 0.0
            return
        v = abs(self.current_linear_vel)
        k = (self.MAX_ANGULAR_SPEED - self.MIN_ANGULAR_SPEED) / self.MAX_LINEAR_SPEED
        w_max = max(self.MIN_ANGULAR_SPEED, self.MAX_ANGULAR_SPEED - k * v)
        err = self._norm(self.target_heading - math.radians(self.env.robot.direction))
        if abs(err) < math.radians(8):
            w = w_max * err / math.radians(8)
        else:
            w = math.copysign(w_max, err)
        self.current_angular_vel = max(-w_max, min(w_max, w))

    @staticmethod
    def _norm(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi