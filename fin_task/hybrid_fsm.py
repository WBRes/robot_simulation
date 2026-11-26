import math
from navigation_algorithms import (
    CostMapSafeTurn,
    GlobalPathPlanner,
    DynamicDangerDetector,
    norm_angle,
)
from movement_controller import MovementController


class HybridFSMController:
    STATE_DRIVE = "DRIVE"
    STATE_SAFE_TURN = "SAFE_TURN"
    STATE_DETOUR = "DETOUR"
    STATE_RECOVERY = "RECOVERY"
    STATE_FREEZE = "FREEZE"

    STATE_RL = "DRIVE"
    STATE_TURN = "SAFE_TURN"
    STATE_TURN_TO_LITTER = "SAFE_TURN"

    FREEZE_DISTANCE = 30.0
    FREEZE_ANGLE_CONE = 60.0
    FREEZE_COOLDOWN = 40
    FREEZE_POST_COOLDOWN = 120
    FREEZE_TIMEOUT = 90

    STILL_AS_STATIC_TICKS = 45

    TURN_TOL = math.radians(10)
    TURN_TIMEOUT = 90
    EMERGENCY_TO_TURN = 20

    DETOUR_TIMEOUT = 240
    DETOUR_REPLAN_TICKS = 20
    DETOUR_EXIT_CLEAR = 70.0

    STALL_TICKS = 60
    STALL_DIST = 6.0
    NO_PROGRESS_LIMIT = 400
    MAX_RECOVERY_FAILS = 3
    WAYPOINT_SKIP_FAILS = 12

    PATH_REPLAN_TICKS = 90
    PATH_ADVANCE_DIST = 30.0
    PATH_MAX_DEVIATION = 60.0
    PATH_GOAL_TOL = 25.0

    def __init__(self, env):
        self.env = env
        self.movement = MovementController(env)
        self.safe_turn_algo = CostMapSafeTurn()
        self.planner = GlobalPathPlanner(env)
        self.dynamic_danger_algo = DynamicDangerDetector(
            freeze_distance=self.FREEZE_DISTANCE,
            freeze_angle_cone_deg=self.FREEZE_ANGLE_CONE,
        )
        self.recovery_count = 0
        self.reset()

    def reset(self):
        self.current_state = self.STATE_DRIVE
        self.state_timer = 0
        self.current_target = None
        self.target_type = None
        self.target_obj = None
        self.locked_wp_idx = None
        self.freeze_cooldown_timer = 0
        self.freeze_post_cooldown_timer = 0
        self.freeze_timer = 0
        self.no_progress_steps = 0
        self.last_dist = float('inf')
        self.movement.reset()
        self.safe_turn_algo.reset_memory()
        self.planner.invalidate()
        self.detour_angle = None
        self.detour_ticks = 0
        self.recovery_fails = 0
        self.recovery_count = 0
        self._probe = None
        self.path = None
        self.path_idx = 0
        self.path_goal = None
        self.path_tick = -10 ** 9

        self._dyn_track = {}

    #  Своя оценка скорости и "стояния" людей/велосипедистов
    def _update_dyn_tracks(self):
        tracks = self._dyn_track
        alive = []
        for obj in list(self.env.people) + list(self.env.cyclists):
            key = id(obj)
            alive.append(key)
            rec = tracks.get(key)
            if rec is None:
                tracks[key] = {'obj': obj, 'x': obj.x, 'y': obj.y,
                               'vx': 0.0, 'vy': 0.0, 'still': 0}
                continue
            dx = obj.x - rec['x']
            dy = obj.y - rec['y']
            if math.hypot(dx, dy) < 0.05:
                rec['still'] += 1
            else:
                rec['still'] = 0
            rec['vx'] = 0.5 * rec['vx'] + 0.5 * dx
            rec['vy'] = 0.5 * rec['vy'] + 0.5 * dy
            rec['x'], rec['y'] = obj.x, obj.y
        if len(tracks) > len(alive):
            s = set(alive)
            for k in list(tracks.keys()):
                if k not in s:
                    del tracks[k]

    def _dynamic_danger(self) -> bool:
        if self.freeze_post_cooldown_timer > 0:
            return False
        if self.current_state == self.STATE_RECOVERY:
            return False
        r = self.env.robot
        for obj in list(self.env.people) + list(self.env.cyclists):
            if getattr(obj, 'state', None) == 'resting':
                continue
            rec = self._dyn_track.get(id(obj))
            still = rec['still'] if rec else 0
            if still >= self.STILL_AS_STATIC_TICKS:
                continue
            dist = math.hypot(obj.x - r.x, obj.y - r.y)
            if dist > self.FREEZE_DISTANCE:
                continue
            a = math.degrees(math.atan2(obj.y - r.y, obj.x - r.x)) % 360
            diff = (a - r.direction + 180) % 360 - 180
            if abs(diff) >= self.FREEZE_ANGLE_CONE:
                continue
            vx = rec['vx'] if rec else 0.0
            vy = rec['vy'] if rec else 0.0
            if math.hypot(vx, vy) < 0.05:
                return True
            if vx * (r.x - obj.x) + vy * (r.y - obj.y) < 0:
                continue
            return True
        return False

    def get_action(self, obs=None, rl_action=None):
        self.state_timer += 1
        if self.freeze_post_cooldown_timer > 0:
            self.freeze_post_cooldown_timer -= 1
        self._update_dyn_tracks()
        self._sync_target()

        if self._dynamic_danger():
            if self.current_state != self.STATE_FREEZE:
                self._enter(self.STATE_FREEZE)
                self.freeze_cooldown_timer = 0
                self.freeze_timer = 0
            return self.movement.brake()

        if self.current_state == self.STATE_FREEZE:
            self.freeze_timer += 1
            if self.freeze_cooldown_timer < self.FREEZE_COOLDOWN:
                self.freeze_cooldown_timer += 1
                return self.movement.brake()
            if self.freeze_timer > self.FREEZE_TIMEOUT:
                self.freeze_post_cooldown_timer = self.FREEZE_POST_COOLDOWN
            self.freeze_timer = 0
            self._enter(self.STATE_SAFE_TURN)
            return self.movement.brake()

        if self.current_target is None:
            return self.movement.brake()
        tx, ty = self.current_target

        if self.current_state == self.STATE_RECOVERY:
            lin, ang = self.movement.get_recovery_velocity(tx, ty)
            if self.movement.recovery_finished:
                self.movement.needs_recovery = False
                self.path_tick = -10 ** 9
                self._enter(self.STATE_SAFE_TURN)
            return lin, ang

        if self.current_state == self.STATE_SAFE_TURN:
            return self._step_safe_turn(tx, ty)

        if self.current_state == self.STATE_DETOUR:
            return self._step_detour(tx, ty)

        # DRIVE
        px, py = self._drive_target(tx, ty)
        lin, ang = self.movement.get_velocity(px, py)
        if (self.movement.current_mode == self.movement.MODE_EMERGENCY
                and self.state_timer > self.EMERGENCY_TO_TURN):
            self._enter(self.STATE_SAFE_TURN)
        if self.movement.needs_recovery:
            self._on_stuck(tx, ty)
        self._track_progress(tx, ty)
        return lin, ang

    #  A*: маршрут
    def _ensure_path(self, tx, ty):
        now = self.env.step_count
        r = self.env.robot
        need = (self.path is None or self.path_goal is None
                or math.hypot(tx - self.path_goal[0], ty - self.path_goal[1]) > self.PATH_GOAL_TOL
                or now - self.path_tick > self.PATH_REPLAN_TICKS)
        if not need and self.path:
            if self._dist_to_path(r.x, r.y) > self.PATH_MAX_DEVIATION:
                need = True
        if need:
            self.path = self.planner.plan(r.x, r.y, tx, ty)
            self.path_idx = 0
            self.path_goal = (tx, ty)
            self.path_tick = now

    def _dist_to_path(self, x, y):
        if not self.path or len(self.path) < 2:
            return 0.0
        best = float('inf')
        for i in range(len(self.path) - 1):
            x1, y1 = self.path[i]
            x2, y2 = self.path[i + 1]
            dx, dy = x2 - x1, y2 - y1
            L2 = dx * dx + dy * dy
            if L2 < 1e-6:
                d = math.hypot(x - x1, y - y1)
            else:
                t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / L2))
                d = math.hypot(x - (x1 + t * dx), y - (y1 + t * dy))
            best = min(best, d)
        return best

    def _drive_target(self, tx, ty):
        self._ensure_path(tx, ty)
        if not self.path:
            return (tx, ty)
        r = self.env.robot
        while self.path_idx < len(self.path) - 1:
            px, py = self.path[self.path_idx]
            nx, ny = self.path[self.path_idx + 1]
            if (math.hypot(px - r.x, py - r.y) < self.PATH_ADVANCE_DIST
                    or self.planner.line_free(r.x, r.y, nx, ny)):
                self.path_idx += 1
            else:
                break
        return self.path[self.path_idx]

    #  Локальные манёвры
    def _plan_local(self, px, py):
        r = self.env.robot
        return self.safe_turn_algo.plan(
            robot_x=r.x, robot_y=r.y, robot_direction_deg=r.direction,
            lidar_distances=r.lidar_distances, lidar_types=r.lidar_types,
            num_rays=r.num_rays, ray_step_rad=self.movement.ray_step,
            target_x=px, target_y=py, tick=self.env.step_count)

    def _step_safe_turn(self, tx, ty):
        r = self.env.robot
        self.movement._scan_front(math.radians(r.direction))
        px, py = self._drive_target(tx, ty)
        res = self._plan_local(px, py)
        if res['blocked']:
            self.path_tick = -10 ** 9
            self._on_stuck(tx, ty)
            return self.movement.brake()
        self.detour_angle = res['angle']
        err = abs(norm_angle(self.detour_angle - math.radians(r.direction)))
        if err < self.TURN_TOL or self.state_timer > self.TURN_TIMEOUT:
            self._enter(self.STATE_DETOUR)
            self.detour_ticks = 0
            return self._step_detour(tx, ty)
        sx = r.x + 100 * math.cos(self.detour_angle)
        sy = r.y + 100 * math.sin(self.detour_angle)
        return self.movement.get_velocity(sx, sy, is_safe_turn=True)

    def _step_detour(self, tx, ty):
        r = self.env.robot
        self.detour_ticks += 1
        self.movement._scan_front(math.radians(r.direction))
        px, py = self._drive_target(tx, ty)
        a = math.atan2(py - r.y, px - r.x)
        d = math.hypot(px - r.x, py - r.y)
        if self.movement._path_clearance(a, max_dist=120.0) >= min(self.DETOUR_EXIT_CLEAR, d):
            self._finish_detour()
            return self.movement.get_velocity(px, py)
        if self.detour_ticks > self.DETOUR_TIMEOUT:
            self.path_tick = -10 ** 9
            self._finish_detour()
            return self.movement.get_velocity(px, py)
        if (self.movement.needs_recovery
                or self.state_timer % self.DETOUR_REPLAN_TICKS == 0):
            res = self._plan_local(px, py)
            if res['blocked']:
                self.path_tick = -10 ** 9
                self._on_stuck(tx, ty)
                return self.movement.brake()
            if abs(norm_angle(res['angle'] - self.detour_angle)) > math.radians(35):
                self.detour_angle = res['angle']
                self._enter(self.STATE_SAFE_TURN)
                return self.movement.brake()
            self.detour_angle = res['angle']
            self.movement.needs_recovery = False
        sx = r.x + 120 * math.cos(self.detour_angle)
        sy = r.y + 120 * math.sin(self.detour_angle)
        lin, ang = self.movement.get_velocity(sx, sy)
        self._track_progress(tx, ty)
        return lin, ang

    def _finish_detour(self):
        self.detour_angle = None
        self.detour_ticks = 0
        self.no_progress_steps = 0
        self.last_dist = float('inf')
        self._probe = None
        self._enter(self.STATE_DRIVE)

    def _on_stuck(self, tx, ty):
        self.recovery_fails += 1
        self.movement.needs_recovery = False
        if self.recovery_fails >= self.WAYPOINT_SKIP_FAILS and self.target_type == 'wp':
            print("FSM: waypoint недостижим, пропускаю")
            self.recovery_fails = 0
            self.env.wp_idx += 1
            if self.env.wp_idx >= len(self.env.route):
                self.env.phase = 'cleanup'
            self._clear_target()
            self.path_tick = -10 ** 9
            self._enter(self.STATE_DRIVE)
            return
        if self.recovery_fails >= self.MAX_RECOVERY_FAILS:
            self.path_tick = -10 ** 9
            self.recovery_fails = 0
        self.trigger_recovery()

    def _sync_target(self):
        env = self.env
        if self.target_type == 'litter':
            if self.target_obj not in env.litters:
                self._clear_target()
        elif self.target_type == 'wp':
            if env.wp_idx != self.locked_wp_idx or env.wp_idx >= len(env.route):
                self._clear_target()

        if self.current_target is None or self.target_type == 'wp':
            visible = [l for l in env.litters if env._is_litter_visible(l)]
            if visible:
                r = env.robot
                nearest = min(visible, key=lambda l: math.hypot(l.x - r.x, l.y - r.y))
                if self.target_obj is not nearest:
                    self.path_tick = -10 ** 9
                self.current_target = (nearest.x, nearest.y)
                self.target_type = 'litter'
                self.target_obj = nearest
                self.last_dist = float('inf')
                if self.current_state == self.STATE_DRIVE:
                    self._enter(self.STATE_SAFE_TURN)
                return

        if self.current_target is None and env.wp_idx < len(env.route):
            self.current_target = env.route[env.wp_idx]
            self.target_type = 'wp'
            self.locked_wp_idx = env.wp_idx
            self.last_dist = float('inf')
            self.path_tick = -10 ** 9
            if self.current_state == self.STATE_DRIVE:
                self._enter(self.STATE_SAFE_TURN)

    def _clear_target(self):
        self.current_target = None
        self.target_type = None
        self.target_obj = None
        self.locked_wp_idx = None
        self.last_dist = float('inf')
        self.detour_angle = None
        self._probe = None
        self.path = None
        self.path_goal = None
        self.path_tick = -10 ** 9

    def invalidate_target(self):
        self._clear_target()

    def _track_progress(self, tx, ty):
        r = self.env.robot
        now = self.env.step_count
        d = math.hypot(tx - r.x, ty - r.y)
        if d < self.last_dist - 2.0:
            self.last_dist = d
            self.no_progress_steps = 0
            self.recovery_fails = 0
            self._probe = (r.x, r.y, now)
            return
        self.no_progress_steps += 1
        if self._probe is None:
            self._probe = (r.x, r.y, now)
        px, py, pt = self._probe
        if now - pt >= self.STALL_TICKS:
            moved = math.hypot(r.x - px, r.y - py)
            self._probe = (r.x, r.y, now)
            if moved < self.STALL_DIST:
                self.no_progress_steps = 0
                self._on_stuck(tx, ty)
            elif self.no_progress_steps > self.NO_PROGRESS_LIMIT:
                self.no_progress_steps = 0
                self.path_tick = -10 ** 9
                self._ensure_path(tx, ty)

    def _enter(self, state):
        if self.current_state != state:
            self.current_state = state
            self.state_timer = 0

    def trigger_recovery(self):
        # защита от двойного вызова из train_env.step()
        if self.current_state == self.STATE_RECOVERY and self.state_timer < 10:
            return
        self.movement.begin_recovery()
        self.no_progress_steps = 0
        self.recovery_count += 1
        self._enter(self.STATE_RECOVERY)
        self.state_timer = 0

    def _angle_to(self, tx, ty):
        r = self.env.robot
        a = math.atan2(ty - r.y, tx - r.x)
        return norm_angle(a - math.radians(r.direction))