import math
import heapq
from typing import List, Optional, Tuple


def norm_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi



#  Обычный планировщик: выбор проезжаемого направления по лидару
class CostMapSafeTurn:
    CELL_SIZE = 8.0
    MAP_RADIUS = 120.0
    DIRECTIONS = 72
    ROBOT_RADIUS = 13.0
    INFLATE_MARGIN = 6.0
    SQUEEZE_MARGIN = 1.5
    DYNAMIC_EXTRA = 5.0
    COMMIT_DIST = 55.0
    LOOKAHEAD = 110.0
    W_GOAL = 1.0
    W_TURN = 0.30
    W_CLEAR = 0.20
    HYSTERESIS_BONUS = 0.30
    HYSTERESIS_TTL = 120

    def __init__(self, cell_size=CELL_SIZE, map_radius=MAP_RADIUS,
                 directions=DIRECTIONS, robot_radius=ROBOT_RADIUS,
                 commit_dist=COMMIT_DIST, lookahead=LOOKAHEAD):
        self.cell_size = cell_size
        self.map_radius = map_radius
        self.num_dirs = directions
        self.robot_radius = robot_radius
        self.commit_dist = commit_dist
        self.lookahead = lookahead
        self.grid_n = int(round(2 * self.map_radius / self.cell_size))
        self._disk_normal = self._make_disk(self.robot_radius + self.INFLATE_MARGIN)
        self._disk_squeeze = self._make_disk(self.robot_radius + self.SQUEEZE_MARGIN)
        self._disk_dyn = self._make_disk(self.robot_radius + self.INFLATE_MARGIN + self.DYNAMIC_EXTRA)
        self.reset_memory()

    def reset_memory(self):
        self._last_angle = None
        self._last_tick = -10 ** 9

    def _make_disk(self, radius_px):
        k = int(math.ceil(radius_px / self.cell_size)) + 1
        offs = []
        for oy in range(-k, k + 1):
            for ox in range(-k, k + 1):
                if math.hypot(ox, oy) * self.cell_size <= radius_px + self.cell_size * 0.5:
                    offs.append((ox, oy))
        return offs

    def plan(self, robot_x, robot_y, robot_direction_deg,
             lidar_distances, lidar_types, num_rays, ray_step_rad,
             target_x, target_y, tick=0):
        n, cs, R = self.grid_n, self.cell_size, self.map_radius
        heading = math.radians(robot_direction_deg)
        g_norm = [[0] * n for _ in range(n)]
        g_sq = [[0] * n for _ in range(n)]

        for i in range(num_rays):
            t = lidar_types[i]
            if t is None or t == 'litter':
                continue
            d = lidar_distances[i]
            if d > R:
                continue
            a = heading + i * ray_step_rad
            gx = int((d * math.cos(a) + R) / cs)
            gy = int((d * math.sin(a) + R) / cs)
            dyn = t in ('person', 'cyclist')
            for ox, oy in (self._disk_dyn if dyn else self._disk_normal):
                x, y = gx + ox, gy + oy
                if 0 <= x < n and 0 <= y < n:
                    g_norm[y][x] = 1
            for ox, oy in (self._disk_dyn if dyn else self._disk_squeeze):
                x, y = gx + ox, gy + oy
                if 0 <= x < n and 0 <= y < n:
                    g_sq[y][x] = 1

        step = cs * 0.75
        n_steps = int(self.lookahead / step)
        results = []
        for k in range(self.num_dirs):
            a = -math.pi + k * (2 * math.pi / self.num_dirs)
            ca, sa = math.cos(a), math.sin(a)
            c_n = c_s = self.lookahead
            hit_n = hit_s = False
            for j in range(1, n_steps + 1):
                s = j * step
                gx = int((s * ca + R) / cs)
                gy = int((s * sa + R) / cs)
                if not (0 <= gx < n and 0 <= gy < n):
                    break
                if not hit_n and g_norm[gy][gx]:
                    c_n, hit_n = s, True
                if not hit_s and g_sq[gy][gx]:
                    c_s, hit_s = s, True
                if hit_n and hit_s:
                    break
            results.append((a, c_n, c_s))

        goal_a = math.atan2(target_y - robot_y, target_x - robot_x)
        goal_d = math.hypot(target_x - robot_x, target_y - robot_y)
        need = min(self.commit_dist, max(12.0, goal_d))

        def score_of(a, clearance):
            s = (self.W_GOAL * abs(norm_angle(a - goal_a))
                 + self.W_TURN * abs(norm_angle(a - heading))
                 - self.W_CLEAR * min(clearance, self.lookahead) / self.lookahead)
            if (self._last_angle is not None
                    and tick - self._last_tick < self.HYSTERESIS_TTL
                    and abs(norm_angle(a - self._last_angle)) < math.radians(20)):
                s -= self.HYSTERESIS_BONUS
            return s

        cand = [(score_of(a, cn), a, cn) for a, cn, cs_ in results if cn >= need]
        squeeze = False
        if not cand:
            cand = [(score_of(a, cs_), a, cs_) for a, cn, cs_ in results if cs_ >= need * 0.75]
            squeeze = True
        if cand:
            cand.sort()
            sc, a, cl = cand[0]
            self._last_angle, self._last_tick = a, tick
            return {'angle': a, 'clearance': cl, 'blocked': False, 'squeeze': squeeze, 'score': sc}

        a, cn, cs_ = max(results, key=lambda r: r[2])
        return {'angle': a, 'clearance': cs_, 'blocked': True, 'squeeze': True, 'score': float('inf')}

    def find_safe_direction(self, robot_x, robot_y, robot_direction_deg,
                            lidar_distances, lidar_types, num_rays, ray_step_rad,
                            target_x, target_y, tick=0):
        res = self.plan(robot_x, robot_y, robot_direction_deg, lidar_distances,
                        lidar_types, num_rays, ray_step_rad, target_x, target_y, tick)
        return None if res['blocked'] else res['angle']



#  Планировщик A*: по известной карте (упрощение)
class GlobalPathPlanner:
    CELL = 10.0
    HARD_CLEAR = 22.0
    SOFT_CLEAR = 40.0
    SOFT_WEIGHT = 4.0
    MAX_EXPANSIONS = 15000
    GOAL_SEARCH_R = 80.0

    CAR_R = 20.0
    PLACE_R = 14.0
    TREE_R = 12.0

    def __init__(self, env):
        self.env = env
        self.nx = int(math.ceil(env.WIDTH / self.CELL))
        self.ny = int(math.ceil(env.HEIGHT / self.CELL))
        self.blocked = [False] * (self.nx * self.ny)
        self.penalty = [0.0] * (self.nx * self.ny)
        self._sig = None
        self.last_path_len = 0.0
        self.last_expansions = 0

    # построение карты
    def _signature(self):
        e = self.env
        return (tuple((round(b.x, 1), round(b.y, 1), len(b.polygon)) for b in e.buildings),
                tuple((round(o.x, 1), round(o.y, 1)) for o in (e.cars + e.places + e.trees)))

    def invalidate(self):
        self._sig = None

    def rebuild(self, force=False):
        sig = self._signature()
        if not force and sig == self._sig:
            return
        self._sig = sig
        e = self.env
        fm = getattr(e, 'FENCE_MARGIN', 12.0)
        circles = ([(o.x, o.y, self.CAR_R) for o in e.cars]
                   + [(o.x, o.y, self.PLACE_R) for o in e.places]
                   + [(o.x, o.y, self.TREE_R) for o in e.trees])
        span = max(1e-6, self.SOFT_CLEAR - self.HARD_CLEAR)
        for cy in range(self.ny):
            wy = (cy + 0.5) * self.CELL
            for cx in range(self.nx):
                wx = (cx + 0.5) * self.CELL
                d = min(wx - fm, e.WIDTH - fm - wx, wy - fm, e.HEIGHT - fm - wy)
                for b in e.buildings:
                    if d <= 0.0:
                        break
                    bd = b.distance_to_point(wx, wy)
                    if bd < d:
                        d = bd
                for (ox, oy, r) in circles:
                    if d <= 0.0:
                        break
                    od = math.hypot(wx - ox, wy - oy) - r
                    if od < d:
                        d = od
                i = cy * self.nx + cx
                self.blocked[i] = d < self.HARD_CLEAR
                self.penalty[i] = 0.0 if d >= self.SOFT_CLEAR else \
                    max(0.0, min(1.0, (self.SOFT_CLEAR - d) / span)) * self.SOFT_WEIGHT

    def cell_of(self, x, y):
        cx = max(0, min(self.nx - 1, int(x / self.CELL)))
        cy = max(0, min(self.ny - 1, int(y / self.CELL)))
        return cy * self.nx + cx

    def center_of(self, idx):
        cx, cy = idx % self.nx, idx // self.nx
        return ((cx + 0.5) * self.CELL, (cy + 0.5) * self.CELL)

    def is_blocked_xy(self, x, y):
        if not (0 <= x < self.env.WIDTH and 0 <= y < self.env.HEIGHT):
            return True
        return self.blocked[self.cell_of(x, y)]

    def nearest_free_cell(self, x, y, max_r=None):
        max_r = self.GOAL_SEARCH_R if max_r is None else max_r
        best, best_d = None, float('inf')
        rad = int(math.ceil(max_r / self.CELL))
        cx0, cy0 = int(x / self.CELL), int(y / self.CELL)
        for cy in range(max(0, cy0 - rad), min(self.ny, cy0 + rad + 1)):
            for cx in range(max(0, cx0 - rad), min(self.nx, cx0 + rad + 1)):
                i = cy * self.nx + cx
                if self.blocked[i]:
                    continue
                wx, wy = (cx + 0.5) * self.CELL, (cy + 0.5) * self.CELL
                d = math.hypot(wx - x, wy - y)
                if d < best_d and d <= max_r:
                    best_d, best = d, i
        return best

    def line_free(self, x1, y1, x2, y2):
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < 1e-6:
            return True
        steps = max(2, int(dist / (self.CELL * 0.5)))
        ux, uy = (x2 - x1) / dist, (y2 - y1) / dist
        for k in range(1, steps + 1):
            s = dist * k / steps
            if self.is_blocked_xy(x1 + ux * s, y1 + uy * s):
                return False
        return True



    def plan(self, sx, sy, gx, gy) -> Optional[List[Tuple[float, float]]]:
        self.rebuild()
        nx, ny = self.nx, self.ny
        si = self.cell_of(sx, sy)
        goal_adjusted = False
        gi = self.cell_of(gx, gy)
        if self.blocked[gi]:
            alt = self.nearest_free_cell(gx, gy)
            if alt is None:
                self.last_path_len = 0.0
                return None
            gi = alt
            goal_adjusted = True

        if si == gi:
            pts = [(sx, sy)]
            if not goal_adjusted:
                pts.append((gx, gy))
            self.last_path_len = math.hypot(pts[-1][0] - sx, pts[-1][1] - sy)
            return pts

        gcx, gcy = gi % nx, gi // nx
        SQRT2 = math.sqrt(2.0)
        neigh = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2))

        def h(idx):
            dx = abs(idx % nx - gcx)
            dy = abs(idx // nx - gcy)
            return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)

        g = {si: 0.0}
        came = {}
        closed = set()
        counter = 0
        heap = [(h(si), 0, si)]
        expansions = 0
        found = False

        while heap and expansions < self.MAX_EXPANSIONS:
            _, _, cur = heapq.heappop(heap)
            if cur in closed:
                continue
            closed.add(cur)
            expansions += 1
            if cur == gi:
                found = True
                break
            cx, cy = cur % nx, cur // nx
            gcur = g[cur]
            for dx, dy, step in neigh:
                ax, ay = cx + dx, cy + dy
                if not (0 <= ax < nx and 0 <= ay < ny):
                    continue
                ni = ay * nx + ax
                if ni in closed:
                    continue
                if self.blocked[ni] and ni != si:
                    continue
                if dx != 0 and dy != 0:
                    if self.blocked[cy * nx + ax] or self.blocked[ay * nx + cx]:
                        continue
                ng = gcur + step * (1.0 + self.penalty[ni])
                if ng < g.get(ni, float('inf')):
                    g[ni] = ng
                    came[ni] = cur
                    counter += 1
                    heapq.heappush(heap, (ng + h(ni), counter, ni))

        self.last_expansions = expansions
        if not found:
            self.last_path_len = 0.0
            return None

        raw = []
        cur = gi
        while cur != si:
            raw.append(self.center_of(cur))
            cur = came[cur]
        raw.reverse()
        pts = [(sx, sy)] + raw
        if not goal_adjusted:
            pts[-1] = (gx, gy)

        path = self._smooth(pts)
        self.last_path_len = sum(math.hypot(path[i + 1][0] - path[i][0],
                                            path[i + 1][1] - path[i][1])
                                 for i in range(len(path) - 1))
        return path

    def _smooth(self, pts):
        if len(pts) <= 2:
            return pts
        out = [pts[0]]
        i = 0
        n = len(pts)
        while i < n - 1:
            best = i + 1
            j = i + 1
            while j < n:
                if self.line_free(pts[i][0], pts[i][1], pts[j][0], pts[j][1]):
                    best = j
                    j += 1
                else:
                    break
            out.append(pts[best])
            i = best
        return out

class DynamicDangerDetector:
    def __init__(self, freeze_distance: float = 30.0,
                 freeze_angle_cone_deg: float = 60.0,
                 still_as_static_ticks: int = 60):
        self.freeze_distance = freeze_distance
        self.freeze_angle_cone = freeze_angle_cone_deg
        self.still_as_static_ticks = still_as_static_ticks

    def check(self, robot_x, robot_y, robot_direction_deg,
              people, cyclists, freeze_post_cooldown_active=False) -> bool:
        if freeze_post_cooldown_active:
            return False
        for obj in list(people) + list(cyclists):
            if getattr(obj, 'state', None) == 'resting':
                continue
            if getattr(obj, 'still_ticks', 0) >= self.still_as_static_ticks:
                continue
            dist = math.hypot(obj.x - robot_x, obj.y - robot_y)
            if dist > self.freeze_distance:
                continue
            a = math.degrees(math.atan2(obj.y - robot_y, obj.x - robot_x)) % 360
            diff = (a - robot_direction_deg + 180) % 360 - 180
            if abs(diff) >= self.freeze_angle_cone:
                continue
            px, py = getattr(obj, 'prev_x', obj.x), getattr(obj, 'prev_y', obj.y)
            vx, vy = obj.x - px, obj.y - py
            speed = math.hypot(vx, vy)
            if speed < 0.05:
                return True
            if vx * (robot_x - obj.x) + vy * (robot_y - obj.y) < 0:
                continue
            return True
        return False