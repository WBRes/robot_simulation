import pygame
import random
import math
from datetime import datetime

class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.waypoints = [(100, 100), (700, 100), (700, 500), (100, 500), (400, 300), (100, 500)]
        self.current_wp = 0
        self.speed = 2
        self.direction = 0

        self.lidar_range = 150
        self.num_rays = 24
        self.lidar_distances = [self.lidar_range] * self.num_rays
        self.lidar_types = [None] * self.num_rays
        self.num_cameras = 4
        self.camera_range = 75
        self.camera_angle = 90

        self.seen_person_colors = set()
        self.person_count = 0
        self.seen_cyclist_colors = set()
        self.cyclist_count = 0
        self.litter_count = 0
        self.car_count = 0
        self.fence_break_count = 0
        self.tree_count = 0
        self.litter_collected = 0
        self.place_count = 0

        self.following_target = False
        self.target = None
        self.target_type = None
        self.target_mix = 0.3

        self.avoiding = False
        self.avoid_steps = 0
        self.avoid_angle = 0

        self.manual_mode = False

        self.waypoint_timer = 0
        self.waypoint_timeout = 1200

    def clamp_position(self):
        self.x = max(10, min(790, self.x))
        self.y = max(10, min(590, self.y))

    def apply_velocity(self, linear_vel, angular_vel_deg):
        self.direction = (self.direction + angular_vel_deg) % 360
        rad = math.radians(self.direction)
        self.x += linear_vel * math.cos(rad)
        self.y += linear_vel * math.sin(rad)
        self.clamp_position()

    def person_in_lidar(self, px, py, radius=6):
        cx, cy = px, py
        radius_sq = radius * radius
        for i in range(self.num_rays):
            angle = math.radians(i * 360 / self.num_rays)
            dx = math.cos(angle)
            dy = math.sin(angle)
            acx = cx - self.x
            acy = cy - self.y
            t = acx * dx + acy * dy
            if t < 0:
                closest_x, closest_y = self.x, self.y
            elif t > self.lidar_range:
                closest_x = self.x + self.lidar_range * dx
                closest_y = self.y + self.lidar_range * dy
            else:
                closest_x = self.x + t * dx
                closest_y = self.y + t * dy
            dist_sq = (closest_x - cx) ** 2 + (closest_y - cy) ** 2
            if dist_sq <= radius_sq:
                return True
        return False

    def cast_ray(self, angle_rad, obstacles):
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        min_dist = self.lidar_range
        hit_type = None
        for (ox, oy, R, otype) in obstacles:
            ocx = self.x - ox
            ocy = self.y - oy
            b = ocx * dx + ocy * dy
            c = ocx * ocx + ocy * ocy - R * R
            disc = b * b - c
            if disc < 0:
                continue
            sq = math.sqrt(disc)
            t1 = -b - sq
            t2 = -b + sq
            if t1 >= 0:
                t = t1
            elif t2 >= 0:
                t = 0.0
            else:
                continue
            if t < min_dist:
                min_dist = t
                hit_type = otype
        return min_dist, hit_type

    def update_lidar(self, obstacles, world_width=800, world_height=600, fence_margin=8.0, clamp_fence=True):
        base = math.radians(self.direction)
        step = math.radians(360 / self.num_rays)
        for i in range(self.num_rays):
            angle = base + i * step
            d, t = self.cast_ray(angle, obstacles)
            if clamp_fence:
                dx, dy = math.cos(angle), math.sin(angle)
                tf = float('inf')
                if dx > 1e-6:
                    tf = min(tf, (world_width - fence_margin - self.x) / dx)
                elif dx < -1e-6:
                    tf = min(tf, (fence_margin - self.x) / dx)
                if dy > 1e-6:
                    tf = min(tf, (world_height - fence_margin - self.y) / dy)
                elif dy < -1e-6:
                    tf = min(tf, (fence_margin - self.y) / dy)
                if tf < d:
                    d = tf
                    t = 'fence'
            self.lidar_distances[i] = d
            self.lidar_types[i] = t

    def choose_target(self, people, cyclists):
        if self.manual_mode:
            return
        wp_x, wp_y = self.waypoints[self.current_wp]
        dx = wp_x - self.x
        dy = wp_y - self.y
        wp_dist = math.hypot(dx, dy)
        if wp_dist < 80:
            return
        closest = None
        min_dist = float('inf')
        for p in people:
            dist = math.hypot(self.x - p.x, self.y - p.y)
            if dist <= self.lidar_range and p.color not in self.seen_person_colors:
                if dist < min_dist:
                    min_dist = dist
                    closest = p
                    self.target_type = 'person'
        for c in cyclists:
            dist = math.hypot(self.x - c.x, self.y - c.y)
            if dist <= self.lidar_range and c.color not in self.seen_cyclist_colors:
                if dist < min_dist:
                    min_dist = dist
                    closest = c
                    self.target_type = 'cyclist'
        if closest:
            self.target = closest
            self.following_target = True
        else:
            self.target = None
            self.following_target = False

    def obstacle_ahead(self, objects, distances=[40, 45, 50], angles=[-45, -20, 0, 20, 45]):
        if self.manual_mode:
            return None
        detected = {'left': False, 'front': False, 'right': False}
        for dist in distances:
            for ang in angles:
                rad = math.radians(self.direction + ang)
                fx = self.x + dist * math.cos(rad)
                fy = self.y + dist * math.sin(rad)
                for obj in objects:
                    if isinstance(obj, (Person, Litter, Car, PublicPlace, Cyclist)):
                        if math.hypot(fx - obj.x, fy - obj.y) < 15:
                            if ang < 0:
                                detected['left'] = True
                            elif ang > 0:
                                detected['right'] = True
                            else:
                                detected['front'] = True
        if detected['front']:
            return 'front'
        if detected['left'] and detected['right']:
            return 'both'
        if detected['left']:
            return 'left'
        if detected['right']:
            return 'right'
        return None

    def avoid_collision(self, objects, min_dist=18):
        for obj in objects:
            dist = math.hypot(self.x - obj.x, self.y - obj.y)
            if dist < min_dist:
                angle = math.atan2(self.y - obj.y, self.x - obj.x)
                self.x += self.speed * math.cos(angle)
                self.y += self.speed * math.sin(angle)
                self.direction = math.degrees(angle)
                self.clamp_position()
                return True
        return False

    def handle_manual_input(self):
        keys = pygame.key.get_pressed()
        dx = 0
        dy = 0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx = -1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx = 1
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy = -1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy = 1
        if dx != 0 or dy != 0:
            length = math.hypot(dx, dy)
            dx /= length
            dy /= length
            self.x += dx * self.speed
            self.y += dy * self.speed
            self.direction = math.degrees(math.atan2(dy, dx))
        self.clamp_position()

    def move(self, people, cars, places, cyclists, trees, litters):
        all_obstacles = people + cars + places + cyclists + trees

        # На случай если робот застрял, то через 20 сек меняет маршрут
        if not self.following_target:
            self.waypoint_timer += 1

        if self.waypoint_timer >= self.waypoint_timeout:
            self.current_wp = (self.current_wp + 1) % len(self.waypoints)
            self.waypoint_timer = 0
            return

        if not self.manual_mode:
            self.avoid_collision(all_obstacles, min_dist=18)

        if self.manual_mode:
            self.avoid_collision(all_obstacles, min_dist=18)
            self.handle_manual_input()
            return

        # 1. Преследование человека или велосипедиста
        if self.following_target and self.target:
            dx = self.target.x - self.x
            dy = self.target.y - self.y
            dist_to_target = math.hypot(dx, dy)
            if self.target_type == 'person':
                already_seen = self.target.color in self.seen_person_colors
            else:
                already_seen = self.target.color in self.seen_cyclist_colors
            camera_sees = self.object_in_cam(self.target.x, self.target.y)
            if camera_sees or already_seen:
                if self.target_type == 'person':
                    self.seen_person_colors.add(self.target.color)
                    self.person_count += 1
                    log_event(f"Обнаружен человек {self.target.color}, всего обнаружено: {self.person_count}")
                    #print(f"Обнаружен человек {self.target.color}, всего обнаружено: {self.person_count}")
                else:
                    self.seen_cyclist_colors.add(self.target.color)
                    self.cyclist_count += 1
                    log_event(f"Обнаружен человек {self.target.color}, всего обнаружено: {self.person_count}")
                    #print(f"Обнаружен человек {self.target.color}, всего обнаружено: {self.person_count}")
                self.following_target = False
                self.target = None
                self.target_type = None
                return
            else:
                if dist_to_target < self.camera_range - 10:
                    if dist_to_target > 0:
                        self.x -= (dx / dist_to_target) * self.speed * 0.5
                        self.y -= (dy / dist_to_target) * self.speed * 0.5
                else:
                    wp_x, wp_y = self.waypoints[self.current_wp]
                    wp_dx = wp_x - self.x
                    wp_dy = wp_y - self.y
                    wp_dist = math.hypot(wp_dx, wp_dy)
                    move_x = ((1 - self.target_mix) * wp_dx + self.target_mix * dx)
                    move_y = ((1 - self.target_mix) * wp_dy + self.target_mix * dy)
                    move_len = math.hypot(move_x, move_y)
                    if wp_dist < 8:
                        wp_dx = 0
                        wp_dy = 0
                    else:
                        wp_dx /= wp_dist
                        wp_dy /= wp_dist
                    if dist_to_target > 0:
                        target_dx = dx / dist_to_target
                    if move_len > 0.05:
                        move_x /= move_len
                        move_y /= move_len
                        self.x += move_x * self.speed
                        self.y += move_y * self.speed

                        self.direction = math.degrees(math.atan2(move_y, move_x))
                        self.following_target = False
                        self.target = None
                        return
                self.avoid_collision(all_obstacles, min_dist=10)
                self.clamp_position()
                return

        # 2. Объезд препятствий
        if not self.avoiding:
            obstacle = self.obstacle_ahead(all_obstacles)
            if obstacle == 'front':
                left_free = True
                right_free = True
                for dist in [30, 40, 50]:
                    rad_left = math.radians(self.direction - 45)
                    fx_left = self.x + dist * math.cos(rad_left)
                    fy_left = self.y + dist * math.sin(rad_left)
                    rad_right = math.radians(self.direction + 45)
                    fx_right = self.x + dist * math.cos(rad_right)
                    fy_right = self.y + dist * math.sin(rad_right)
                    for obj in all_obstacles:
                        if isinstance(obj, (Person, Litter, Car, PublicPlace, Cyclist)):
                            if math.hypot(fx_left - obj.x, fy_left - obj.y) < 5:
                                left_free = False
                            if math.hypot(fx_right - obj.x, fy_right - obj.y) < 5:
                                right_free = False
                if left_free and not right_free:
                    turn = -45
                elif right_free and not left_free:
                    turn = 45
                elif left_free and right_free:
                    turn = 45
                else:
                    turn = 180
                self.avoiding = True
                self.avoid_steps = 25
                self.avoid_angle = self.direction + turn
            elif obstacle == 'left':
                self.avoiding = True
                self.avoid_steps = 25
                self.avoid_angle = self.direction + 30
            elif obstacle == 'right':
                self.avoiding = True
                self.avoid_steps = 25
                self.avoid_angle = self.direction - 30
        if self.avoiding:
            rad = math.radians(self.avoid_angle)
            self.x += self.speed * math.cos(rad)
            self.y += self.speed * math.sin(rad)
            self.avoid_steps -= 1
            if self.avoid_steps <= 0:
                self.avoiding = False
            else:
                if self.obstacle_ahead(all_obstacles, distances=[30], angles=[0]) == 'front':
                    self.avoid_angle += 20
                    self.avoid_steps = max(self.avoid_steps, 20)
            self.direction = self.avoid_angle
            self.avoid_collision(all_obstacles, min_dist=15)
            self.clamp_position()
            return

        # 3. Движение по waypoints
        wp_x, wp_y = self.waypoints[self.current_wp]
        dx = wp_x - self.x
        dy = wp_y - self.y
        dist = math.hypot(dx, dy)

        for obj in (cars + trees + litters + places):
            if math.hypot(self.x - obj.x, self.y - obj.y) < self.camera_range:
                if math.hypot(obj.x - wp_x, obj.y - wp_y) < 45:
                    self.current_wp = (self.current_wp + 1) % len(self.waypoints)
                    self.waypoint_timer = 0
                    return

        if dist < 8:
            self.current_wp = (self.current_wp + 1) % len(self.waypoints)
            self.waypoint_timer = 0
            return

        self.x += dx / dist * self.speed
        self.y += dy / dist * self.speed
        self.direction = math.degrees(math.atan2(dy, dx))
        self.clamp_position()


    def object_in_cam(self, ox, oy):
        dx = ox - self.x
        dy = oy - self.y
        dist = math.hypot(dx, dy)
        if dist > self.camera_range:
            return False
        angle_to_point = math.degrees(math.atan2(dy, dx)) % 360
        for i in range(self.num_cameras):
            cam_dir = (self.direction + i * 360 / self.num_cameras) % 360
            angle_diff = abs(angle_to_point - cam_dir)
            angle_diff = min(angle_diff, 360 - angle_diff)
            if angle_diff <= self.camera_angle / 2:
                return True
        return False

    def draw(self, screen):
        body = pygame.Surface((34, 18), pygame.SRCALPHA)
        pygame.draw.rect(body, (255, 220, 0), (0, 0, 26, 18), 2)
        pygame.draw.rect(body, (255, 255, 255), (9, 5, 8, 8), 0)
        pygame.draw.polygon(body, (255, 120, 0), [(26, 4), (26, 14), (33, 9)])
        rotated = pygame.transform.rotate(body, -self.direction)
        screen.blit(rotated, rotated.get_rect(center=(int(self.x), int(self.y))))

        type_colors = {
            'person':  (255, 60, 60),
            'cyclist': (255, 140, 0),
            'litter':  (120, 255, 120),
            'car':     (160, 160, 255),
            'place':   (255, 255, 120),
            'tree':    (60, 200, 60),
            'fence':   (155, 94, 64),
            None:       (80, 255, 80),
        }
        base = math.radians(self.direction)
        for i in range(self.num_rays):
            angle = base + math.radians(i * 360 / self.num_rays)
            d = self.lidar_distances[i]
            t = self.lidar_types[i] if i < len(self.lidar_types) else None
            color = type_colors.get(t, (80, 255, 80))
            end_x = self.x + d * math.cos(angle)
            end_y = self.y + d * math.sin(angle)
            pygame.draw.line(screen, color, (self.x, self.y), (end_x, end_y), 1)
            if d < self.lidar_range:
                pygame.draw.circle(screen, color, (int(end_x), int(end_y)), 2)

        for i in range(self.num_cameras):
            cam_dir = self.direction + i * 360 / self.num_cameras
            left_angle = math.radians(cam_dir - self.camera_angle / 2)
            right_angle = math.radians(cam_dir + self.camera_angle / 2)
            far_left = (self.x + self.camera_range * math.cos(left_angle),
                        self.y + self.camera_range * math.sin(left_angle))
            far_right = (self.x + self.camera_range * math.cos(right_angle),
                         self.y + self.camera_range * math.sin(right_angle))
            pygame.draw.line(screen, (80, 80, 255), (self.x, self.y), far_left, 1)
            pygame.draw.line(screen, (80, 80, 255), (self.x, self.y), far_right, 1)
            arc_points = []
            for s in range(16):
                ang = left_angle + (right_angle - left_angle) * s / 15
                arc_points.append((self.x + self.camera_range * math.cos(ang),
                                   self.y + self.camera_range * math.sin(ang)))
            pygame.draw.lines(screen, (80, 80, 255), False, arc_points, 1)


class Person:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(0.8, 1.5)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color

    def move(self):
        self.x += self.vx
        self.y += self.vy
        body = 6
        if self.x < body:
            self.x = body
            self.vx = -self.vx
        if self.x > 800 - body:
            self.x = 800 - body
            self.vx = -self.vx
        if self.y < body:
            self.y = body
            self.vy = -self.vy
        if self.y > 600 - body:
            self.y = 600 - body
            self.vy = -self.vy

    def draw(self, screen, robot, show_all_people):
        if show_all_people:
            color = self.color
        else:
            """
            если постоянное моргание звёзд раздражает, то можно использовать эту конструкцию:
            dist = math.hypot(robot.x - self.x, robot.y - self.y)
            if dist > robot.lidar_range:
                return
            также можно увеличить self.num_rays в __init__ в классе Robot
            """
            if not robot.person_in_lidar(self.x, self.y):
                return
            camera_sees = robot.object_in_cam(self.x, self.y)
            if camera_sees:
                color = self.color
            else:
                color = (80, 80, 80)
        outer_radius = 6
        inner_radius = 3
        points = []
        for i in range(10):
            angle = math.radians(i * 36 - 90)
            if i % 2 == 0:
                r = outer_radius
            else:
                r = inner_radius
            px = self.x + r * math.cos(angle)
            py = self.y + r * math.sin(angle)
            points.append((px, py))
        pygame.draw.polygon(screen, color, points, 0)
        pygame.draw.polygon(screen, (0, 0, 0), points, 1)


class Cyclist:
    def __init__(self, x, y, direction, color):
        self.x = x
        self.y = y
        self.direction = direction
        self.speed = 2.5
        self.color = color

    def move(self, bike_path_y, min_x=50, max_x=750):
        self.x += self.direction * self.speed
        if self.x < min_x:
            self.x = min_x
            self.direction = 1
        elif self.x > max_x:
            self.x = max_x
            self.direction = -1

    def draw(self, screen, robot, show_all_cyclists):
        if show_all_cyclists:
            color = self.color
        else:
            if not robot.person_in_lidar(self.x, self.y, radius=6):
                return
            camera_sees = robot.object_in_cam(self.x, self.y)
            if camera_sees:
                color = self.color
            else:
                color = (80, 80, 80)
        if self.direction == 1:
            angle = 0
        else:
            angle = math.pi
        points = []
        size = 6
        for i in range(3):
            a = angle + math.radians(i * 120 - 90)
            px = self.x + size * math.cos(a)
            py = self.y + size * math.sin(a)
            points.append((px, py))
        pygame.draw.polygon(screen, color, points, 0)
        pygame.draw.polygon(screen, (0, 0, 0), points, 1)


class Litter:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.counted = False

    def draw(self, screen, show_all_litters):
        if show_all_litters or self.counted:
            w = 12
            h = 7
            left = self.x - w // 2
            top = self.y - h // 2
            pygame.draw.rect(screen, (120, 120, 120), (left, top, w, h), border_radius=4)
            pygame.draw.rect(screen, (60, 60, 60), (left, top, w, h), 1, border_radius=4)
            rope_x = left + w - 2
            rope_y = self.y
            pygame.draw.circle(screen, (150, 150, 150), (rope_x, rope_y), 3, 0)
            pygame.draw.circle(screen, (80, 80, 80), (rope_x, rope_y), 3, 1)
            pygame.draw.line(screen, (150, 150, 150), (rope_x, rope_y), (rope_x + 3, rope_y - 2), 2)
            pygame.draw.line(screen, (150, 150, 150), (rope_x, rope_y), (rope_x + 3, rope_y + 2), 2)


class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.counted = False

    def draw(self, screen, show_all_cars):
        if not (show_all_cars or self.counted):
            return
        w = 26
        h = 12
        left = int(self.x - w // 2)
        top = int(self.y - h // 2)
        pygame.draw.rect(screen, (180, 180, 200), (left, top, w, h), border_radius=3)
        pygame.draw.rect(screen, (80, 80, 100), (left, top, w, h), 1, border_radius=3)
        for x_off in (left + 5, left + w - 5):
            pygame.draw.circle(screen, (40, 40, 50), (x_off, top + h - 2), 3)
            pygame.draw.circle(screen, (40, 40, 50), (x_off, top + 1), 3)
        pygame.draw.circle(screen, (255, 200, 100), (left + 2, top + h // 2 - 1), 1)
        pygame.draw.circle(screen, (255, 200, 100), (left + 2, top + h // 2 + 1), 1)


class PublicPlace:
    def __init__(self, x, y, place_type):
        self.x = x
        self.y = y
        self.type = place_type
        self.radius = 25
        self.visitor_count = 0
        self.color_seen = set()
        self.counted = False

    def draw(self, screen, show_all_public_places):
        if not (show_all_public_places or self.counted):
            return

        if self.type == 'скамейка':
            w = 16
            h = 10
            left = self.x - w // 2
            top = self.y - h // 2
            pygame.draw.rect(screen, (139, 69, 19), (left, top, w, h), 2)
            y1 = top + h // 3
            y2 = top + 2 * h // 3
            pygame.draw.line(screen, (139, 69, 19), (left + 2, y1), (left + w - 2, y1), 2)
            pygame.draw.line(screen, (139, 69, 19), (left + 2, y2), (left + w - 2, y2), 2)

        elif self.type == 'турник':
            pygame.draw.line(screen, (100, 100, 100), (self.x - 6, self.y - 8), (self.x - 6, self.y + 4), 2)
            pygame.draw.line(screen, (100, 100, 100), (self.x + 6, self.y - 8), (self.x + 6, self.y + 4), 2)
            pygame.draw.line(screen, (100, 100, 100), (self.x - 6, self.y - 5), (self.x + 6, self.y - 5), 2)

        else:
            color = (160, 82, 45)
            size = 10
            pygame.draw.circle(screen, color, (self.x, self.y), size, 2)


class FenceBreak:
    def __init__(self, x, y, side, width=40):
        self.x = x
        self.y = y
        self.side = side
        self.width = width
        self.detected = False


def draw_fence(screen, fence_breaks, show_all_fence):
    fence_color = (155, 94, 64)
    thickness = 6

    if show_all_fence:
        def need_break(br):
            return True
    else:
        def need_break(br):
            return br.detected
    x = 0
    while x < 800:
        broken = False
        for br in fence_breaks:
            if br.side == 'top' and need_break(br):
                if br.x - br.width // 2 <= x <= br.x + br.width // 2:
                    broken = True
                    x = br.x + br.width // 2
                    break
        if not broken:
            end_x = min(x + 10, 800)
            pygame.draw.line(screen, fence_color, (x, 0), (end_x, 0), thickness)
            x = end_x
        else:
            x += 1
    x = 0
    while x < 800:
        broken = False
        for br in fence_breaks:
            if br.side == 'bottom' and need_break(br):
                if br.x - br.width // 2 <= x <= br.x + br.width // 2:
                    broken = True
                    x = br.x + br.width // 2
                    break
        if not broken:
            end_x = min(x + 10, 800)
            pygame.draw.line(screen, fence_color, (x, 600 - thickness / 2), (end_x, 600 - thickness / 2), thickness)
            x = end_x
        else:
            x += 1
    y = 0
    while y < 600:
        broken = False
        for br in fence_breaks:
            if br.side == 'left' and need_break(br):
                if br.x - br.width // 2 <= y <= br.x + br.width // 2:
                    broken = True
                    y = br.x + br.width // 2
                    break
        if not broken:
            end_y = min(y + 10, 600)
            pygame.draw.line(screen, fence_color, (0, y), (0, end_y), thickness)
            y = end_y
        else:
            y += 1
    y = 0
    while y < 600:
        broken = False
        for br in fence_breaks:
            if br.side == 'right' and need_break(br):
                if br.x - br.width // 2 <= y <= br.x + br.width // 2:
                    broken = True
                    y = br.x + br.width // 2
                    break
        if not broken:
            end_y = min(y + 10, 600)
            pygame.draw.line(screen, fence_color, (800 - thickness / 2, y), (800 - thickness / 2, end_y), thickness)
            y = end_y
        else:
            y += 1


class Tree:
    def __init__(self, x, y, tree_type):
        self.x = x
        self.y = y
        self.counted = False
        self.type = tree_type

    def draw(self, screen, show_all_trees):
        trunk_w = 6
        trunk_h = 20
        x = self.x - trunk_w / 2
        y = self.y - trunk_h
        if show_all_trees or self.counted:
            pygame.draw.rect(screen, (251, 225, 185), (x, y, trunk_w, trunk_h))
            if self.type == 'Ёлка':
                tiers = [
                    {'width': 20, 'height': 20, 'y': 0},
                    {'width': 15, 'height': 18, 'y': -15},
                    {'width': 10, 'height': 16, 'y': -30}
                ]
                for tier in tiers:
                    left = (self.x - tier['width'], y + tier['y'])
                    right = (self.x + tier['width'], y + tier['y'])
                    top = (self.x, y + tier['y'] - tier['height'])
                    points = [left, right, top]
                    pygame.draw.polygon(screen, (34, 139, 34), points, 0)
            else:
                pygame.draw.circle(screen, (92, 169, 4), (x + trunk_w / 2, y - trunk_h), 30)


def generate_good_position(pos, x_range, y_range, min_dist=20, max_attempts=100):
    for i in range(max_attempts):
        x = random.randint(x_range[0], x_range[1])
        y = random.randint(y_range[0], y_range[1])
        conflict = True
        for (px, py) in pos:
            if math.hypot(x - px, y - py) < min_dist:
                conflict = False
                break
        if conflict:
            return [x, y]
    print('bad')
    return [random.randint(x_range[0], x_range[1]), random.randint(y_range[0], y_range[1])]

def build_lidar_obstacles(people, cyclists, litters, cars, places, trees):
    obstacles = []
    for p in people:
        obstacles.append((p.x, p.y, 6, 'person'))
    for c in cyclists:
        obstacles.append((c.x, c.y, 6, 'cyclist'))
    for l in litters:
        obstacles.append((l.x, l.y, 5, 'litter'))
    for car in cars:
        obstacles.append((car.x, car.y, 14, 'car'))
    for pl in places:
        obstacles.append((pl.x, pl.y, 10, 'place'))
    for t in trees:
        obstacles.append((t.x, t.y, 10, 'tree'))
    return obstacles


def log_event(message):
    with open("simulation.log", "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")