import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import math
from datetime import datetime
import torch

class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.waypoints = [(100, 100), (700, 100), (700, 500), (100, 500), (400, 300), (100, 500)]
        self.current_wp = 0
        self.speed = 2
        self.direction = 0

        self.lidar_range = 150
        self.num_rays = 24
        self.camera_range = 75
        self.camera_angle = 90
        self.camera_direction = 0

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

    def move(self, people, cars, places, cyclists, trees):
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

        for obj in (cars + trees + places):
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

    def update_camera_direction(self, people, cyclists):
        closest_obj = None
        min_dist = float('inf')
        for p in people:
            dist = math.hypot(self.x - p.x, self.y - p.y)
            if dist <= self.lidar_range and dist < min_dist:
                min_dist = dist
                closest_obj = p
        for c in cyclists:
            dist = math.hypot(self.x - c.x, self.y - c.y)
            if dist <= self.lidar_range and dist < min_dist:
                min_dist = dist
                closest_obj = c
        if closest_obj:
            dx = closest_obj.x - self.x
            dy = closest_obj.y - self.y
            self.camera_direction = math.degrees(math.atan2(dy, dx))
        else:
            self.camera_direction = self.direction

    def object_in_cam(self, ox, oy):
        dx = ox - self.x
        dy = oy - self.y
        dist = math.hypot(dx, dy)
        if dist > self.camera_range:
            return False
        angle_to_point = math.degrees(math.atan2(dy, dx))
        cam_dir = self.camera_direction % 360
        angle_diff = abs(angle_to_point - cam_dir)
        angle_diff = min(angle_diff, 360 - angle_diff)
        return angle_diff <= self.camera_angle / 2

    def draw(self, screen):
        rect_w = 20
        rect_h = 20
        rect_left = int(self.x) - rect_w // 2
        rect_top = int(self.y) - rect_h // 2
        pygame.draw.rect(screen, 'yellow', (rect_left, rect_top, rect_w, rect_h), 2)

        for i in range(self.num_rays):
            angle = math.radians(i * 360 / self.num_rays)
            end_x = self.x + self.lidar_range * math.cos(angle)
            end_y = self.y + self.lidar_range * math.sin(angle)
            pygame.draw.line(screen, 'green', (self.x, self.y), (end_x, end_y), 1)

        left_angle = math.radians(self.camera_direction - self.camera_angle / 2)
        right_angle = math.radians(self.camera_direction + self.camera_angle / 2)
        far_left = (self.x + self.camera_range * math.cos(left_angle),
                    self.y + self.camera_range * math.sin(left_angle))
        far_right = (self.x + self.camera_range * math.cos(right_angle),
                     self.y + self.camera_range * math.sin(right_angle))
        pygame.draw.line(screen, 'blue', (self.x, self.y), far_left, 2)
        pygame.draw.line(screen, 'blue', (self.x, self.y), far_right, 2)

        arc_points = []
        steps = 30
        for i in range(steps + 1):
            t = i / steps
            angle = left_angle + t * (right_angle - left_angle)
            x = self.x + self.camera_range * math.cos(angle)
            y = self.y + self.camera_range * math.sin(angle)
            arc_points.append((x, y))
        if len(arc_points) > 1:
            pygame.draw.lines(screen, 'blue', False, arc_points, 2)


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

def log_event(message):
    with open("simulation.log", "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")

class RobotEnvWaypoints(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render_mode=None, max_steps=500, num_waypoints=6):
        super(RobotEnvWaypoints, self).__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.width = 800
        self.height = 600
        self.num_waypoints = num_waypoints

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(16,),
            dtype=np.float32
        )

        self.screen = None
        self.clock = None
        self.step_count = 0
        self.max_sensor_range = 200

        self.robot = None
        self.people = []
        self.cyclists = []
        self.litters = []
        self.cars = []
        self.places = []
        self.trees = []
        self.fence_breaks = []
        self.waypoints = []
        self.current_wp_idx = 0
        self.target_litter = None
        self.prev_dist_to_wp = 1000.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.robot = Robot(self.width // 2, self.height // 2)
        self.robot.manual_mode = True

        self.waypoints = []
        for _ in range(self.num_waypoints):
            x = random.randint(60, self.width - 60)
            y = random.randint(60, self.height - 60)
            self.waypoints.append((x, y))
        self.current_wp_idx = 0
        self.prev_dist_to_wp = 1000.0

        num_litters = random.randint(2, 4)
        litter_positions = []
        for _ in range(num_litters):
            pos = generate_good_position(litter_positions, [50, 750], [50, 550], min_dist=30)
            litter_positions.append(pos)
        self.litters = [Litter(pos[0], pos[1]) for pos in litter_positions]

        num_people = random.randint(0, 2)
        self.people = []
        colors = ['red', 'blue', 'green', 'yellow', 'purple']
        for _ in range(num_people):
            x = random.randint(50, 750)
            y = random.randint(50, 550)
            color = random.choice(colors)
            self.people.append(Person(x, y, color))

        num_cars = random.randint(0, 1)
        self.cars = []
        car_positions = []
        for _ in range(num_cars):
            pos = generate_good_position(car_positions, [710, 750], [410, 550], min_dist=30)
            car_positions.append(pos)
            self.cars.append(Car(pos[0], pos[1]))

        num_trees = random.randint(0, 2)
        self.trees = []
        tree_positions = []
        for _ in range(num_trees):
            pos = generate_good_position(tree_positions, [50, 680], [50, 550], min_dist=50)
            tree_positions.append(pos)
            self.trees.append(Tree(pos[0], pos[1], random.choice(['Ёлка', 'Дуб'])))

        self._update_target()

        self.step_count = 0
        obs = self._get_observation()
        info = {}

        if self.render_mode == 'human':
            self._render_frame()

        return obs, info

    def step(self, action):
        self.step_count += 1
        done = False

        linear_vel = float(action[0]) * 2.0
        angular_vel = float(action[1]) * 3.0

        angle = math.radians(self.robot.direction)
        dx = linear_vel * math.cos(angle)
        dy = linear_vel * math.sin(angle)
        self.robot.x += dx
        self.robot.y += dy
        self.robot.direction += angular_vel * 10
        self.robot.direction %= 360

        self.robot.x = max(10, min(self.width - 10, self.robot.x))
        self.robot.y = max(10, min(self.height - 10, self.robot.y))

        for p in self.people:
            p.move()

        collision = False
        all_obstacles = self.people + self.cars + self.trees
        for obj in all_obstacles:
            if math.hypot(self.robot.x - obj.x, self.robot.y - obj.y) < 15:
                collision = True
                break

        litter_collected = False
        for litter in self.litters[:]:
            if math.hypot(self.robot.x - litter.x, self.robot.y - litter.y) < 15:
                self.litters.remove(litter)
                litter_collected = True
        self._update_target()

        reward = -0.05

        # 1. Движение к waypoint
        if self.current_wp_idx < len(self.waypoints):
            wp_x, wp_y = self.waypoints[self.current_wp_idx]
            dist_to_wp = math.hypot(self.robot.x - wp_x, self.robot.y - wp_y)

            if dist_to_wp < self.prev_dist_to_wp:
                reward += 2.0 * (self.prev_dist_to_wp - dist_to_wp) / 10.0
            self.prev_dist_to_wp = dist_to_wp

            if dist_to_wp < 20:
                reward += 200.0
                self.current_wp_idx += 1
                if self.current_wp_idx >= len(self.waypoints):
                    done = True
                    reward += 500.0
                else:
                    self.prev_dist_to_wp = 1000.0
        else:
            done = True

        if litter_collected:
            reward += 500.0
            if len(self.litters) == 0:
                reward += 500.0

        if abs(linear_vel) < 0.1 and abs(angular_vel) < 0.1:
            reward -= 1.0
        if collision:
            reward -= 50.0
            done = True

        if (self.robot.x <= 15 or self.robot.x >= self.width - 15 or
            self.robot.y <= 15 or self.robot.y >= self.height - 15):
            reward -= 1.0
        truncated = self.step_count >= self.max_steps
        if truncated:
            reward -= 10.0
        obs = self._get_observation()
        info = {'litters_left': len(self.litters), 'wp_idx': self.current_wp_idx}
        if self.render_mode == 'human':
            self._render_frame()
        return obs, reward, done, truncated, info

    def _update_target(self):
        if not self.litters:
            self.target_litter = None
            return
        min_dist = float('inf')
        closest = None
        for l in self.litters:
            d = math.hypot(self.robot.x - l.x, self.robot.y - l.y)
            if d < min_dist:
                min_dist = d
                closest = l
        self.target_litter = closest

    def _get_observation(self):
        obs = []
        obs.append(self.robot.x / self.width)
        obs.append(self.robot.y / self.height)
        if self.current_wp_idx < len(self.waypoints):
            wp_x, wp_y = self.waypoints[self.current_wp_idx]
            obs.append(wp_x / self.width)
            obs.append(wp_y / self.height)
        else:
            obs.append(0.5)
            obs.append(0.5)
        if self.current_wp_idx < len(self.waypoints):
            wp_x, wp_y = self.waypoints[self.current_wp_idx]
            dist = math.hypot(self.robot.x - wp_x, self.robot.y - wp_y)
            obs.append(dist / 800.0)
        else:
            obs.append(1.0)
        if self.target_litter:
            obs.append(self.target_litter.x / self.width)
            obs.append(self.target_litter.y / self.height)
        else:
            obs.append(0.5)
            obs.append(0.5)
        if self.target_litter:
            dist = math.hypot(self.robot.x - self.target_litter.x,
                              self.robot.y - self.target_litter.y)
            obs.append(dist / 800.0)
        else:
            obs.append(1.0)
        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        robot_angle = math.radians(self.robot.direction)
        all_obstacles = self.people + self.cars + self.trees
        min_dists = [self.max_sensor_range] * 4
        for obj in all_obstacles:
            dx_obj = obj.x - self.robot.x
            dy_obj = obj.y - self.robot.y
            dist = math.hypot(dx_obj, dy_obj)
            if dist > self.max_sensor_range:
                continue
            angle_to_obj = math.atan2(dy_obj, dx_obj)
            rel_angle = angle_to_obj - robot_angle
            rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
            if abs(rel_angle) < math.radians(45):
                idx = 0
            elif abs(rel_angle) > math.radians(135):
                idx = 1
            elif rel_angle > 0:
                idx = 2
            else:
                idx = 3
            if dist < min_dists[idx]:
                min_dists[idx] = dist
        for d in min_dists:
            obs.append(min(d / self.max_sensor_range, 1.0))
        obs.append(len(self.litters) / 5.0)
        obs.append(1.0 if self.target_litter else 0.0)
        obs.append(0.0)
        obs.append(0.0)
        return np.array(obs, dtype=np.float32)

    def render(self):
        if self.render_mode == 'human':
            self._render_frame()

    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
        self.screen.fill((48, 194, 107))
        for i, (wx, wy) in enumerate(self.waypoints):
            color = (0, 0, 255) if i == self.current_wp_idx else (100, 100, 255)
            pygame.draw.circle(self.screen, color, (int(wx), int(wy)), 8, 2)
            font = pygame.font.SysFont('Arial', 14)
            text = font.render(str(i+1), True, (255,255,255))
            self.screen.blit(text, (wx-5, wy-15))

        rect = pygame.Rect(self.robot.x-10, self.robot.y-10, 20, 20)
        pygame.draw.rect(self.screen, 'yellow', rect, 2)

        for l in self.litters:
            l.draw(self.screen, True)

        for p in self.people:
            p.draw(self.screen, self.robot, True)
        for c in self.cars:
            c.draw(self.screen, True)
        for t in self.trees:
            t.draw(self.screen, True)

        if self.target_litter:
            pygame.draw.circle(self.screen, (255,0,0),
                               (int(self.target_litter.x), int(self.target_litter.y)), 5)

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

if __name__ == "__main__":
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используемое устройство: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    num_envs = 8
    def make_env():
        return RobotEnvWaypoints(render_mode=None, max_steps=500, num_waypoints=6)

    env = SubprocVecEnv([make_env for _ in range(num_envs)])
    eval_env = RobotEnvWaypoints(render_mode=None, max_steps=500, num_waypoints=6)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=5e-4,
        buffer_size=2_000_000,
        learning_starts=10000,
        batch_size=1024,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        target_entropy=-0.5,
        target_update_interval=1,
        verbose=1,
        tensorboard_log="./sac_waypoints_tensorboard/",
        device=device,
        policy_kwargs=dict(net_arch=[512, 512])
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model_waypoints/",
        log_path="./logs_waypoints/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./checkpoints_waypoints/",
        name_prefix="sac_waypoints"
    )

    total_timesteps = 1_000_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name="SAC_Waypoints"
    )

    model.save("sac_waypoints_final")
    print("Обучение завершено. Модель сохранена.")

    demo_env = RobotEnvWaypoints(render_mode='human', max_steps=500, num_waypoints=6)
    obs, _ = demo_env.reset()
    best_model = SAC.load("./best_model_waypoints/best_model.zip")
    for _ in range(1000):
        action, _ = best_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = demo_env.step(action)
        if done or truncated:
            obs, _ = demo_env.reset()
    demo_env.close()