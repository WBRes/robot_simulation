import pygame
import random
import math

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

    def obstacle_ahead(self, objects, distances=[40,45,50], angles=[-45, -20, 0, 20, 45]):
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

        
    def move(self, people, litters, cars, places, cyclists, trees):
        all_obstacles = people + litters + cars + places + cyclists + trees
        
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
                    print(f"Обнаружен человек {self.target.color}, всего обнаружено: {self.person_count}")
                else:
                    self.seen_cyclist_colors.add(self.target.color)
                    self.cyclist_count += 1
                    print(f"Обнаружен велосипедист {self.target.color}, всего обнаружено: {self.cyclist_count}")
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
                        if isinstance(obj,(Person, Litter, Car, PublicPlace, Cyclist)):
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
            left = self.x - w//2
            top = self.y - h//2
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
        pygame.draw.circle(screen, (255, 200, 100), (left + 2, top + h//2 - 1), 1)
        pygame.draw.circle(screen, (255, 200, 100), (left + 2, top + h//2 + 1), 1)


class PublicPlace:
    def __init__(self, x, y, place_type):
        self.x = x
        self.y = y
        self.type = place_type
        self.radius = 25
        self.visitor_count = 0
        self.seen_colors = set()
        self.counted = False

    def draw(self, screen, show_all_public_places):
        if not (show_all_public_places or self.counted):
            return

        if self.type == 'скамейка':
            w = 16
            h = 10
            left = self.x - w//2
            top = self.y - h//2
            pygame.draw.rect(screen, (139, 69, 19), (left, top, w, h), 2)
            y1 = top + h//3
            y2 = top + 2*h//3
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
                if br.x - br.width//2 <= x <= br.x + br.width//2:
                    broken = True
                    x = br.x + br.width//2
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
                if br.x - br.width//2 <= x <= br.x + br.width//2:
                    broken = True
                    x = br.x + br.width//2
                    break
        if not broken:
            end_x = min(x + 10, 800)
            pygame.draw.line(screen, fence_color, (x, 600-thickness/2), (end_x, 600-thickness/2), thickness)
            x = end_x
        else:
            x += 1
    y = 0
    while y < 600:
        broken = False
        for br in fence_breaks:
            if br.side == 'left' and need_break(br):
                if br.x - br.width//2 <= y <= br.x + br.width//2:
                    broken = True
                    y = br.x + br.width//2
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
                if br.x - br.width//2 <= y <= br.x + br.width//2:
                    broken = True
                    y = br.x + br.width//2
                    break
        if not broken:
            end_y = min(y + 10, 600)
            pygame.draw.line(screen, fence_color, (800-thickness/2, y), (800-thickness/2, end_y), thickness)
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
        x = self.x - trunk_w/2
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
                pygame.draw.circle(screen, (92, 169, 4), (x+trunk_w/2, y-trunk_h), 30)

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

def draw_rotated_text(surface, text, pos, angle, color=(255,255,255)):
    text_surface = font.render(text, True, color)
    rotated_surface = pygame.transform.rotate(text_surface, angle)
    rotated_rect = rotated_surface.get_rect(center=pos)
    surface.blit(rotated_surface, rotated_rect)


pygame.init()
screen = pygame.display.set_mode((800, 600))
font = pygame.font.SysFont('arial', 30)
pygame.display.set_caption("Robot simulation")
clock = pygame.time.Clock()
random.seed(42)
FPS = 60

robot = Robot(400, 300)
people_colors = ['red', 'blue', 'green', 'yellow', 'purple']
people = [Person(random.randint(100, 700), random.randint(100, 500), people_colors[i]) for i in range(5)]
litter_pos = []
litter_pos.append([random.randint(50, 750), random.randint(50, 550)])
for i in range(7):
    litter_pos.append(generate_good_position(litter_pos, [50, 750], [50, 550], min_dist=20, max_attempts=100))
litters = [Litter(litter_pos[i][0], litter_pos[i][1]) for i in range(8)]
car_pos = []
car_pos.append([random.randint(710, 750), random.randint(410, 550)])
for i in range(3):
    car_pos.append(generate_good_position(car_pos, [710, 750], [410, 550], min_dist=20, max_attempts=100))
cars = [Car(car_pos[i][0], car_pos[i][1]) for i in range(4)]
place_pos = []
place_pos.append([random.randint(50, 750), random.randint(50, 550)])
for i in range(3):
    place_pos.append(generate_good_position(place_pos, [50, 750], [50, 550], min_dist=20, max_attempts=100))
places_list = ['скамейка', 'скамейка', 'турник', 'беседка']
places = [PublicPlace(place_pos[i][0], place_pos[i][1], places_list[i]) for i in range(4)]
bike_path_y = 150
bike_path_x_min = 50
bike_path_x_max = 750
cyclist_colors = ['red', 'blue', 'green', 'yellow', 'purple']
cyclists = [
    Cyclist(200, bike_path_y, 1, cyclist_colors[0]),
    Cyclist(400, bike_path_y, -1, cyclist_colors[1]),
    Cyclist(600, bike_path_y, 1, cyclist_colors[2]),
]
fence_breaks = []
for i in range(2):
    fence_breaks.append(FenceBreak(random.randint(50, 750), 0, 'top', width=60))
for i in range(2):
    fence_breaks.append(FenceBreak(random.randint(50, 750), 600, 'bottom', width=60))
for i in range(1):
    fence_breaks.append(FenceBreak(random.randint(50, 550), 0, 'left', width=60))
for i in range(1):
    fence_breaks.append(FenceBreak(random.randint(50, 550), 800, 'right', width=60))
tree_pos = []
tree_pos.append([random.randint(50, 680), random.randint(50, 550)])
for i in range(4):
    tree_pos.append(generate_good_position(tree_pos, [50, 680], [50, 550], min_dist=50, max_attempts=100))
tree_list = ['Ёлка', 'Дуб', 'Ёлка', 'Ёлка', 'Дуб']
trees = [Tree(tree_pos[i][0], tree_pos[i][1], tree_list[i]) for i in range(5)]

mode = False

show_all_people = mode
show_all_cyclists = mode
show_all_litters = mode
show_all_cars = mode
show_all_public_places = mode
show_all_fence = mode
show_all_trees = mode

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_e:
                show_all_people = not show_all_people
                show_all_cyclists = not show_all_cyclists
                show_all_litters = not show_all_litters
                show_all_cars = not show_all_cars
                show_all_public_places = not show_all_public_places
                show_all_fence = not show_all_fence
                show_all_trees = not show_all_trees
            elif event.key == pygame.K_q:
                robot.manual_mode = not robot.manual_mode
                print("Ручное управление:", "On" if robot.manual_mode else "Off")
                if not robot.manual_mode:
                    robot.following_target = False
                    robot.avoiding = False

    for p in people:
        p.move()
    for c in cyclists:
        c.move(bike_path_y, bike_path_x_min, bike_path_x_max)
    robot.choose_target(people, cyclists)
    robot.move(people, litters, cars, places, cyclists, trees)
    robot.update_camera_direction(people, cyclists)
            
    for place in places:
        dist = math.hypot(robot.x - place.x, robot.y - place.y)
        if not place.counted and dist <= robot.lidar_range:
            place.counted = True
            print(f"Обнаружено место: {place.type} на координатах ({place.x}, {place.y})")
        for p in people:
            dist_to_place = math.hypot(place.x - p.x, place.y - p.y)
            if dist_to_place <= place.radius:
                dist_to_robot = math.hypot(robot.x - p.x, robot.y - p.y)
                lidar_sees = dist_to_robot <= robot.lidar_range
                camera_sees = robot.object_in_cam(p.x, p.y)
                if lidar_sees and camera_sees and p.color not in place.seen_colors:
                    place.seen_colors.add(p.color)
                    place.visitor_count += 1
                    print(f"{place.type} посетил человек {p.color}, всего посетителей было: {place.visitor_count}")

    for litter in litters:
        dist = math.hypot(robot.x - litter.x, robot.y - litter.y)
        if not litter.counted and dist <= robot.lidar_range:
            litter.counted = True
            robot.litter_count += 1
            print(f"Мусор на координатах ({litter.x}, {litter.y}), всего обнаружено: {robot.litter_count}")

    for car in cars:
        dist = math.hypot(robot.x - car.x, robot.y - car.y)
        if not car.counted and dist <= robot.lidar_range:
            car.counted = True
            robot.car_count += 1
            print(f"Машина на координатах ({car.x}, {car.y}), всего обнаружено: {robot.car_count}")

    for br in fence_breaks:
        if not br.detected:
            if br.side == 'top':
                br_x, br_y = br.x, 0
            elif br.side == 'bottom':
                br_x, br_y = br.x, 600
            elif br.side == 'left':
                br_x, br_y = 0, br.x
            else:
                br_x, br_y = 800, br.x
            dist = math.hypot(robot.x - br_x, robot.y - br_y)
            if dist <= robot.lidar_range:
                br.detected = True
                robot.fence_break_count += 1
                print(f"Поломка на {br.side} стороне, всего: {robot.fence_break_count}")

    for t in trees:
        dist = math.hypot(robot.x - t.x, robot.y - t.y)
        if not t.counted and dist <= robot.lidar_range:
            t.counted = True
            robot.tree_count += 1
            print(f"Дерево на координатах ({t.x}, {t.y}), всего обнаружено: {robot.tree_count}")

    screen.fill((48, 194, 107))
    #велодорожка
    pygame.draw.line(screen, 'white', (bike_path_x_min-50, bike_path_y-2), (bike_path_x_max+50, bike_path_y-2), 1)
    pygame.draw.line(screen, (150,150,150), (bike_path_x_min-50, bike_path_y), (bike_path_x_max+50, bike_path_y), 4)
    pygame.draw.line(screen, 'white', (bike_path_x_min-50, bike_path_y+2), (bike_path_x_max+50, bike_path_y+2), 1)
    #обычная дорога
    pygame.draw.rect(screen, (202, 204, 206), (0, 500, 500, 8))
    pygame.draw.rect(screen, (202, 204, 206), (500, 500, 8, 100))
    #парковка
    pygame.draw.rect(screen, (126, 137, 148), (700, 400, 100, 150))
    draw_rotated_text(screen, "Parking", (750, 475), 90)

    
    robot.draw(screen)
    for p in people:
        p.draw(screen, robot, show_all_people)
    for c in cyclists:
        c.draw(screen, robot, show_all_cyclists)
    for litter in litters:
        litter.draw(screen, show_all_litters)
    for car in cars:
        car.draw(screen, show_all_cars)
    for place in places:
        place.draw(screen, show_all_public_places)
    for t in trees:
        t.draw(screen, show_all_trees)
    draw_fence(screen, fence_breaks, show_all_fence)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
