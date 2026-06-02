import pygame
import random
import math

class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.waypoints = [(100, 100), (700, 100), (700, 500), (100, 500),
                          (400, 300), (100, 500), (700, 500), (400, 300), (100, 100)]
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

        self.following_target = False
        self.target = None          
        self.target_type = None    

        self.avoiding = False
        self.avoid_steps = 0
        self.avoid_angle = 0

        self.manual_mode = False

        self.waypoint_stuck_frames = 0
        self.waypoint_min_distance = float('inf')
        self.waypoint_best_distance = float('inf')

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

    def obstacle_ahead(self, objects, distances=[40,45,50], angles=[0,-30,30]):
        if self.manual_mode:
            return False
        for dist in distances:
            for ang in angles:
                rad = math.radians(self.direction + ang)
                fx = self.x + dist * math.cos(rad)
                fy = self.y + dist * math.sin(rad)
                for obj in objects:
                    if isinstance(obj, (Person, Litter, Car, PublicPlace, Cyclist)):
                        d = math.hypot(fx - obj.x, fy - obj.y)
                        if d < 15:
                            return True
        return False

    def avoid_collision(self, objects, min_dist=18):
        for obj in objects:
            dist = math.hypot(self.x - obj.x, self.y - obj.y)
            if dist < min_dist:
                angle = math.atan2(self.y - obj.y, self.x - obj.x)
                self.x += 3 * math.cos(angle)
                self.y += 3 * math.sin(angle)
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

    def move(self, people, litters, cars, places, cyclists):
        all_obstacles = people + litters + cars + places + cyclists
        if not self.manual_mode:
            self.avoid_collision(all_obstacles, min_dist=18)

        if self.manual_mode:
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
            if dist_to_target <= self.camera_range or already_seen:
                self.following_target = False
                self.target = None
                self.target_type = None
            else:
                if dist_to_target < 25:
                    back_angle = math.radians(self.direction + 180)
                    self.x -= 1 * math.cos(back_angle)
                    self.y -= 1 * math.sin(back_angle)
                    self.direction += 30
                else:
                    if dist_to_target > 0:
                        self.x += dx / dist_to_target * self.speed
                        self.y += dy / dist_to_target * self.speed
                self.avoid_collision(all_obstacles, min_dist=15)
                self.clamp_position()
                return

        # 2. Объезд препятствий
        if not self.avoiding:
            if self.obstacle_ahead(all_obstacles):
                self.avoiding = True
                self.avoid_steps = 20
                self.avoid_angle = self.direction + 45
        if self.avoiding:
            rad = math.radians(self.avoid_angle)
            self.x += self.speed * math.cos(rad)
            self.y += self.speed * math.sin(rad)
            self.avoid_steps -= 1
            if self.avoid_steps <= 0:
                self.avoiding = False
            else:
                if self.obstacle_ahead(all_obstacles, distances=[30], angles=[0]):
                    self.avoid_angle += 20
                    self.avoid_steps = max(self.avoid_steps, 20)
            self.direction = self.avoid_angle
            self.avoid_collision(all_obstacles, min_dist=15)
            self.clamp_position()
            return

        # 3. Обычное движение по waypoints
        target_x, target_y = self.waypoints[self.current_wp]
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.hypot(dx, dy)

        if dist < self.speed:
            self.x, self.y = target_x, target_y
            self.current_wp = (self.current_wp + 1) % len(self.waypoints)
            self.waypoint_stuck_frames = 0
            self.waypoint_min_distance = float('inf')
            self.waypoint_best_distance = float('inf')
            self.clamp_position()
            return

        if dist < self.waypoint_min_distance:
            self.waypoint_min_distance = dist

        self.waypoint_stuck_frames += 1
        stuck = False
        if self.waypoint_stuck_frames >= 90:
            if self.waypoint_min_distance > self.waypoint_best_distance - 10:
                stuck = True
        if not stuck and self.waypoint_stuck_frames >= 60 and dist > 50:
            if self.waypoint_min_distance > self.waypoint_best_distance - 5:
                stuck = True

        if stuck:
            old_wp = self.current_wp
            self.current_wp = (self.current_wp + 1) % len(self.waypoints)
            self.waypoint_stuck_frames = 0
            self.waypoint_min_distance = float('inf')
            self.waypoint_best_distance = float('inf')
            return

        if self.waypoint_min_distance < self.waypoint_best_distance - 1:
            self.waypoint_best_distance = self.waypoint_min_distance
            self.waypoint_stuck_frames = 0
            self.waypoint_min_distance = float('inf')

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
        rect_w = rect_h = 10
        rect_left = int(self.x) - rect_w // 2
        rect_top = int(self.y) - rect_h // 2
        if show_all_litters:
            pygame.draw.rect(screen, 'green', (rect_left, rect_top, rect_w, rect_h), 2)
        else:
            if self.counted:
                pygame.draw.rect(screen, 'green', (rect_left, rect_top, rect_w, rect_h), 2)

class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.counted = False

    def draw(self, screen, show_all_cars):
        rect_w, rect_h = 20, 12
        rect_left = int(self.x) - rect_w // 2
        rect_top = int(self.y) - rect_h // 2
        if show_all_cars:
            pygame.draw.ellipse(screen, 'orange', (rect_left, rect_top, rect_w, rect_h), 2)
        else:
            if self.counted:
                pygame.draw.ellipse(screen, 'orange', (rect_left, rect_top, rect_w, rect_h), 2)

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
        if self.type == 'скамейка':
            color = (139, 69, 19)
            size = 6
        elif self.type == 'турник':
            color = (100, 100, 100)
            size = 4
        else:
            color = (160, 82, 45)
            size = 10

        if show_all_public_places:
            pygame.draw.circle(screen, color, (int(self.x), int(self.y)), size, 2)
        else:
            if self.counted:
                pygame.draw.circle(screen, color, (int(self.x), int(self.y)), size, 2)

class FenceBreak:
    def __init__(self, x, y, side, width=40):
        self.x = x
        self.y = y
        self.side = side
        self.width = width
        self.detected = False
        
def draw_fence(screen, fence_breaks, show_all_fence):
    fence_color = (100, 100, 100)
    thickness = 3

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
            pygame.draw.line(screen, fence_color, (x, 600), (end_x, 600), thickness)
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
            pygame.draw.line(screen, fence_color, (800, y), (800, end_y), thickness)
            y = end_y
        else:
            y += 1

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Robot simulation")
clock = pygame.time.Clock()
FPS = 60
random.seed(42)

robot = Robot(400, 300)
people_colors = ['red', 'blue', 'green', 'yellow', 'purple']
people = [Person(random.randint(100, 700), random.randint(100, 500), people_colors[i]) for i in range(5)]
litters = [Litter(random.randint(50, 750), random.randint(50, 550)) for i in range(8)]
cars = [Car(random.randint(50, 750), random.randint(50, 550)) for i in range(4)]
places = [
    PublicPlace(200, 200, 'скамейка'),
    PublicPlace(600, 150, 'скамейка'),
    PublicPlace(300, 450, 'турник'),
    PublicPlace(650, 400, 'беседка'),
]
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

mode = False

show_all_people = mode
show_all_cyclists = mode
show_all_litters = mode
show_all_cars = mode
show_all_public_places = mode
show_all_fence = mode

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
    robot.move(people, litters, cars, places, cyclists)
    robot.update_camera_direction(people, cyclists)

    for p in people:
        dist = math.hypot(robot.x - p.x, robot.y - p.y)
        if dist <= 20:
            angle = math.atan2(robot.y - p.y, robot.x - p.x)
            robot.x += 5 * math.cos(angle)
            robot.y += 5 * math.sin(angle)
            robot.clamp_position()
    for c in cyclists:
        dist = math.hypot(robot.x - c.x, robot.y - c.y)
        if dist <= 20:
            angle = math.atan2(robot.y - c.y, robot.x - c.x)
            robot.x += 5 * math.cos(angle)
            robot.y += 5 * math.sin(angle)
            robot.clamp_position()

    for p in people:
        dist = math.hypot(robot.x - p.x, robot.y - p.y)
        lidar_sees = dist <= robot.lidar_range
        camera_sees = robot.object_in_cam(p.x, p.y)
        if lidar_sees and camera_sees and p.color not in robot.seen_person_colors:
            robot.seen_person_colors.add(p.color)
            robot.person_count += 1
            print(f"Обнаружен человек {p.color}, всего обнаружено: {robot.person_count}")

    for c in cyclists:
        dist = math.hypot(robot.x - c.x, robot.y - c.y)
        lidar_sees = dist <= robot.lidar_range
        camera_sees = robot.object_in_cam(c.x, c.y)
        if lidar_sees and camera_sees and c.color not in robot.seen_cyclist_colors:
            robot.seen_cyclist_colors.add(c.color)
            robot.cyclist_count += 1
            print(f"Обнаружен велосипедист {c.color}, всего обнаружено: {robot.cyclist_count}")

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

    screen.fill('grey')
    pygame.draw.line(screen, 'white', (bike_path_x_min, bike_path_y-2), (bike_path_x_max, bike_path_y-2), 1)
    pygame.draw.line(screen, (150,150,150), (bike_path_x_min, bike_path_y), (bike_path_x_max, bike_path_y), 4)
    pygame.draw.line(screen, 'white', (bike_path_x_min, bike_path_y+2), (bike_path_x_max, bike_path_y+2), 1)
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
    draw_fence(screen, fence_breaks, show_all_fence)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
