import pygame
import random
import math
from simulation import (Robot, Person, Cyclist, Litter, Car, PublicPlace,
                        FenceBreak, Tree, draw_fence,wwwwwww generate_good_position,
                        build_lidar_obstacles, log_event)

def draw_rotated_text(surface, text, pos, angle, color=(255, 255, 255)):
    text_surface = font.render(text, True, color)
    rotated_surface = pygame.transform.rotate(text_surface, angle)
    rotated_rect = rotated_surface.get_rect(center=pos)
    surface.blit(rotated_surface, rotated_rect)

#Основные настройки окна
pygame.init()
screen = pygame.display.set_mode((800, 600))
font = pygame.font.SysFont('arial', 30)
pygame.display.set_caption("Robot simulation")
clock = pygame.time.Clock()
# random.seed(42)
FPS = 60

#Инициализация характеристик объектов
PEOPLE_COLORS = ['red', 'blue', 'green', 'yellow', 'purple']
CYCLIST_COLORS = ['red', 'blue', 'green', 'yellow', 'purple']
WEEK_COL = {
    'people': 5,
    'cyclists': 3,
    'litter_detected': 8,
    'litter_collected': 8,
    'cars': 4,
    'trees': 5,
    'fence_breaks': 4,
    'places': 4
}
VARIATION = 2

#Инициализация объектов
robot = Robot(400, 300)

people_col = random.randint(WEEK_COL['people'] - VARIATION, WEEK_COL['people'] + VARIATION)
people = []
for i in range(people_col):
    color = random.choice(PEOPLE_COLORS)
    people.append(Person(random.randint(100, 700), random.randint(100, 500), color))

litter_col = random.randint(WEEK_COL['litter_detected'] - VARIATION, WEEK_COL['litter_detected'] + VARIATION)
litter_pos = []
first_pos = [random.randint(50, 750), random.randint(50, 550)]
litter_pos.append(first_pos)
for i in range(litter_col - 1):
    litter_pos.append(generate_good_position(litter_pos, [50, 750], [50, 550], min_dist=20, max_attempts=100))
litters = [Litter(litter_pos[i][0], litter_pos[i][1]) for i in range(litter_col)]

car_col = random.randint(WEEK_COL['cars'] - VARIATION, WEEK_COL['cars'] + VARIATION)
car_pos = []
first_car = [random.randint(710, 750), random.randint(410, 550)]
car_pos.append(first_car)
for i in range(car_col - 1):
    car_pos.append(generate_good_position(car_pos, [710, 750], [410, 550], min_dist=20, max_attempts=100))
cars = [Car(car_pos[i][0], car_pos[i][1]) for i in range(car_col)]

places_col = random.randint(WEEK_COL['places'] - VARIATION, WEEK_COL['places'] + VARIATION)
place_types = ['скамейка', 'скамейка', 'турник', 'беседка']
place_pos = []
first_place = [random.randint(50, 750), random.randint(50, 550)]
place_pos.append(first_place)
for i in range(places_col - 1):
    place_pos.append(generate_good_position(place_pos, [50, 750], [50, 550], min_dist=20, max_attempts=100))
places = []
for i in range(places_col):
    t = random.choice(place_types)
    places.append(PublicPlace(place_pos[i][0], place_pos[i][1], t))

bike_path_y = 150
bike_path_x_min = 50
bike_path_x_max = 750

cyclists_col = random.randint(WEEK_COL['cyclists'] - VARIATION, WEEK_COL['cyclists'] + VARIATION)
cyclists = []
for i in range(cyclists_col):
    x = random.randint(bike_path_x_min, bike_path_x_max)
    direction = 1 if random.random() < 0.5 else -1
    color = random.choice(CYCLIST_COLORS)
    cyclists.append(Cyclist(x, bike_path_y, direction, color))

fence_breaks = []
for i in range(2):
    fence_breaks.append(FenceBreak(random.randint(50, 750), 0, 'top', width=60))
for i in range(2):
    fence_breaks.append(FenceBreak(random.randint(50, 750), 600, 'bottom', width=60))
for i in range(1):
    fence_breaks.append(FenceBreak(random.randint(50, 550), 0, 'left', width=60))
for i in range(1):
    fence_breaks.append(FenceBreak(random.randint(50, 550), 800, 'right', width=60))

tree_count = random.randint(WEEK_COL['trees'] - VARIATION, WEEK_COL['trees'] + VARIATION)
tree_types = ['Ёлка', 'Дуб']
tree_pos = []
first_tree = [random.randint(50, 680), random.randint(50, 550)]
tree_pos.append(first_tree)
for i in range(tree_count - 1):
    tree_pos.append(generate_good_position(tree_pos, [50, 680], [50, 550], min_dist=50, max_attempts=100))
trees = []
for i in range(tree_count):
    t = random.choice(tree_types)
    trees.append(Tree(tree_pos[i][0], tree_pos[i][1], t))

mode = False

show_all_people = mode
show_all_cyclists = mode
show_all_litters = mode
show_all_cars = mode
show_all_public_places = mode
show_all_fence = mode
show_all_trees = mode

STATS_INTERVAL = 30000
last_stats_time = pygame.time.get_ticks()

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
    robot.move(people, cars, places, cyclists, trees, litters)
    lidar_obstacles = build_lidar_obstacles(people, cyclists, litters, cars, places, trees)
    robot.update_lidar(lidar_obstacles)

    for place in places:
        dist = math.hypot(robot.x - place.x, robot.y - place.y)
        if not place.counted and dist <= robot.lidar_range:
            robot.place_count += 1
            place.counted = True
            log_event(f"Обнаружено место: {place.type} на координатах ({place.x}, {place.y})")
            #print(f"Обнаружено место: {place.type} на координатах ({place.x}, {place.y})")
        for p in people:
            dist_to_place = math.hypot(place.x - p.x, place.y - p.y)
            if dist_to_place <= place.radius:
                dist_to_robot = math.hypot(robot.x - p.x, robot.y - p.y)
                lidar_sees = dist_to_robot <= robot.lidar_range
                camera_sees = robot.object_in_cam(p.x, p.y)
                if lidar_sees and camera_sees and not p.color in place.color_seen:
                    place.visitor_count += 1
                    place.color_seen.add(p.color)
                    log_event(f"{place.type} посетил человек {p.color}, всего посетителей было: {place.visitor_count}")
                    #print(f"{place.type} посетил человек {p.color}, всего посетителей было: {place.visitor_count}")

    for litter in litters[:]:
        dist = math.hypot(robot.x - litter.x, robot.y - litter.y)
        if not litter.counted and dist <= robot.lidar_range:
            litter.counted = True
            robot.litter_count += 1
            log_event(f"Мусор на координатах ({litter.x}, {litter.y})")
            #print(f"Мусор на координатах ({litter.x}, {litter.y})")
        if math.hypot(robot.x - litter.x, robot.y - litter.y) < 15:
            litters.remove(litter)
            robot.litter_collected += 1
            log_event(f"Мусор собран на координатах ({litter.x}, {litter.y}). Всего собрано: {robot.litter_collected}")
            #print(f"Мусор собран на координатах ({litter.x}, {litter.y}). Всего собрано: {robot.litter_collected}")

    for car in cars:
        dist = math.hypot(robot.x - car.x, robot.y - car.y)
        if not car.counted and dist <= robot.lidar_range:
            car.counted = True
            robot.car_count += 1
            log_event(f"Машина на координатах ({car.x}, {car.y}), всего обнаружено: {robot.car_count}")
            #print(f"Машина на координатах ({car.x}, {car.y}), всего обнаружено: {robot.car_count}")

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
                log_event(f"Поломка на {br.side} стороне, всего: {robot.fence_break_count}")
                #print(f"Поломка на {br.side} стороне, всего: {robot.fence_break_count}")

    for t in trees:
        dist = math.hypot(robot.x - t.x, robot.y - t.y)
        if not t.counted and dist <= robot.lidar_range:
            t.counted = True
            robot.tree_count += 1
            log_event(f"Дерево на координатах ({t.x}, {t.y}), всего обнаружено: {robot.tree_count}")
            #print(f"Дерево на координатах ({t.x}, {t.y}), всего обнаружено: {robot.tree_count}")

    current_time = pygame.time.get_ticks()
    if current_time - last_stats_time >= STATS_INTERVAL:
        current_stats = {
            'people': robot.person_count,
            'cyclists': robot.cyclist_count,
            'litter_detected': robot.litter_count,
            'litter_collected': robot.litter_collected,
            'cars': robot.car_count,
            'trees': robot.tree_count,
            'fence_breaks': robot.fence_break_count,
            'places': robot.place_count
        }
        print("\n\n\n\n\n\n\n\nСтатистика:")
        for key, val in current_stats.items():
            base = WEEK_COL.get(key, 0)
            diff = val - base
            if diff >= 0:
                sign = "+"
            else:
                sign = " "
            print(f"{key}: {val} (На прошлой неделе: {base}, изменение: {sign}{diff})")
        log_event("Статистический отчёт: " + str(current_stats))
        last_stats_time = current_time
    screen.fill((48, 194, 107))
    # велодорожка
    pygame.draw.line(screen, 'white', (bike_path_x_min - 50, bike_path_y - 2), (bike_path_x_max + 50, bike_path_y - 2),
                     1)
    pygame.draw.line(screen, (150, 150, 150), (bike_path_x_min - 50, bike_path_y), (bike_path_x_max + 50, bike_path_y),
                     4)
    pygame.draw.line(screen, 'white', (bike_path_x_min - 50, bike_path_y + 2), (bike_path_x_max + 50, bike_path_y + 2),
                     1)
    # обычная дорога
    pygame.draw.rect(screen, (202, 204, 206), (0, 500, 500, 8))
    pygame.draw.rect(screen, (202, 204, 206), (500, 500, 8, 100))
    # парковка
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