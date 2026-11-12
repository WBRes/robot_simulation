class CurriculumManager:
    CURRICULUM = [
        # Этап 1: Статическая среда (медленно и аккуратно)
        {
            'start_step': 0,
            'stage': 1,
            'name': '1.1 - Простая среда (1 круг)',
            'env_kwargs': {
                'num_cars': 1,
                'num_places': 1,
                'num_trees': 1,
                'num_buildings': 1,
                'litters_range': (1, 2),
                'max_steps': 8000,
            },
            'reward_kwargs': {
                'R_STEP': -0.02,
                'R_PROGRESS': 0.1,
                'R_COLLISION_STATIC': -150.0,
                'R_WAYPOINT': 150.0,
                'R_LITTER': 150.0,
                'R_ROUTE_DONE': 300.0,
                'R_NEAR_OBSTACLE': -0.1,
            }
        },
        {
            'start_step': 300_000,
            'stage': 1,
            'name': '1.2 - Средняя среда (1.5 круга)',
            'env_kwargs': {
                'num_cars': 2,
                'num_places': 2,
                'num_trees': 2,
                'num_buildings': 2,
                'litters_range': (2, 3),
                'max_steps': 10000,
            },
            'reward_kwargs': {
                'R_STEP': -0.02,
                'R_PROGRESS': 0.1,
                'R_COLLISION_STATIC': -150.0,
                'R_WAYPOINT': 150.0,
                'R_LITTER': 150.0,
                'R_ROUTE_DONE': 300.0,
                'R_NEAR_OBSTACLE': -0.1,
            }
        },
        {
            'start_step': 600_000,
            'stage': 1,
            'name': '1.3 - Полный этап 1 (2 круга)',
            'env_kwargs': {
                'num_cars': 3,
                'num_places': 3,
                'num_trees': 3,
                'num_buildings': 3,
                'litters_range': (2, 4),
                'max_steps': 12000,
            },
            'reward_kwargs': {
                'R_STEP': -0.02,
                'R_PROGRESS': 0.1,
                'R_COLLISION_STATIC': -150.0,
                'R_WAYPOINT': 150.0,
                'R_LITTER': 150.0,
                'R_ROUTE_DONE': 300.0,
                'R_NEAR_OBSTACLE': -0.1,
            }
        },

        # Этап 2: Динамическая среда (осторожное патрулирование)
        {
            'start_step': 900_000,
            'stage': 2,
            'name': '2.1 - 1 велосипедист (2 круга)',
            'env_kwargs': {
                'num_cars': 4,
                'num_places': 4,
                'num_trees': 5,
                'num_buildings': 3,
                'num_cyclists': 1,
                'num_people': 0,
                'litters_range': (3, 4),
                'max_steps': 12000,
            },
            'reward_kwargs': {
                'R_STEP': -0.02,
                'R_PROGRESS': 0.1,
                'R_COLLISION_STATIC': -150.0,
                'R_COLLISION_DYNAMIC': -150.0,
                'R_WAYPOINT': 150.0,
                'R_LITTER': 150.0,
                'R_ROUTE_DONE': 300.0,
                'R_NEAR_OBSTACLE': -0.1,
            }
        },
        {
            'start_step': 1_200_000,
            'stage': 2,
            'name': '2.2 - 1 велосипедист + 1 человек (2 круга)',
            'env_kwargs': {
                'num_cars': 4,
                'num_places': 4,
                'num_trees': 5,
                'num_buildings': 3,
                'num_cyclists': 1,
                'num_people': 1,
                'litters_range': (3, 4),
                'max_steps': 12000,
            },
            'reward_kwargs': {
                'R_STEP': -0.02,
                'R_PROGRESS': 0.1,
                'R_COLLISION_STATIC': -150.0,
                'R_COLLISION_DYNAMIC': -150.0,
                'R_WAYPOINT': 150.0,
                'R_LITTER': 150.0,
                'R_ROUTE_DONE': 300.0,
                'R_NEAR_OBSTACLE': -0.1,
            }
        },
        {
            'start_step': 1_500_000,
            'stage': 2,
            'name': '2.3 - 1 велосипедист + 2 человека (2.5 круга)',
            'env_kwargs': {
                'num_cars': 4,
                'num_places': 4,
                'num_trees': 5,
                'num_buildings': 3,
                'num_cyclists': 1,
                'num_people': 2,
                'litters_range': (4, 5),
                'max_steps': 14000,
            },
            'reward_kwargs': {
                'R_STEP': -0.02,
                'R_PROGRESS': 0.1,
                'R_COLLISION_STATIC': -150.0,
                'R_COLLISION_DYNAMIC': -150.0,
                'R_WAYPOINT': 150.0,
                'R_LITTER': 150.0,
                'R_ROUTE_DONE': 300.0,
                'R_NEAR_OBSTACLE': -0.1,
            }
        },
        {
            'start_step': 1_800_000,
            'stage': 2,
            'name': '2.4 - 2 велосипедиста + 3 человека (2.5 круга)',
            'env_kwargs': {
                'num_cars': 4,
                'num_places': 4,
                'num_trees': 5,
                'num_buildings': 3,
                'num_cyclists': 2,
                'num_people': 3,
                'litters_range': (5, 6),
                'max_steps': 14000,
            },
            'reward_kwargs': {
                'R_STEP': -0.02,
                'R_PROGRESS': 0.1,
                'R_COLLISION_STATIC': -150.0,
                'R_COLLISION_DYNAMIC': -150.0,
                'R_WAYPOINT': 150.0,
                'R_LITTER': 150.0,
                'R_ROUTE_DONE': 300.0,
                'R_NEAR_OBSTACLE': -0.1,
            }
        },
        {
            'start_step': 2_100_000,
            'stage': 2,
            'name': '2.5 - Финал: 2 велосипедиста + 5 человека (3 круга)',
            'env_kwargs': {
                'num_cars': 4,
                'num_places': 4,
                'num_trees': 5,
                'num_buildings': 3,
                'num_cyclists': 2,
                'num_people': 5,
                'litters_range': (6, 7),
                'max_steps': 16000,
            },
            'reward_kwargs': {
                'R_STEP': -0.02,
                'R_PROGRESS': 0.1,
                'R_COLLISION_STATIC': -150.0,
                'R_COLLISION_DYNAMIC': -150.0,
                'R_WAYPOINT': 150.0,
                'R_LITTER': 150.0,
                'R_ROUTE_DONE': 300.0,
                'R_NEAR_OBSTACLE': -0.1,
            }
        },
    ]

    TOTAL_TIMESTEPS = 2_400_000


    def __init__(self):
        self.current_substage_idx = 0
        self.last_substage_name = None

    DEFAULT_ENV = {
        'wp': 6,
        'wp_radius': 40,
    }

    DEFAULT_REWARD = {
        'R_ROUTE_DONE': 300.0,
        'R_ALL_LITTER': 500.0,
        'R_BOUNDARY': -10.0,
        'R_COLLISION_DYNAMIC': -50.0,
    }


    def get_env_kwargs(self, substage):
        kwargs = dict(self.DEFAULT_ENV)
        kwargs.update(substage['env_kwargs'])
        return kwargs

    def get_reward_kwargs(self, substage):
        kwargs = dict(self.DEFAULT_REWARD)
        kwargs.update(substage['reward_kwargs'])
        return kwargs

    def get_segment_bounds(self, idx):
        start = self.CURRICULUM[idx]['start_step']
        end = (self.CURRICULUM[idx + 1]['start_step']
               if idx + 1 < len(self.CURRICULUM)
               else self.TOTAL_TIMESTEPS)
        return start, end, end - start

    def get_substage_for_timestep(self, timestep):
        current_substage = self.CURRICULUM[0]
        for substage in self.CURRICULUM:
            if timestep >= substage['start_step']:
                current_substage = substage
            else:
                break
        return current_substage

    def get_substage_index(self, timestep):
        idx = 0
        for i, substage in enumerate(self.CURRICULUM):
            if timestep >= substage['start_step']:
                idx = i
        return idx

    def check_transition(self, timestep):
        new_idx = self.get_substage_index(timestep)
        if new_idx != self.current_substage_idx:
            self.current_substage_idx = new_idx
            new_substage = self.CURRICULUM[new_idx]
            self.last_substage_name = new_substage['name']
            return True, new_substage
        return False, None

    def get_initial_substage(self):
        return self.CURRICULUM[0]

    def print_curriculum(self):
        print("\n" + "=" * 70)
        print("Поэтапное обучение")
        print("=" * 70)
        for i, substage in enumerate(self.CURRICULUM):
            next_step = (self.CURRICULUM[i + 1]['start_step']
                         if i + 1 < len(self.CURRICULUM)
                         else self.TOTAL_TIMESTEPS)
            duration = next_step - substage['start_step']
            print(f"\n[{i + 1}/{len(self.CURRICULUM)}] {substage['name']}")
            print(f"    Диапазон: {substage['start_step']:,} - {next_step:,} шагов ({duration:,} шагов)")
            print(f"    Этап: {substage['stage']}")
            print(f"    Объекты: машины={substage['env_kwargs'].get('num_cars', 0)}, "
                  f"люди={substage['env_kwargs'].get('num_people', 0)}, "
                  f"велосипедисты={substage['env_kwargs'].get('num_cyclists', 0)}, "
                  f"мусор={substage['env_kwargs']['litters_range']}")
            print(f"    Награды: R_STEP={substage['reward_kwargs']['R_STEP']}, "
                  f"R_COLLISION_STATIC={substage['reward_kwargs'].get('R_COLLISION_STATIC', -50.0)}")
        print("\n" + "=" * 70)
        print(f"Всего шагов обучения: {self.TOTAL_TIMESTEPS:,}")
        print("=" * 70 + "\n")