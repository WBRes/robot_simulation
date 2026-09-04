import sys
import os
import pygame
import torch
import numpy as np
from stable_baselines3 import SAC
from train_sac_gpu import RobotEnvWaypoints

MODEL_PATH = "./best_model_waypoints/best_model.zip"

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Ошибка: модель не найдена по пути {MODEL_PATH}")
        sys.exit(1)

    print(f"Загрузка модели: {MODEL_PATH}")
    model = SAC.load(MODEL_PATH)
    print("Модель загружена.")

    env = RobotEnvWaypoints(render_mode='human', max_steps=500, num_waypoints=6)
    obs, info = env.reset()

    total_reward = 0
    episodes = 0
    steps = 0
    running = True

    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    steps = 0
                    print("Эпизод перезапущен.")
                    continue

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if hasattr(env, 'screen') and env.screen:
            pygame.display.set_caption(f"Robot Test, Reward: {total_reward}, Steps: {steps}")

        if done or truncated:
            episodes += 1
            print(f"Эпизод {episodes} завершён. Награда: {total_reward}, шагов: {steps}")
            obs, info = env.reset()
            total_reward = 0
            steps = 0

        clock.tick(30)

    env.close()
    print("Тестирование завершено.")

if __name__ == "__main__":
    main()