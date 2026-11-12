# tensorboard --logdir=".\sac_curriculum_tensorboard"
# python train_curriculum.py

import os
import sys
import json
import shutil

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import get_schedule_fn

from train_env import RobotEnvWaypoints
from curriculum_manager import CurriculumManager

from stable_baselines3.common.callbacks import BaseCallback


class EarlyStoppingCallback(BaseCallback):
    def __init__(self, patience=3, min_improvement=10.0, verbose=0):
        super().__init__(verbose)
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_reward = -float('inf')
        self.patience_counter = 0

    def _on_step(self) -> bool:
        if self.n_calls % (self.model.env.num_envs * 5000) == 0:
            infos = self.locals.get('infos', [])
            current_reward = -float('inf')

            for info in infos:
                if 'episode' in info and 'r' in info['episode']:
                    current_reward = info['episode']['r']
                    break

            if current_reward == -float('inf'):
                return True

            if current_reward > self.best_reward + self.min_improvement:
                self.best_reward = current_reward
                self.patience_counter = 0
                print(f"\nНовый рекорд: {current_reward:.1f}")
            else:
                self.patience_counter += 1
                print(f"\nНет улучшения ({self.patience_counter}/{self.patience})")

            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping! Обучение остановлено.")
                return False

        return True

NUM_ENVS = 4
EVAL_EVERY = 20_000
N_EVAL_EPISODES = 20
ROOT = "./curriculum"
PROGRESS_FILE = os.path.join(ROOT, "progress.json")


class TeeOutput:
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()


def build_vec_env(stage, env_kwargs, reward_kwargs, n_envs, dummy=False):
    def make():
        return RobotEnvWaypoints(render_mode=None, stage=stage,
                                 env_kwargs=env_kwargs, reward_kwargs=reward_kwargs)
    if dummy:
        return DummyVecEnv([make])
    return SubprocVecEnv([make for _ in range(n_envs)])


def make_model(env, device, lr=3e-4):
    return SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=512,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        target_entropy=-1.0,
        target_update_interval=1,
        verbose=1,
        tensorboard_log="./sac_curriculum_tensorboard/",
        device=device,
        policy_kwargs=dict(net_arch=[512, 512]),
    )


def load_model(path, env, device, lr):
    model = SAC.load(path, env=env, device=device)
    model.learning_rate = lr
    model.lr_schedule = get_schedule_fn(lr)
    return model


def save_progress(idx):
    os.makedirs(ROOT, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"next_substage": idx}, f)


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("next_substage", 0)
    return 0


def main():
    sys.stdout = TeeOutput("training_log.txt")
    sys.stderr = sys.stdout

    resume = "--resume" in sys.argv
    manager = CurriculumManager()
    manager.print_curriculum()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    start_idx = load_progress() if resume else 0
    if start_idx >= len(manager.CURRICULUM):
        print("Обучение уже завершено.")
        return

    for idx in range(start_idx, len(manager.CURRICULUM)):
        sub = manager.CURRICULUM[idx]
        stage = sub['stage']
        name = sub['name']
        seg_start, seg_end, duration = manager.get_segment_bounds(idx)
        env_kwargs = manager.get_env_kwargs(sub)
        reward_kwargs = manager.get_reward_kwargs(sub)

        print("\n" + "=" * 70)
        print(f"[{idx + 1}/{len(manager.CURRICULUM)}] {name}")
        print(f"Шаги: {seg_start:,} -> {seg_end:,}  ({duration:,})")
        print(f"Среда: {env_kwargs}")
        print(f"Награды: {reward_kwargs}")
        print("=" * 70 + "\n")

        sub_dir = os.path.join(ROOT, f"substage_{idx + 1}")
        os.makedirs(sub_dir, exist_ok=True)

        env = build_vec_env(stage, env_kwargs, reward_kwargs, NUM_ENVS)
        eval_env = build_vec_env(stage, env_kwargs, reward_kwargs, 1, dummy=True)

        prev_final = os.path.join(ROOT, f"substage_{idx}", "final.zip")

        if idx == 0:
            model = make_model(env, device, lr=3e-4)
        elif os.path.exists(prev_final):
            print(f"Загрузка модели из {prev_final}")
            model = SAC.load(prev_final, env=env, device=device)
            lr = 1e-4 if idx >= 4 else 3e-4
            model.learning_rate = lr
            model.lr_schedule = get_schedule_fn(lr)
        else:
            model = make_model(env, device, lr=3e-4)

        best_stage_dir = f"./best_stage{stage}/"
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_stage_dir,
            log_path=os.path.join(sub_dir, "eval_logs"),
            eval_freq=max(EVAL_EVERY // NUM_ENVS, 1),
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
            render=False,
            verbose=1,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=max(50_000 // NUM_ENVS, 1),
            save_path=os.path.join(sub_dir, "checkpoints"),
            name_prefix=f"sac_sub{idx + 1}",
        )

        early_stopping = EarlyStoppingCallback(patience=3, min_improvement=10.0)

        model.learn(
            total_timesteps=duration,
            callback=[eval_callback, checkpoint_callback, early_stopping],
            tb_log_name=f"SAC_sub{idx + 1}_{stage}",
            reset_num_timesteps=True,
        )

        model.save(os.path.join(sub_dir, "final"))
        best_src = os.path.join(best_stage_dir, "best_model.zip")
        if os.path.exists(best_src):
            shutil.copy(best_src, os.path.join(sub_dir, "best.zip"))

        print(f"\nПодэтап {idx + 1} завершён. Модель: {sub_dir}/final.zip\n")

        env.close()
        eval_env.close()
        save_progress(idx + 1)

    print("\n" + "=" * 70)
    print("Поэтапное обучение завершено")
    print("Финальные веса: ./best_stage2/best_model.zip")
    print("=" * 70)


if __name__ == '__main__':
    main()