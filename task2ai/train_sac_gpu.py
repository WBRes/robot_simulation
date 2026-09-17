"""
Двухэтапное обучение.

Этап 1:python train_sac_gpu.py

Этап 2 (полный мир):python train_sac_gpu.py 2
"""
import sys
import torch
from train_env import RobotEnvWaypoints

STAGE = int(sys.argv[1]) if len(sys.argv) > 1 else 1

TOTAL_TIMESTEPS = {1: 600_000, 2: 800_000}

def make_env():
    return RobotEnvWaypoints(render_mode=None, stage=STAGE)


if __name__ == '__main__':
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Этап {STAGE} | устройство: {device} ===")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    num_envs = 4
    env = SubprocVecEnv([make_env for _ in range(num_envs)])
    eval_env = RobotEnvWaypoints(render_mode=None, stage=STAGE)

    stop_on_reward = None
    if STAGE == 1:
        stop_on_reward = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./best_stage{STAGE}/",
        log_path=f"./logs_stage{STAGE}/",
        eval_freq=10_000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        callback_on_new_best=stop_on_reward,
    )
    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=250)

    if STAGE == 1:
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=1024,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            target_entropy=-1.0,
            target_update_interval=1,
            verbose=1,
            tensorboard_log="./sac_waypoints_tensorboard/",
            device=device,
            policy_kwargs=dict(net_arch=[512, 512]),
        )
    else:
        model = SAC.load("./best_stage1/best_model.zip", device=device)
        model.set_env(env)

        from stable_baselines3.common.utils import get_schedule_fn

        model.learning_rate = 1e-4
        model.lr_schedule = get_schedule_fn(1e-4)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./best_stage{STAGE}/",
        log_path=f"./logs_stage{STAGE}/",
        eval_freq=20_000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=f"./checkpoints_stage{STAGE}/",
        name_prefix=f"sac_stage{STAGE}",
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS[STAGE],
        callback=[eval_callback, checkpoint_callback],
        tb_log_name=f"SAC_Waypoints_stage{STAGE}",
    )

    model.save(f"sac_waypoints_stage{STAGE}_final")
    print(f"Этап {STAGE} завершён. Модель сохранена.")

    env.close()
    eval_env.close()