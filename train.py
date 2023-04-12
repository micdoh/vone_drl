import yaml
import wandb
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from pathlib import Path
from datetime import datetime
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback
from callback import CustomCallback
from util_funcs import make_env, define_logs, choose_schedule, parse_args
import env.envs.VoneEnv as VoneEnv
import torch as th
th.autograd.set_detect_anomaly(True)


if __name__ == "__main__":

    args = parse_args()
    conf = yaml.safe_load(Path(args.env_file).read_text())
    callbacks = []

    start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    # Setup wandb run
    if not args.no_wandb:

        wandb.setup(wandb.Settings(program="train.py", program_relpath="train.py"))
        run = wandb.init(
            project=conf["project"],
            dir=args.log_dir,
            monitor_gym=False,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
        wandb.config.update(args)
        wandb.config.update(conf)
        run.name = args.id if args.id else run.id
        (
            log_dir,
            log_file,
            logger,
            model_save_file,
        ) = define_logs(run.id, args.log_dir, args.log)
        callbacks.append(
            WandbCallback(
                gradient_save_freq=0,
                verbose=2,
            )
        )
        wandb.define_metric("episode_number")
        wandb.define_metric("acceptance_ratio", step_metric="episode_number")
        conf["env_args"]["wandb_log"] = True

    else:
        (
            log_dir,
            log_file,
            logger,
            model_save_file,
        ) = define_logs((args.id if args.id else start_time), args.log_dir, args.log)

    conf["env_args"]["reward_success"] = args.reward_success
    conf["env_args"]["reward_failure"] = args.reward_failure
    conf["env_args"]["reject_action"] = args.reject_action
    conf["env_args"]["use_afterstate"] = args.use_afterstate
    # Setup environment
    env = [
        make_env(conf["env_name"], seed=i, **conf["env_args"])
        for i in range(args.n_procs)
    ]
    env = (
        SubprocVecEnv(env, start_method="fork")
        if args.multithread
        else DummyVecEnv(env)
    )

    # Define callbacks
    if args.save_model:

        callbacks.append(
            CustomCallback(
                env=env,
                data_file=args.output_file,
                model_file=model_save_file,
                save_model=args.save_model,
                save_episode_info=args.save_episode_info,
            )
        )

    callback_list = CallbackList(callbacks)

    # Create agent
    agent_kwargs = dict(
        verbose=0,
        device="cuda",
        gamma=args.gamma,
        learning_rate=choose_schedule(args.lr_schedule, args.learning_rate),
        gae_lambda=args.gae_lambda,
        n_steps=args.n_steps,
        batch_size=args.batch_size if args.batch_size else args.n_steps,
        clip_range=choose_schedule(args.clip_range_schedule, args.clip_range),
        clip_range_vf=choose_schedule(args.clip_range_vf_schedule, args.clip_range_vf),
        n_epochs=args.n_epochs,
        ent_coef=args.ent_coef,
        use_afterstate=args.use_afterstate,
    )
    if args.multistep_masking:
        agent_kwargs.update(
            multistep_masking=args.multistep_masking,
            multistep_masking_attr=args.multistep_masking_attr,
            multistep_masking_n_steps=args.multistep_masking_n_steps,
            action_interpreter=args.action_interpreter,
        )
    agent_args = ("MultiInputPolicy", env)

    model = (
        MaskablePPO(*agent_args, **agent_kwargs)
        if args.masking
        else PPO(*agent_args, **agent_kwargs)
    )

    # If retraining
    if args.model_file:
        logger.warn(f"Loading model for retraining: {args.model_file}")
        model.set_parameters(args.model_file)
        model.set_env(env)

    elif args.artifact:
        logger.warn(f"Loading model for retraining: {args.artifact}")
        artifact = run.use_artifact(args.artifact, type='model')
        artifact_file = artifact.download(root=log_dir.resolve()) / f"{artifact.name.split(':')[0]}.zip"
        logger.warn(f"Artifact file: {artifact_file.resolve()}")
        model.set_parameters(artifact_file)
        model.set_env(env)

    model.learn(
        total_timesteps=args.total_timesteps, callback=callback_list
    )

    if args.save_model and not args.no_wandb:

        art = wandb.Artifact(model_save_file.stem, type="model")
        art.add_file(model_save_file.resolve())
        wandb.log_artifact(art)

    env.close()

    if not args.no_wandb:
        run.finish()
