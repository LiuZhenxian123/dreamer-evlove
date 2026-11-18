import atexit
import pathlib
import sys
import warnings

import hydra
import torch

import tools
from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs
from sea.runtime import SeaRuntime
from trainer import OnlineTrainer

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    # Mirror stdout/stderr to a file under logdir while keeping console output.
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    logger = tools.Logger(logdir)
    # save config
    logger.log_hydra_config(config)

    replay_buffer = Buffer(config.buffer)

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space, env_info = make_envs(config.env)
    priv_spec, task_names = env_info["priv_spec"], env_info["tasks"]
    if (config.sea.enabled or config.sea.dump_data_at) and priv_spec is None:
        raise ValueError(f"env {config.env.task} does not provide privileged state; sea needs metaworld or dmc")

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
        priv_spec=priv_spec,
        n_tasks=len(task_names),
    ).to(config.device)

    sea = None
    if priv_spec is not None:
        sea = SeaRuntime(config.sea, priv_spec, task_names, int(config.env.steps), logdir, logger, seed=config.seed)
        sea.start(agent)

    policy_trainer = OnlineTrainer(
        config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs, sea=sea, task_names=task_names
    )
    policy_trainer.begin(agent)
    policy_trainer.save(agent)


if __name__ == "__main__":
    main()
