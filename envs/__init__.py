import atexit

from . import parallel, wrappers


def make_envs(config):
    suite = config.task.split("_", 1)[0]

    if suite == "isaaclab":
        return _make_isaaclab_envs(config)

    def env_constructor(evaluation):
        def constructor(idx):
            return lambda: make_env(config, idx, evaluation)

        return constructor

    train_envs = parallel.ParallelEnv(env_constructor(False), config.env_num, config.device)
    eval_envs = (
        parallel.ParallelEnv(env_constructor(True), config.eval_episode_num, config.device)
        if config.eval_episode_num > 0
        else None
    )
    obs_space = train_envs.observation_space
    act_space = train_envs.action_space
    return train_envs, eval_envs, obs_space, act_space, env_info(config, train_envs)


def task_list(config):
    """Full task names. `env.task` may join several tasks of one suite with '+': 'metaworld_reach+push'."""
    suite, names = config.task.split("_", 1)
    return [f"{suite}_{name}" for name in names.split("+")]


def env_info(config, envs):
    """Privileged-state layout (None if the suite has none) and the task names."""
    from sea.privileged import PrivSpec

    suite = config.task.split("_", 1)[0]
    spec = None
    if suite in ("metaworld", "dmc"):
        spec = PrivSpec.from_json(envs.envs[0].priv_spec()())
    return {"priv_spec": spec, "tasks": task_list(config)}


def _make_isaaclab_envs(config):
    if int(config.eval_episode_num) > 0:
        raise ValueError(
            "IsaacLab environments do not support separate eval envs yet. Set eval_episode_num: 0 in the env config."
        )

    # AppLauncher must be created before any isaaclab.* module is imported.
    from isaaclab.app import AppLauncher

    headless = getattr(config, "headless", True)
    launcher = AppLauncher(headless=headless, enable_cameras=True)
    sim_app = launcher.app

    from envs.isaaclab import IsaacLabVecEnv, create_isaaclab_env

    raw_env = create_isaaclab_env(config)

    train_envs = IsaacLabVecEnv(raw_env, simulation_app=sim_app, image_size=tuple(config.size))

    # Ensure the SimulationApp is shut down cleanly on process exit.
    atexit.register(train_envs.close)

    eval_envs = None  # Not supported yet for IsaacLab.

    obs_space = train_envs.observation_space
    act_space = train_envs.action_space
    return train_envs, eval_envs, obs_space, act_space, {"priv_spec": None, "tasks": [config.task]}


def make_env(config, id, evaluation=False):
    tasks = task_list(config)
    task_index = id % len(tasks)
    suite, task = tasks[task_index].split("_", 1)
    if len(tasks) > 1 and suite != "metaworld":
        raise NotImplementedError("multi-task runs are only supported for Meta-World (shared privileged layout)")
    if suite == "dmc":
        import envs.dmc as dmc

        shift = config.get("eval_shift" if evaluation else "shift", None)
        env = dmc.DeepMindControl(task, config.action_repeat, config.size, seed=config.seed + id, shift=shift)
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.gray,
            noops=config.noops,
            lives=config.lives,
            sticky=config.sticky,
            actions=config.actions,
            length=config.time_limit,
            pooling=config.pooling,
            aggregate=config.aggregate,
            resize=config.resize,
            autostart=config.autostart,
            clip_reward=config.clip_reward,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "metaworld":
        import envs.metaworld as metaworld

        env = metaworld.MetaWorld(
            task,
            config.action_repeat,
            config.size,
            config.camera,
            config.seed + id,
            configs=config.get("eval_configs" if evaluation else "configs", "random"),
            config_seed=config.get("config_seed", 0),
            task_index=task_index,
            num_tasks=len(tasks),
        )
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit // config.action_repeat)
    return wrappers.Dtype(env)
