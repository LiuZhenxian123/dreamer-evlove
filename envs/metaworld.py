import gymnasium as gym
import numpy as np

from sea.privileged import MetaWorldPriv


class MetaWorld(gym.Env):
    """Pixel Meta-World task that also emits privileged simulator state.

    configs: which object/goal layouts an episode can start from.
        "random"  a fresh layout every episode (the base repository's behaviour)
        "train"   one of the 50 fixed ML1 training layouts
        "test"    one of the 50 held-out ML1 layouts (zero-shot configuration shift)
    task_index/num_tasks: position of this task in a multi-task run; with more than one
        task the observation carries a one-hot "task" vector for the encoder.
    """

    def __init__(
        self,
        name,
        action_repeat=1,
        size=(64, 64),
        camera=None,
        seed=0,
        configs="random",
        config_seed=0,
        task_index=0,
        num_tasks=1,
    ):
        import metaworld

        self._camera = camera
        env_name = name + "-v3"
        if configs == "random":
            bench = metaworld.MT1(env_name, seed=seed)
            self._layouts = None
            env = bench.train_classes[env_name](render_mode="rgb_array", camera_name=self._camera)
            env.set_task(bench.train_tasks[0])
        elif configs in ("train", "test"):
            # config_seed (not the run seed) fixes the layouts, so every run and every
            # method sees the same train/held-out split.
            bench = metaworld.ML1(env_name, seed=config_seed)
            self._layouts = bench.train_tasks if configs == "train" else bench.test_tasks
            env = bench.train_classes[env_name](render_mode="rgb_array", camera_name=self._camera)
            env.set_task(self._layouts[0])
        else:
            raise ValueError(f"configs must be random|train|test, got {configs!r}")

        if self._camera == "corner2":
            env.model.cam_pos[2] = [0.75, 0.075, 0.7]

        self._env = env
        self._env.mujoco_renderer.width = size[1]
        self._env.mujoco_renderer.height = size[0]
        if self._layouts is None:
            self._env._freeze_rand_vec = False
        self._rng = np.random.default_rng(seed)
        self._size = size
        self._action_repeat = action_repeat
        self._task_index = int(task_index)
        self._num_tasks = int(num_tasks)
        self._priv = MetaWorldPriv(env)
        self.reward_range = [-np.inf, np.inf]

    def priv_spec(self):
        return self._priv.spec.to_json()

    @property
    def observation_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "state": self._env.observation_space,
            "priv": gym.spaces.Box(-np.inf, np.inf, (self._priv.spec.dim,), dtype=np.float32),
            "task_id": gym.spaces.Box(0, np.iinfo(np.int32).max, (1,), dtype=np.int32),
            "log_success": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
        }
        if self._num_tasks > 1:
            spaces["task"] = gym.spaces.Box(0, 1, (self._num_tasks,), dtype=np.float32)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        # action range in original environmens is -1 ~ 1
        return gym.spaces.Box(
            self._env.action_space.low,
            self._env.action_space.high,
            dtype=np.float32,
        )

    def _obs(self, state, is_first, is_last, is_terminal, success):
        obs = {
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
            "image": self.render(),
            "state": state,
            "priv": self._priv(self._env, state),
            "task_id": np.array([self._task_index], dtype=np.int32),
            "log_success": success,
        }
        if self._num_tasks > 1:
            onehot = np.zeros(self._num_tasks, dtype=np.float32)
            onehot[self._task_index] = 1.0
            obs["task"] = onehot
        return obs

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0.0
        success = 0.0
        for _ in range(self._action_repeat):
            state, rew, terminated, truncated, info = self._env.step(action)
            success += float(info["success"])
            reward += rew
            if terminated or truncated:
                break
        # make sure success is always 0 or 1
        success = bool(min(success, 1.0))
        # meta-world doesn't reset env after success in training mode
        is_last = terminated or truncated
        # success is not terminate, to allow bootstrapping
        obs = self._obs(state, False, is_last, terminated, success)
        return obs, reward, is_last, {}

    def reset(self, **kwargs):
        if self._layouts is not None:
            self._env.set_task(self._layouts[int(self._rng.integers(len(self._layouts)))])
        state, _ = self._env.reset()
        return self._obs(state, True, False, False, False)

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        if self._camera == "corner2":
            return np.flip(self._env.render(), axis=0)

        return self._env.render()
