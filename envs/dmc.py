import gymnasium as gym
import numpy as np

from sea.privileged import DMCPriv


class DeepMindControl(gym.Env):
    metadata = {}

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, seed=0, shift=None):
        if name.endswith("_subtle"):
            is_subtle = True
        else:
            is_subtle = False

        if "sparse" in name or "finger_turn" in name:
            _name, difficulty = name.rsplit("_", 1)
            domain, task = _name.rsplit("_", 1)
            task = task + "_" + difficulty
        else:
            domain, task = name.rsplit("_", 1)

        if is_subtle:
            from envs import dmc_subtle

            func = getattr(dmc_subtle, name)
            self._env = func(random=seed)
        elif isinstance(domain, str):
            from dm_control import suite

            self._env = suite.load(
                domain,
                task,
                task_kwargs={"random": seed},
            )
        else:
            assert task is None
            self._env = domain()

        # Physical-dynamics shift: multiplicative factors on actuator strength and contact
        # friction (1.0 = the default simulator). Used for zero-shot evaluation.
        shift = dict(shift or {})
        unknown = set(shift) - {"motor", "friction"}
        if unknown:
            raise ValueError(f"unknown dynamics shift keys: {sorted(unknown)}")
        model = self._env.physics.model
        model.actuator_gear[:, 0] *= float(shift.get("motor", 1.0))
        model.geom_friction[:, 0] *= float(shift.get("friction", 1.0))
        self._priv = DMCPriv(self._env.physics)

        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = dict(quadruped=2, fish=3).get(domain, 0)
        self._camera = camera
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            if len(value.shape) == 0:
                shape = (1,)
            else:
                shape = value.shape
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        spaces["priv"] = gym.spaces.Box(-np.inf, np.inf, (self._priv.spec.dim,), dtype=np.float32)
        spaces["task_id"] = gym.spaces.Box(0, np.iinfo(np.int32).max, (1,), dtype=np.int32)
        return gym.spaces.Dict(spaces)

    def priv_spec(self):
        return self._priv.spec.to_json()

    def _extras(self, obs):
        obs["priv"] = self._priv(self._env.physics)
        obs["task_id"] = np.array([0], dtype=np.int32)
        return obs

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if time_step.last():
                break
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render()
        obs = self._extras(obs)
        # There is no terminal state in DMC
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        obs["is_last"] = time_step.last()
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self, **kwargs):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render()
        obs = self._extras(obs)
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        obs["is_last"] = time_step.last()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)
