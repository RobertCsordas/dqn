import gym
import numpy as np


class DeepmindHackWrapper(gym.Wrapper):
    def __init__(self, env, max_noop: int, episodic_life = True):
        super().__init__(env)
        self.noop_index = self.env.unwrapped.get_action_meanings().index("NOOP")
        self.fire_index = self.env.unwrapped.get_action_meanings().index("FIRE") \
                          if "FIRE" in self.env.unwrapped.get_action_meanings() else None
        self.episodic_life = episodic_life
        self.was_done = True
        self.max_noop = max_noop

    def do_step(self, action):
        new_frame, reward, done, info = self.env.step(action)
        self.was_done = done
        return new_frame, reward, done, info

    def reset(self):
        if self.was_done:
            frame = self.env.reset()

        for _ in range(np.random.randint(0, self.max_noop + 1) if self.was_done else 1):
            frame, _, done, _ = self.do_step(self.noop_index)
            if done:
                return self.reset()

        if self.fire_index is not None:
            frame, _, done, _ = self.do_step(self.fire_index)
            if done:
                return self.reset()

        if self.episodic_life:
            self.lives = self.env.unwrapped.ale.lives()

        return frame

    def step(self, ac):
        new_frame, reward, done, info = self.do_step(ac)

        if self.episodic_life:
            lives = self.env.unwrapped.ale.lives()
            if 0 < lives < self.lives:
                self.lives = lives
                done = True

        return new_frame, reward, done, info
