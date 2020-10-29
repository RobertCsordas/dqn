#!/usr/bin/env python3

import gym
import torch
import torch.utils.data
import numpy as np
from src.replay_buffer import ReplayBuffer
from src.network import AtariNet
from copy import deepcopy
import time
import wandb
from typing import Callable, List
import threading
from queue import Queue

wandb.init(project='dqn')

gamma = 0.99
TARGET_SWITCH = 10000
MAXLEN = 10000
PREFILL = 50000
REPLY_BUFFER_SIZE = 1000000
BATCH_SIZE = 32
STEPS_PER_TRAIN = 4
NOOP_MAX = 30


class DQN:
    grayscale_coeffs = np.asarray([0.11, 0.59, 0.3], dtype=np.float32)

    @staticmethod
    def preprocess_frame(frame: np.ndarray) -> np.ndarray:
        return (frame[::2, ::2].astype(np.float32) * DQN.grayscale_coeffs).sum(-1).astype(np.uint8)

    @staticmethod
    def frame_to_nn(frame: torch.Tensor) -> torch.Tensor:
        return frame.float() / 255.0

    def __init__(self, env: str):
        self.device = torch.device("cuda")

        self.env = gym.make(env)
        self.noop_index = self.env.unwrapped.get_action_meanings().index("NOOP")
        self.n_actions = self.env.action_space.n
        self.img_shape = self.preprocess_frame(self.env.reset()).shape

        self.memory = ReplayBuffer(self.img_shape, REPLY_BUFFER_SIZE, discount_factor=gamma)

        self.game_steps = 0
        self.rand_fill()

        self.loader = torch.utils.data.DataLoader(self.memory, batch_size=BATCH_SIZE, pin_memory=True, num_workers=0)
        self.net = AtariNet([4, *self.img_shape], self.n_actions).to(self.device)
        self.loss = torch.nn.SmoothL1Loss().to(self.device)
        # self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=0.00025, eps=0.01, alpha=0.95, centered=True)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)

        self.loss_sum = 0
        self.loss_cnt = 0
        self.last_test_time = time.time()
        self.last_video_time = 0
        self.copy_network()

        self.prefetch_queue = Queue(maxsize=1)
        self.loader_thread = threading.Thread(target = self.loader_thread)
        self.loader_thread.start()

    def loader_thread(self):
        while True:
            for d in self.loader:
                self.prefetch_queue.put({k: v.to(self.device) for k, v in d.items()})

    def play(self, get_action: Callable[[int, List[np.ndarray]], int], train: bool, step_hook = lambda: None, maxlen=MAXLEN):
        total_reward = 0

        observation = self.preprocess_frame(self.env.reset())
        all_frames = [observation]*4

        for t in range(maxlen):
            action = get_action(t, all_frames[-4:])
            new_frame, reward, done, info = self.env.step(action)
            if train:
                self.memory.add(observation, action, reward, done)
                self.game_steps += 1

            observation = self.preprocess_frame(new_frame)
            all_frames.append(observation)
            total_reward += reward

            step_hook()
            if done:
                break

        return total_reward, all_frames

    def render_video(self, all_frames: List[np.ndarray]) -> np.ndarray:
        return np.stack(all_frames, axis=0)[:, np.newaxis]

    def rand_fill(self):
        print("Filling the replay buffer with random data")
        while self.memory.count < PREFILL:
            print("Starting new episode. Data so far:", self.memory.count)
            _, frames = self.play(lambda i, observation: self.env.action_space.sample(), True)
        self.game_steps = 0
        print("Prefill completed.")

    def log_loss(self, loss: float):
        self.loss_sum += loss
        self.loss_cnt += 1
        if self.loss_cnt == 100:
            wandb.log({"loss": self.loss_sum / self.loss_cnt}, step=self.game_steps)
            self.loss_sum = 0
            self.loss_cnt = 0

    def copy_network(self):
        self.target_init_step = self.game_steps
        self.predictor = deepcopy(self.net)
        self.predictor.eval()

    def train_step(self):
        data = self.prefetch_queue.get()

        action = data["action"].long()
        frames = self.frame_to_nn(data["frames"])

        pred = self.net(frames[:,:-1])
        pred = pred.gather(index=action, dim=1)
        with torch.no_grad():
            # Double DQN update: choose best action to bootstrap against from the predictor model
            next_value = self.net(frames[:, 1:])
            _, next_value_target_index = self.predictor(frames[:, 1:]).max(-1, keepdim=True)
            next_value = torch.gather(next_value, 1, next_value_target_index)

        target = gamma*next_value*(1.0-data["is_done"].float()) + data["reward"]
        l = self.loss(pred, target)

        self.optimizer.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()

        self.log_loss(l.item())
        if self.game_steps - self.target_init_step > TARGET_SWITCH*STEPS_PER_TRAIN:
            self.copy_network()

    def get_epsilon(self) -> float:
        e_start = 1.0
        e_end = 0.1
        n = 1000000.0

        return max(e_start - (e_start - e_end)/n * self.game_steps, e_end)

    def get_action(self, iteration: int, observations: List[np.ndarray], train: bool=True) -> int:
        epsilon = self.get_epsilon()
        if train and iteration==0:
            self.n_noop = np.random.randint(0, NOOP_MAX+1)

        if train and iteration < self.n_noop:
            return self.noop_index
        elif train and (np.random.random() < epsilon):
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                observation = np.stack(observations, axis=0)
                input = self.frame_to_nn(torch.tensor(observation, device=self.device).unsqueeze(0))
                pred = self.net(input)
                _, amax= pred[0].max(-1)
                return amax.item()

    def train(self):
        while True:
            def do_train():
                if self.game_steps % STEPS_PER_TRAIN == 0:
                    self.train_step()

            log = {}
            log["epsilon"] = self.get_epsilon()
            log["train_reward"], frames = self.play(self.get_action, train=True, step_hook=do_train)
            print(f"Step {self.game_steps}: Episode completed in {len(frames)} steps. Reward: {log['train_reward']}. Epsilon: {log['epsilon']}")
            frames = None

            now = time.time()
            if now - self.last_test_time > 60:
                log["test_reward"], frames = self.play(lambda i, observation: self.get_action(i, observation, train=False), train=False)
                self.last_test_time = now
                print(f"--> TEST: Step {self.game_steps}: Episode completed in {len(frames)} stpes. Reward: {log['test_reward']}")

                if now - self.last_video_time > 10*60:
                    log["video"] = wandb.Video(self.render_video(frames[-300:]), fps=10)
                    self.last_video_time = now

                frames = None

            wandb.log(log, step=self.game_steps)

DQN('BreakoutDeterministic-v4').train()