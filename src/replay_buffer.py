import numpy as np
from typing import List, Dict
import torch.utils.data


class ReplayBuffer(torch.utils.data.Dataset):
    def __init__(self, frame_shape: List[int], count: int, discount_factor: float, stack_frames: int = 4):
        self.frame_buffer = np.empty([count]+list(frame_shape), dtype=np.uint8)
        self.action_buffer = np.empty([count], dtype=np.uint8)
        self.reward_buffer = np.empty([count], dtype=np.float32)
        self.is_done = np.empty([count], dtype=np.uint8)
        self.count = 0
        self.write_to = 0
        self.capacity = count
        self.discount_factor = discount_factor
        self.stack_frames = stack_frames

    def add(self, frame: np.ndarray, action: int, reward: float, is_done: bool):
        if not self.full:
            self.count += 1

        self.frame_buffer[self.write_to] = frame
        self.action_buffer[self.write_to] = action
        self.reward_buffer[self.write_to] = reward
        self.is_done[self.write_to] = is_done

        self.write_to = (self.write_to + 1) % self.capacity

    @property
    def full(self) -> bool:
        return self.count == self.capacity

    def get_frames(self, start: int, count: int) -> np.ndarray:
        end = start + count
        if end <= self.capacity:
            return self.frame_buffer[start:end]
        else:
            return np.take(self.frame_buffer, np.arange(start, end), axis=0, mode='wrap')

    def get(self, start: int):
        if self.full:
            start = (start + self.write_to) % self.capacity

        step = (start + self.stack_frames - 1) % self.capacity
        frames = self.get_frames(start, self.stack_frames+1)

        # If it is the start of the episode, replicate the first frame multiple time.
        for start_index in range(self.stack_frames-2, -1, -1):
            if self.is_done[(start + start_index) % self.capacity]:
                frames = np.copy(frames)  # don't corrupt the buffer
                frames[:start_index+1] = frames[start_index+1]
                break

        return frames, self.reward_buffer[step], self.action_buffer[step], self.is_done[step]

    def __len__(self):
        return 999999999999999

    @property
    def valid_range(self) -> int:
        return self.count - self.stack_frames - 1

    def __getitem__(self, item: int) -> Dict[str, np.ndarray]:
        item = np.random.randint(0, self.valid_range)

        frames, reward, action, is_done = self.get(item)
        return {
            "frames": frames,
            "reward": np.asfarray([reward], dtype=np.float32),
            "action": np.asarray([action], dtype=np.uint8),
            "is_done": np.asarray([is_done], dtype=np.uint8)
        }