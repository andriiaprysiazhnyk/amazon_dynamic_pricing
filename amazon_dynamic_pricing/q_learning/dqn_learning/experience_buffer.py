import torch


class ExperienceBuffer:
    def __init__(self, features_dimension, buffer_size, batch_size):
        self.features_dimension = features_dimension
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.x = torch.empty(self.buffer_size, self.features_dimension, dtype=torch.float32)
        self.actions = torch.empty(self.buffer_size, 1, dtype=torch.long)
        self.y = torch.empty(self.buffer_size, dtype=torch.float32)
        self.current_samples = 0

    def add_sample(self, x, action, y):
        if self.current_samples < self.buffer_size:
            index = self.current_samples
            self.current_samples += 1
        else:
            index = torch.randint(high=self.buffer_size, size=(1,)).item()

        self.x[index] = x
        self.actions[index, 0] = action
        self.y[index] = y

    def sample(self):
        indexes = torch.randint(high=self.current_samples, size=(self.batch_size,))
        return self.x[indexes], self.actions[indexes], self.y[indexes]
