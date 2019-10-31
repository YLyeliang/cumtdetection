import numpy as np


class RepeatDataset(object):

    def __init__(self, dataset, times):
        self.dataset = dataset      # a dataset class
        self.times = times
        self.CLASSES = dataset.CLASSES  # a tuple containing classes
        if hasattr(self.dataset, 'flag'):
            self.flag = np.tile(self.dataset.flag, times)

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        return self.times * self._ori_len
