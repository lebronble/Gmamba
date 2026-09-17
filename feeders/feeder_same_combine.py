import pickle

import numpy as np
from torch.utils.data import Dataset

from feeders import tools


class Feeder(Dataset):
    """Emotion-Gait dual-stream loader.

    Returns motion, pose, label, affective feature, index to stay compatible
    with the original MSH-GT/gait_mamba training style.
    """

    def __init__(
        self,
        data_m_path,
        data_p_path,
        label_path,
        feature_path,
        random_choose=False,
        random_shift=False,
        random_move=False,
        window_size=-1,
        normalization=False,
        debug=False,
        use_mmap=True,
    ):
        self.data_m_path = data_m_path
        self.data_p_path = data_p_path
        self.label_path = label_path
        self.feature_path = feature_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = int(window_size)
        self.normalization = normalization
        self.debug = debug
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.mean_m = self.data_m.mean(axis=0)
            self.std_m = self.data_m.std(axis=0) + 1e-6
            self.mean_p = self.data_p.mean(axis=0)
            self.std_p = self.data_p.std(axis=0) + 1e-6

    def load_data(self):
        with open(self.label_path, "rb") as f:
            names, labels = pickle.load(f, encoding="latin1")
        mmap_mode = "r" if self.use_mmap else None
        self.sample_name = names
        self.label = np.asarray(labels, dtype=np.int64)
        self.data_m = np.load(self.data_m_path, mmap_mode=mmap_mode)
        self.data_p = np.load(self.data_p_path, mmap_mode=mmap_mode)
        self.feature = np.load(self.feature_path, mmap_mode=mmap_mode)
        if self.debug:
            self.sample_name = self.sample_name[:100]
            self.label = self.label[:100]
            self.data_m = self.data_m[:100]
            self.data_p = self.data_p[:100]
            self.feature = self.feature[:100]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_m = np.asarray(self.data_m[index], dtype=np.float32).copy()
        data_p = np.asarray(self.data_p[index], dtype=np.float32).copy()
        feature = np.asarray(self.feature[index], dtype=np.float32).copy()
        label = int(self.label[index])

        if self.normalization:
            data_m = (data_m - self.mean_m) / self.std_m
            data_p = (data_p - self.mean_p) / self.std_p
        if self.random_shift:
            data_m = tools.random_shift(data_m)
            data_p = tools.random_shift(data_p)
        if self.random_choose:
            data_m = tools.random_choose(data_m, self.window_size)
            data_p = tools.random_choose(data_p, self.window_size)
        elif self.window_size > 0:
            data_m = tools.auto_pading(data_m, self.window_size)
            data_p = tools.auto_pading(data_p, self.window_size)
        if self.random_move:
            data_p = tools.random_move(data_p)

        return data_m, data_p, label, feature, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit = [label in rank[i, -top_k:] for i, label in enumerate(self.label)]
        return sum(hit) / len(hit)
