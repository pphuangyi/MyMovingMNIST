"""
the moving_mnist dataset contains 10,000 video of hand-writen digits.
Each video has 20 frames of size 64 x 64.
The npy file is of shape (20, 10000, 64, 64).
"""

from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

ROOT = '../../data/moving_mnist'

class MovingMNIST(Dataset):
    """
    Moving MNIST dataset API
    """
    def __init__(self,
                 path         = ROOT,
                 split        = 'train',
                 seq_len      = 10,
                 horizon      = 1,
                 return_label = False):

        super().__init__()
        assert seq_len + horizon <= 50

        assert split in ('train', 'test')

        self.fnames = sorted(list((Path(path)/split).glob('*npz')),
                             key=lambda x: int(x.stem.split('_')[-1]))

        self.seq_len = seq_len
        self.pred_idx = seq_len + horizon - 1

        self.return_label = return_label

    def __len__(self):
        return len(self.fnames)

    @staticmethod
    def __normalize(data):
        # return 2. * (data / 255. - .5)
        return data / 255.

    def __getitem__(self, index):

        fname = self.fnames[index]

        with np.load(fname) as data:
            video = torch.Tensor(data['video']).to(torch.float32)
            video = video.unsqueeze(1)
            label = torch.Tensor(data['label']).to(torch.float32)

            seq = video[:self.seq_len]
            tgt = video[self.pred_idx]

            seq = self.__normalize(seq)
            tgt = self.__normalize(tgt)

        if self.return_label:
            return seq, tgt, label

        return seq, tgt

def test():
    """
    Test the moving MNIST dataset API
    """
    return_label = True

    dataset = MovingMNIST(return_label=return_label)
    dataloader = DataLoader(dataset, batch_size=4)

    for seq, tgt, label in dataloader:
        print(f'seq shape = {seq.shape}, min={seq.min():.3f}, max={seq.max():.3f}')
        print(f'tgt shape = {tgt.shape}, min={tgt.min():.3f}, max={tgt.max():.3f}')
        print(f'label = {label}')
        break

if __name__ == "__main__":
    test()
