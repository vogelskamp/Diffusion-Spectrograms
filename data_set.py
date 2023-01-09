from torch.utils.data import Dataset
import os
import numpy as np
from tqdm import tqdm


class SpectrogramSet(Dataset):

    def __init__(self, data_path, transform=None):

        self.data_path = data_path
        self.data = self.load_data(self.data_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass

    def load_data(self, path):

        result = []
        for file in tqdm(os.listdir(path)):

            if not file.endswith("-padded.npy"):
                print(file, "does not end in -padded.npy, ignoring..")
                continue

            with open(path + file, 'rb') as f:
                result.append(np.load(f))

        return result


if __name__ == '__main__':
    test = SpectrogramSet(
        data_path="/Users/samvogelskamp/Desktop/Uni/Audio Data Science/data/")
