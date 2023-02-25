import numpy as np
import pickle
import torch
import random
from tqdm import tqdm


class DataSet(torch.utils.data.Dataset):

    def __init__(self,
                 data_path: str,
                 label_path: str):
        self.data_path = data_path
        self.label_path = label_path
        self.load_data()

    def load_data(self):

        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
            # self.label = pickle.load(f)

        self.data = np.load(self.data_path)
        # N C T V M
        N, C, T, V, M = self.data.shape
        self.size = N

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple:
        data = np.array(self.data[index])
        label = self.label[index]
        
        return data, label
    
class Feeder_semi(torch.utils.data.Dataset):

    def __init__(self, data_path, label_path, label_percent=0.1):
        self.data_path = data_path
        self.label_path = label_path
        self.label_percent = label_percent

        self.load_data()

    def load_data(self):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        self.data = np.load(self.data_path)

        n = len(self.label)
        # Record each class sample id
        class_blance = {}
        for i in range(n):
            if self.label[i] not in class_blance:
                class_blance[self.label[i]] = [i]
            else:
                class_blance[self.label[i]] += [i]

        final_choise = []
        for c in class_blance:
            c_num = len(class_blance[c])
            choise = random.sample(class_blance[c], round(self.label_percent * c_num))
            final_choise += choise
        final_choise.sort()

        self.data = self.data[final_choise]
        new_sample_name = []
        new_label = []
        for i in final_choise:
            new_sample_name.append(self.sample_name[i])
            new_label.append(self.label[i])

        self.sample_name = new_sample_name
        self.label = new_label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data = np.array(self.data[index])
        label = self.label[index]
        
        return data, label
