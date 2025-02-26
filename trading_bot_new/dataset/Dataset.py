from torch.utils.data import Dataset
import pandas as pd
import os
import torch
from loguru import logger
import numpy as np
from utils.utils import sigmoid

class TradingDataset(Dataset):
    def __init__(self, data_dir, asset_name, window_size):
        self.data_dir = os.path.join(data_dir,asset_name)
        self.asset_name = asset_name
        self.window_size = window_size
        self.raw_data = self.get_data()
        self.x, self.y = self.pre_process_data()
        
        assert os.path.exists(self.data_dir), f"The path {self.data_dir} does not exist."
        logger.info(f"Create Trading Dataset with {len(self.x)} time entries.")
    
    def get_data(self):
        df = pd.read_csv(self.data_dir, header=None)
        df = df.iloc[:, 1:]  # Select all columns except the first
        rows = df.values.tolist()[1:]  # Convert rows to list format

        for i in range(len(rows)):
            rows[i] = [float(x) for x in rows[i]]
        
        return torch.tensor(rows)

    def get_state(self, data, t, n_days):
        """Returns an n-day state representation ending at time t as a PyTorch tensor.
        
        Args:
            data (list or array-like): Sequence of data points.
            t (int): Current time index.
            n_days (int): Number of days to consider for the state.
            
        Returns:
            torch.Tensor: A tensor of shape (1, n_days - 1) containing sigmoid differences 
                        between consecutive data points. If not enough data points exist, 
                        the state is padded with the first element.
        """
        d = t - n_days + 1
        if d >= 0:
            block = data[d: t + 1]
        else:
            block = [-d * [data[0]] + data[0: t + 1]]
            # Alternatively, you can pad with the first element:
            block = [data[0]] * (-d) + data[0: t + 1]

        res = [sigmoid(block[i + 1] - block[i]) for i in range(n_days - 1)]
        # Return the state as a torch tensor with shape (1, n_days-1)
        return torch.tensor([res], dtype=torch.float)
    
    def pre_process_data(self):
        financial_data = self.raw_data
        x = []
        y = []
        for idx in range(len(financial_data)-1):
            current_state = self.get_state(list(financial_data[:,4]), idx, self.window_size)
            next_state =  current_state = self.get_state(list(financial_data[:,4]), idx+1, self.window_size)
            x.append(current_state)
            y.append(next_state)

        return x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        done = False
        if idx == (self.__len__()-1):
            done = True

        return self.x[idx], self.y[idx], torch.tensor(self.raw_data[:,4][idx]), done
        

if __name__ =="__main__":
    data_dir = "../data/"
    asset_name = "AAPL_2018.csv"
    window_size = 100
    dataset = TradingDataset(data_dir, asset_name, window_size, "train")
