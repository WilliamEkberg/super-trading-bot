from torch.utils.data import Dataset
import pandas as pd
import os
import torch
from loguru import logger
import numpy as np
from utils.utils import sigmoid

class TradingDataset(Dataset):
    def __init__(self, data_dir, asset_name, window_size, mode):
        self.data_dir = os.path.join(data_dir,asset_name)
        self.mode = mode
        self.asset_name = asset_name
        self.data = self.get_data()
        self.window_size = window_size
        assert os.path.exists(self.data_dir), f"The path {self.data_dir} does not exist."
        logger.info(f"Create Trading Dataset with {len(self.data)} time entries.")
    
    def get_data(self):
        df = pd.read_csv(self.data_dir, header=None)
        df = df.iloc[:, 1:]  # Select all columns except the first
        rows = df.values.tolist()[1:]  # Convert rows to list format

        return rows

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Do the processing of the entire dataset upon intitialization, and return the window chunks here in this funciton.
        data_point = self.data[idx]
        data_point = torch.tensor([float(x) for x in data_point])
        transformed_adj_close = self.get_state(data_point[:,4], idx, self.window_size)
        self.data_point[:,4] = transformed_adj_close

        return data_point
        

if __name__ =="__main__":
    data_dir = "../data/"
    asset_name = "AAPL_2018.csv"
    window_size = 100
    dataset = TradingDataset(data_dir, asset_name, window_size, "train")
    
    print(dataset[1])