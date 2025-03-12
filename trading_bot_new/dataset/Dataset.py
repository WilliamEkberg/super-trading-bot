from torch.utils.data import Dataset
import pandas as pd
import os
import torch
from loguru import logger
import numpy as np
from utils.utils import sigmoid
from torch.utils.data import DataLoader

class TradingDataset(Dataset):
    def __init__(self, data_dir, asset_name, window_size, type="train"):
        self.data_dir = os.path.join(data_dir,asset_name)
        self.asset_name = asset_name
        self.window_size = window_size
        self.type = type
        self.raw_data = self.get_data()
        self.x, self.y = self.pre_process_data()
        
        assert os.path.exists(self.data_dir), f"The path {self.data_dir} does not exist."
        logger.info(f"Create Trading Dataset with {len(self.x)} time entries.")
    
    def get_data(self):
        if self.type == "train":
            dataset_path = os.path.join(self.data_dir, "combined_train_data.csv")
        else:
            dataset_path = os.path.join(self.data_dir, "combined_test_data.csv")
        df = pd.read_csv(dataset_path, sep=";")
        #df = pd.read_csv(dataset_path)
        df['Rate'] = df['Rate'].str.replace(',', '.').astype(float)
        df['OMX_Close'] = df['OMX_Close'].str.replace(',', '').astype(float)
        df['Index_Value'] = df['Index_Value'].str.replace(',', '.').astype(float)

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
            current_state = self.get_state(list(financial_data[:,2]), idx, self.window_size)
            next_state = self.get_state(list(financial_data[:,2]), idx+1, self.window_size)
            x.append(current_state)
            y.append(next_state)

        return x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        done = False
        if idx == (self.__len__()-1):
            done = True

        return self.x[idx], self.y[idx], self.raw_data[:,2][idx], done
    
class TradingDataset_V2(Dataset):
    def __init__(self, data_dir, asset_name, window_size):
        self.data_dir = os.path.join(data_dir,asset_name)

        self.asset_name = asset_name
        self.window_size = window_size
        self.data = self.get_data()
        
        assert os.path.exists(self.data_dir), f"The path {self.data_dir} does not exist."
        logger.info(f"Create Trading Dataset with {len(self.data)} time entries.")
    
    def get_data(self):
        df = pd.read_csv(self.data_dir, header=None)
        df = df.iloc[:, 1:]  # Select all columns except the first
        rows = df.values.tolist()[1:]  # Convert rows to list format

        for i in range(len(rows)):
            rows[i] = [float(x) for x in rows[i]]
        
        return torch.tensor(rows)

    def __len__(self):
        return len(self.data)-1

    def __getitem__(self, idx):
        done = False
        if idx == (self.__len__()):
            done = True
        
        d = idx - self.window_size
        if d >= 0:
            x = self.data[0:-1,:][d:idx,:]
            y = self.data[0:-1,:][d+1:idx+1,:]
        else:
            x = torch.vstack([self.data[0,:].repeat([-d,1]), self.data[0:-1,:][0: idx,:]])
            y = torch.vstack([self.data[0,:].repeat([-(d+1),1]), self.data[0:-1,:][0: idx+1,:]])
   
        return x.float(), y.float(), self.data[0:-1,:][:,4][idx].float(), done
    

    
class TradingDataset_V3(Dataset):
    def __init__(self, data_dir, asset_name, window_size, type="train"):
        self.data_dir = os.path.join(data_dir, asset_name)
        self.asset_name = asset_name
        self.window_size = window_size
        self.type = type
        
        assert os.path.exists(self.data_dir), f"The path {self.data_dir} does not exist."
        self.data = self.get_data()
        logger.info(f"Created Trading Dataset with {len(self.data)} time entries.")

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
    
    def get_observation(self, t):
        """Return the current state observation with feature window."""
        k = t - self.window_size
        if k < 0:   # Pad in the beginning for negative t
            features = torch.vstack([self.data[0] for i in range(-k)])
            features = torch.vstack([features, self.data[0:t,:]])
        else:
            features = self.data[t-self.window_size:t, :]
        feature_mean = torch.mean(features)
        feature_std = torch.std(features) + 1e-10  # Avoid division by zero
        normalized_features = (features - feature_mean) / feature_std
        normalized_features = features
        # Flatten the feature matrix to 1D array
        features_flat = normalized_features.flatten().unsqueeze(0)
        state_index = self.get_state(list(self.data[:,2]), t, self.window_size)
        state_omx = self.get_state(list(self.data[:,1]), t, self.window_size)
        return torch.hstack([state_index, state_omx])
        #return torch.hstack([features_flat, state_index])

    def get_data(self):
        if self.type == "train":
            dataset_path = os.path.join(self.data_dir, "combined_train_data.csv")
        else:
            dataset_path = os.path.join(self.data_dir, "combined_test_data.csv")

        df = pd.read_csv(dataset_path, sep=";")
        df['Rate'] = df['Rate'].str.replace(',', '.').astype(float)
        df['OMX_Close'] = df['OMX_Close'].str.replace(',', '').astype(float)
        df['Index_Value'] = df['Index_Value'].str.replace(',', '.').astype(float)

        #rate_features = self.compute_indicators(df, 'Rate')
        #OMX_features = self.compute_indicators(df, 'OMX_Close')
        index_features = self.compute_indicators(df, 'Index_Value')

        return index_features
        #return torch.hstack([rate_features, OMX_features, index_features])
    
    def compute_indicators(self, df, category):
        # Simple Moving Average (SMA)
        df['sma_10'] = df[category].rolling(window=10).mean()
        df['sma_30'] = df[category].rolling(window=30).mean()
        
        # Relative Strength Index (RSI)
        delta = df[category].diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = -delta.clip(upper=0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['ema_12'] = df[category].ewm(span=12).mean()
        df['ema_26'] = df[category].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Forward fill NaN values resulting from rolling windows
        df = df.fillna(method='ffill').fillna(method='bfill')

        return torch.from_numpy(df.values[:,1:].astype(np.float32))

    def __len__(self):
        return len(self.data)-1

    def __getitem__(self, idx):
        done = False
        if idx == (self.__len__()):
            done = True
        
        x = self.get_observation(t=idx)
        y = self.get_observation(t=idx+1)

        current_price = self.data[idx,2]

        return x, y, current_price, done
        
if __name__ =="__main__":
    data_dir = "../data/"
    asset_name = "novotech"
    window_size = 10
    dataset = TradingDataset_V3(data_dir, asset_name, window_size, type="train")
    dataloader_train = DataLoader(dataset, batch_size=1, shuffle=False)
    for batch in dataloader_train:
         print(batch[0])