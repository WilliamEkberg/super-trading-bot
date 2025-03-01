import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

class Trainer():
    def __init__(self, dataloader_train, dataloader_val, trader, batch_size=32, obs_window=10):
        self.train_profit = 0
        self.batch_size = batch_size
        self.obs_window = obs_window
        self.trader = trader

    
    def go_to_gym(self, number_of_episodes, dataset):
        self.number_episodes = number_of_episodes
        self.dataset = dataset
        for ep in self.number_episodes:
            self.train(ep)

    def train(self, episode):
        self.trader.inventory = []
        loss_list = []
        dataset = self.dataset
        dataloader = DataLoader(dataset, batch_size = 1, shuffle=False)
        for (current_state, next_state, value, done) in tqdm(dataloader):
            profit = 0
            
            

            current_action = self.trader.act(current_state)

            if current_action == 1: #Buy
                self.trader.inventory.append(value)

            elif current_action == 2 and len(self.trader.inventory) >0: #Sell
                bought_value = self.trader.inventory.pop(0)
                current_value = value
                profit = current_value - bought_value
                self.train_profit += profit
                


            self.trader.remember(current_state, current_action, profit, next_state, done)

            if len(self.trader.memory) > self.batch_size:
                replay_loss = self.trader.train_experience_replay(self.batch_size)
                if replay_loss != None:
                    loss_list.append(replay_loss)

            current_state = next_state
        print(f'profit: {profit}')
        print(f'self.train_profit: {self.train_profit}')
        print(f'np.mean(loss_list): {np.mean(loss_list)}')
        if episode % 30 ==0:
            self.trader.save(episode)
        return profit, np.mean(loss_list)
    
    def testing(self, dataset):
        profit = 0

        self.trader.inventory = []
        timeline = []

        dataloader = DataLoader(dataset, batch_size = 1, shuffle=False)
        for (current_state, next_state, value, done) in dataloader:
            profit = 0

            current_action = self.trader.act(current_state, is_eval=True)

            if current_action == 1:
                self.trader.inventory.append(value)
                timeline.append((value, "Buying"))
                print("Buying")
        
            elif current_action==2 and len(self.trader.inventory) >0:
                bought_value = self.trader.inventory.pop(0)
                current_value = value
                profit = current_value - bought_value
                self.train_profit += profit
                timeline.append((value, "Selling"))
                print("Selling")
            else:
                timeline.append((value, "HODL"))
                print("HODL")
            self.trader.remember(current_state, current_action, profit, next_state, done)
            current_state = next_state

            if done:
                break
        return profit, timeline
