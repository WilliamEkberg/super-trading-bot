import numpy as np
from tqdm import tqdm

from utils.utils import get_state



class Train():
    def __init__(self, data, traider, batch_size=32, obs_window=10):
        self.train_profit = 0
        self.len_data = len(data)-1
        self.batch_size = batch_size
        self.obs_window = obs_window
        self.traider = traider

    
    def go_to_gym(self, number_of_episodes, dataset):
        self.number_episodes = number_of_episodes
        self.dataset = dataset
        for ep in self.number_episodes:
            self.train(ep)

    def train(self, episode):
        self.traider.inventory = []
        loss_list = []

        current_state = get_state(self.dataset, 0, self.obs_window+1) #the whole dataset????????

        for i in tqdm(range(self.len_data), i+1, total=self.len_data, desc=f"Episode: {episode} out of {self.number_episodes}"):
            profit = 0
            next_state = get_state(self.dataset, i+1, self.obs_window)

            current_action = self.traider.act(current_state)

            if current_action == 1:
                self.traider.inventory.append(self.dataset[i])

            elif current_action == 2 and len(self.traider.inventory) >0:
                bought_value = self.traider.inventory.pop(0)
                current_value = self.dataset[i]
                profit = current_value - bought_value
                self.train_profit += profit

            bool_end = (i == self.len_data-1)
            self.traider.remember(current_state, current_action, profit, next_state, bool_end)

            if len(self.traider.memory) > self.batch_size:
                replay_loss = self.traider.train_experience_replay(self.batch_size)
                if replay_loss != None:
                    loss_list.append(replay_loss)

            current_state = next_state
    
        if episode % 30 ==0:
            self.traider.save(episode)
        return profit, np.mean(loss_list)
    
    def testing(self, dataset):
        self.dataset = dataset
        profit = 0
        len_data = len(self.dataset)-1

        self.traider.inventory = []
        timeline = []

        current_state = get_state(self.dataset, 0, self.obs_window+1)

        for i in range(len_data):
            profit = 0
            next_state = get_state(self.dataset, i+1, self.obs_window+1)
            current_action = self.traider.act(current_state, is_eval=True)

            if current_action == 1:
                self.traider.inventory.append(self.dataset[i])
                timeline.append((dataset[i], "Buying"))
        
            elif current_action==2 and len(self.traider.inventory) >0:
                bought_value = self.traider.inventory.pop(0)
                current_value = self.dataset[i]
                profit = current_value - bought_value
                self.train_profit += profit
                timeline.append((dataset[i], "Selling"))

            else:
                timeline.append((dataset[i], "HODL"))
            
            bool_end = (i == self.len_data-1)
            self.traider.remember(current_state, current_action, profit, next_state, bool_end)
            current_state = next_state

            if bool_end:
                break
        return profit, timeline
