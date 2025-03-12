import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

class Trainer():
    def __init__(self, dataloader_train, dataloader_val, trader, batch_size=32, obs_window=10, mdp=""):
        self.total_profit = 0
        self.train_profit = 0
        self.batch_size = batch_size
        self.obs_window = obs_window
        self.trader = trader
        self.mdp = mdp

    
    def go_to_gym(self, number_of_episodes, dataset):
        self.number_episodes = number_of_episodes
        self.dataset = dataset
        for ep in self.number_episodes:
            self.train_profit =0
            self.train(ep)

    def train(self, episode):
        self.money = 10000
        self.trader.inventory = []
        loss_list = []
        dataset = self.dataset
        dataloader = DataLoader(dataset, batch_size = 1, shuffle=False)
        old_value = None
        for (current_state, next_state, value, done) in tqdm(dataloader):
            profit = 0
            if old_value == None: old_value=value
            change = len(self.trader.inventory)*(value-old_value)
            current_action = self.trader.act(current_state)

            if self.mdp == "all_or_nothing":
                current_action_percentage = current_action
                change = np.tan(change/150)
            elif self.mdp == "all_10%_steps":
                current_action_percentage = current_action/10
                change = -np.exp(-change/150) + 1
                change = np.tanh(change*2)
            else: raise ValueError("Wow, this is the wrong file!")
            #change = np.log(change+3) #seems to work for transformer

            total_shares_value = len(self.trader.inventory)*value
            number_buy = np.floor((current_action_percentage*(total_shares_value+self.money) - total_shares_value)/value)
            number_buy = int(number_buy)
            if number_buy > 0:
                new_shares = [value]*number_buy
                self.trader.inventory.extend(new_shares)
                self.money -= float(value*number_buy)

            elif number_buy < 0:
                number_sell = -number_buy
                bought_value = np.sum(self.trader.inventory[:number_sell])
                if len(self.trader.inventory)==number_sell:
                    self.trader.inventory = []
                else:
                    self.trader.inventory = self.trader.inventory[number_sell:].copy()
                profit = -bought_value +number_sell*value
                self.train_profit += profit
                self.money += float(value*number_sell)
            
            if done:
                self.train_profit += len(self.trader.inventory)*value
                self.train_profit -= np.sum(self.trader.inventory)
                profit += len(self.trader.inventory)*value - np.sum(self.trader.inventory)

            self.trader.remember(current_state, current_action, change, next_state, done) #loss value due to increasing complexity??
            if len(self.trader.memory) > self.batch_size:
                replay_loss = self.trader.train_experience_replay(self.batch_size)
                if replay_loss != None:
                    loss_list.append(replay_loss)
            current_state = next_state
            old_value = value
        print(f'Testing: self.train_profit: {self.train_profit}')
        print(f'np.mean(loss_list): {np.mean(loss_list)}')
        if episode % 10 ==0:
            self.trader.save(episode)
        return float(self.train_profit), np.mean(loss_list)
    
    def testing(self, dataset):
        self.total_profit = 0
        self.money = 10000
        self.trader.inventory = []
        timeline = []

        dataloader = DataLoader(dataset, batch_size = 1, shuffle=False)
        for (current_state, next_state, value, done) in dataloader:
            profit = 0

            current_action = self.trader.act(current_state, is_eval=True)
            #print(current_action)
            if self.mdp == "all_or_nothing":
                current_action_percentage = current_action
            elif self.mdp == "all_10%_steps":
                current_action_percentage = current_action/10
            else: raise ValueError("Wow, this is the wrong file!")

            total_shares_value = len(self.trader.inventory)*value
            Portfolio_value = total_shares_value+self.money

            number_buy = np.floor((current_action_percentage*(total_shares_value+self.money) - total_shares_value)/value)
            number_buy = int(number_buy)

            if number_buy > 0:
                new_shares = [value]*number_buy
                self.trader.inventory.extend(new_shares)
                self.money -= float(value*number_buy)
            
            elif number_buy < 0:
                number_sell = -number_buy
                bought_value = np.sum(self.trader.inventory[:number_sell])
                if len(self.trader.inventory)==number_sell:
                    self.trader.inventory = [].copy()
                else:
                    self.trader.inventory = self.trader.inventory[number_sell:].copy()
                profit = -bought_value +number_sell*value
                self.total_profit += profit
                self.money += value*number_sell

            self.trader.remember(current_state, current_action, profit, next_state, done)
            if done:
                profit += len(self.trader.inventory)*value
                profit -= np.sum(self.trader.inventory)
                self.total_profit += len(self.trader.inventory)*value
                self.total_profit -= np.sum(self.trader.inventory)
            timeline.append((value, current_action_percentage, profit, Portfolio_value, len(self.trader.inventory), number_buy)) #save for the plot
            current_state=next_state
            if done:
                break
        print(f'Validation: self.total_profit: {self.total_profit}')
        return self.total_profit, timeline
