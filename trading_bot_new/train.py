import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

class Trainer():
    def __init__(self, dataloader_train, dataloader_val, trader, batch_size=32, obs_window=10):
        self.total_profit = 0
        self.train_profit = 0
        self.batch_size = batch_size
        self.obs_window = obs_window
        self.trader = trader

    
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
            #if change>250:
                #print(change)
                #print(np.tanh(change/100))
            change = -np.exp(-change/150) + 1
            #print("exp", change)
            change = np.tanh(change*2) #from /100 to /150
            #print("tanh", change)
            #change = np.log(change+3) #seems to work for transformer
            #print("log", change)

            current_action = self.trader.act(current_state)
            current_action_percentage = current_action/10

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
            
                #print("bought value", bought_value)
                #print("sell value", value*number_sell)
                #print("profit", profit)
            #print(change)
            #print(self.trader.inventory)
            #print("len of inventory", len(self.trader.inventory))
            #print("number_buy", number_buy)
            #print(self.money+total_shares_value)
            #if current_action == 1: #Buy
            #    self.trader.inventory.append(value)

            #elif current_action == 2 and len(self.trader.inventory) >0: #Sell
            #    bought_value = self.trader.inventory.pop(0)
            #    current_value = value
            #    profit = current_value - bought_value
            #    self.train_profit += profit
            if done: #change???
                self.train_profit += len(self.trader.inventory)*value
                self.train_profit -= np.sum(self.trader.inventory)
                profit += len(self.trader.inventory)*value - np.sum(self.trader.inventory)
            #print("profit", self.train_profit)
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
            current_action_percentage = current_action/10
            timeline.append((value, current_action_percentage))

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
                    self.trader.inventory = [].copy()
                else:
                    self.trader.inventory = self.trader.inventory[number_sell:].copy()
                profit = -bought_value +number_sell*value
                self.total_profit += profit
                self.money += value*number_sell

            #if current_action == 1:
            #    self.trader.inventory.append(value)
            #    timeline.append((value, "Buying"))
        
            #elif current_action==2 and len(self.trader.inventory) >0:
            #    bought_value = self.trader.inventory.pop(0)
            #    current_value = value
            #    profit = current_value - bought_value
            #    total_profit += profit
            #    timeline.append((value, "Selling"))

            #else:
            #    timeline.append((value, "HODL"))
            #print("Total money", self.money+len(self.trader.inventory)*value)
            self.trader.remember(current_state, current_action, profit, next_state, done)
            if done:
                #print("trader.inventory", self.trader.inventory)
                #print("current state", current_state)
                #print("next_state", next_state)
                #print("value", value)
                self.total_profit += len(self.trader.inventory)*value
                self.total_profit -= np.sum(self.trader.inventory)
            current_state=next_state
            if done:
                break
        print(f'Validation: self.total_profit: {self.total_profit}')
        return self.total_profit, timeline
