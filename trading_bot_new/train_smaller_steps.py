import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch

class Trainer_InSteps():
    def __init__(self, dataloader_train, dataloader_val, trader, batch_size=32, obs_window=10, mdp=""):
        self.total_profit = 0
        self.train_profit = 0
        self.batch_size = batch_size
        self.obs_window = obs_window
        self.trader = trader
        if mdp != "10%_steps": raise ValueError("Wow, the mdp doesn't work with 10'%'steps")
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
        old_percentage = 0
        for (current_state, next_state, value, done) in tqdm(dataloader):
            profit = 0
            if old_value == None: old_value=value
            change = len(self.trader.inventory)*(value-old_value)
            #if change>250:
                #print(change)
                #print(np.tanh(change/100))
            #change = -np.exp(-change/150) + 1 #/150, +1
            #print("exp", change)
            change = np.tanh(change/40) #from /100 to /150 #*2 with exp
            #print(change)
            #print("tanh", change)
            #change = np.log(change+3) #seems to work for transformer
            #print("log", change)
            #print(current_state)
            #print(torch.tensor(old_percentage).view(1,1,-1))
            current_state = torch.cat((current_state, torch.tensor(old_percentage).view(1,1,-1)), dim=2)#for steps
            current_action = self.trader.act(current_state)
            #print(current_action)
            current_action_percentage = (old_percentage*10 + current_action-1)/10
            if current_action_percentage>1: #no bigger invest than 100%
                current_action_percentage = 1
            elif current_action_percentage<0: #no smaller invest than 0%
                current_action_percentage = 0
            next_state = torch.cat((next_state, torch.tensor(current_action_percentage).view(1,1,-1)), dim=2)#for steps
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
            old_percentage = current_action_percentage
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
        old_percentage = 0
        for (current_state, next_state, value, done) in dataloader:
            if len(timeline) == 0: timeline.insert(0, (float(value), 0, 0, 10000, 0, 0))
            profit = 0

            current_state = torch.cat((current_state, torch.tensor(old_percentage).view(1,1,-1)), dim=2)#for steps
            current_action = self.trader.act(current_state)

            current_action_percentage = (old_percentage*10 + current_action-1)/10 #minus 1 to get [0, 1, 2] -> [-0.1, 0, 0.1]

            if current_action_percentage>1: #no bigger invest than 100%
                current_action_percentage = 1
            elif current_action_percentage<0: #no smaller invest than 0%
                current_action_percentage = 0

            next_state = torch.cat((next_state, torch.tensor(current_action_percentage).view(1,1,-1)), dim=2)#for steps


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
            timeline.append((float(value), current_action_percentage, float(profit), float(Portfolio_value), len(self.trader.inventory), number_buy)) #save for the plot
            old_percentage = current_action_percentage
            current_state=next_state
            if done:
                break
        print(f'Validation: self.total_profit: {self.total_profit}')
        return self.total_profit, timeline
