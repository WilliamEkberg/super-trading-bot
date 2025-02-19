import os
import logging
import numpy as np
from tqdm import tqdm

from .utils import format_currency, format_position
from .ops import get_state

def train_model(agent, episode, data, ep_count=100, batch_size=32, window_size=10):
    total_profit = 0
    data_length = len(data) - 1

    # Reset the agent's inventory and memory for this episode
    agent.inventory = []
    avg_loss = []

    # Get the initial state representation
    state = get_state(data, 0, window_size + 1)

    for t in tqdm(range(data_length), total=data_length, leave=True, 
                  desc=f'Episode {episode}/{ep_count}'):
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)

        # Select an action using the agent's policy
        action = agent.act(state)

        # BUY action
        if action == 1:
            agent.inventory.append(data[t])

        # SELL action
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data[t] - bought_price
            reward = delta  # profit or loss from the sale
            total_profit += delta

        # HOLD action does nothing

        done = (t == data_length - 1)
        agent.remember(state, action, reward, next_state, done)

        # Train the agent if we have enough experiences
        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            if loss is not None:
                avg_loss.append(loss)

        state = next_state

    # Periodically save the model
    if episode % 10 == 0:
        agent.save(episode)

    # Return episode statistics
    avg_loss_value = np.mean(avg_loss) if avg_loss else 0
    return (episode, ep_count, total_profit, avg_loss_value)


def evaluate_model(agent, data, window_size, debug=False):
    total_profit = 0
    data_length = len(data) - 1

    history = []
    agent.inventory = []
    
    # Initialize state from the data
    state = get_state(data, 0, window_size + 1)

    for t in range(data_length):
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)
        
        # Use evaluation mode when selecting an action
        action = agent.act(state, is_eval=True)

        # BUY action
        if action == 1:
            agent.inventory.append(data[t])
            history.append((data[t], "BUY"))
            if debug:
                logging.debug(f"Buy at: {format_currency(data[t])}")
        
        # SELL action
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data[t] - bought_price
            reward = delta
            total_profit += delta
            history.append((data[t], "SELL"))
            if debug:
                logging.debug(f"Sell at: {format_currency(data[t])} | Position: {format_position(data[t] - bought_price)}")
        # HOLD action
        else:
            history.append((data[t], "HOLD"))

        done = (t == data_length - 1)
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            break

    return total_profit, history
