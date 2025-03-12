import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def generate_toy_stock_data(n_days=1000, volatility=0.01, trend=0.0005, seed=42):
    """Generate synthetic stock price data with a slight upward trend and random noise."""
    # np.random.seed(seed)
    
    # # Initial price
    # price = 100.0
    # prices = [price]
    
    # # Generate prices with random walk with drift
    # for _ in range(n_days - 1):
    #     change = np.random.normal(trend, volatility) * price
    #     price += change
    #     prices.append(max(price, 1.0))  # Ensure price doesn't go below 1
    
    # # Convert to pandas DataFrame
    # dates = pd.date_range(start='2020-01-01', periods=n_days)
    # df = pd.DataFrame({
    #     'date': dates,
    #     'price': prices
    # })

    df = pd.read_csv("../data/GOOG.csv")

    # Add some technical indicators
    # Simple Moving Average (SMA)
    df['sma_10'] = df['Open'].rolling(window=10).mean()
    df['sma_30'] = df['Open'].rolling(window=30).mean()
    
    # Relative Strength Index (RSI) - simplified calculation
    delta = df['Open'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['ema_12'] = df['Open'].ewm(span=12).mean()
    df['ema_26'] = df['Open'].ewm(span=26).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # Forward fill NaN values resulting from rolling windows
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

# Stock Trading Environment
class StockTradingEnv:
    def __init__(self, price_data, features, window_size=10, initial_balance=10000, 
                 transaction_fee_percent=0.001):
        """
        A stock trading environment for reinforcement learning.
        
        Args:
            price_data: DataFrame with stock price and other features
            features: List of feature column names to include in state
            window_size: Number of time steps to include in the state
            initial_balance: Starting account balance
            transaction_fee_percent: Fee applied to transactions (as a percentage)
        """
        self.data = price_data
        self.features = features
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        
        # Calculate feature dimension
        self.feature_dim = len(features)
        self.state_dim = self.window_size * self.feature_dim + 3  # +3 for balance, shares, portfolio value
        
        # Action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_dim = 3
        
        self.reset()
        
    def reset(self):
        """Reset the environment to the beginning of an episode."""
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_price = self.data['Open'].iloc[self.current_step]
        self.done = False
        self.portfolio_value = self.balance + self.shares_held * self.current_price
        self.trades = []
        
        return self._get_observation()
    
    def _get_observation(self):
        """Return the current state observation with feature window and portfolio info."""
        # Get window of features
        feature_matrix = np.zeros((self.window_size, self.feature_dim))
        
        for i, feature in enumerate(self.features):
            values = self.data[feature].iloc[self.current_step - self.window_size:self.current_step].values

            # Normalize values to improve learning
            feature_mean = np.mean(values)
            feature_std = np.std(values) + 1e-10  # Avoid division by zero
            normalized_values = (values - feature_mean) / feature_std
            
            feature_matrix[:, i] = normalized_values
        
        # Flatten the feature matrix to 1D array
        features_flat = feature_matrix.flatten()
        
        # Add portfolio information
        portfolio_info = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.shares_held * self.current_price / self.initial_balance,  # Normalized position value
            self.portfolio_value / self.initial_balance  # Normalized portfolio value
        ])
        
        return np.concatenate([features_flat, portfolio_info])
    
    def step(self, action):
        """Take an action (BUY/SELL/HOLD) and move to the next time step."""
        # Store the previous portfolio value for reward calculation
        prev_portfolio_value = self.portfolio_value
        
        # Move to the next time step
        self.current_step += 1
        self.current_price = self.data['Open'].iloc[self.current_step]
        
        # Execute trading action
        transaction_cost = 0  # Track transaction costs
        
        if action == 1:  # BUY
            if self.balance > 0:
                # Calculate max shares that can be bought
                max_shares = self.balance // self.current_price
                if max_shares > 0:
                    # Calculate cost with transaction fee
                    cost = max_shares * self.current_price
                    transaction_cost = cost * self.transaction_fee_percent
                    total_cost = cost + transaction_cost
                    
                    # Adjust for transaction costs
                    if total_cost <= self.balance:
                        self.shares_held += max_shares
                        self.balance -= total_cost
                        self.trades.append(('BUY', self.current_step, max_shares, self.current_price))
                    else:
                        # Recalculate with fees in mind
                        adjusted_max_shares = (self.balance / (1 + self.transaction_fee_percent)) // self.current_price
                        cost = adjusted_max_shares * self.current_price
                        transaction_cost = cost * self.transaction_fee_percent
                        self.shares_held += adjusted_max_shares
                        self.balance -= (cost + transaction_cost)
                        self.trades.append(('BUY', self.current_step, adjusted_max_shares, self.current_price))
                    
        elif action == 2:  # SELL
            if self.shares_held > 0:
                # Calculate revenue and transaction fee
                revenue = self.shares_held * self.current_price
                transaction_cost = revenue * self.transaction_fee_percent
                self.balance += (revenue - transaction_cost)
                self.trades.append(('SELL', self.current_step, self.shares_held, self.current_price))
                self.shares_held = 0
        
        # Update portfolio value
        self.portfolio_value = self.balance + self.shares_held * self.current_price
        
        # Calculate reward
        reward = self._calculate_reward(prev_portfolio_value, transaction_cost)
        
        # Check if the episode is done
        if self.current_step >= len(self.data) - 1:
            self.done = True
            # Liquidate any remaining position at the end for proper evaluation
            if self.shares_held > 0:
                final_revenue = self.shares_held * self.current_price
                final_fee = final_revenue * self.transaction_fee_percent
                self.balance += (final_revenue - final_fee)
                self.portfolio_value = self.balance
        
        # Return the next observation, reward, done flag, and additional info
        obs = self._get_observation()
        info = {
            'portfolio_value': self.portfolio_value,
            'current_price': self.current_price,
            'shares_held': self.shares_held,
            'balance': self.balance
        }
        
        return obs, reward, self.done, info
    
    def _calculate_reward(self, prev_portfolio_value, transaction_cost):
        """Calculate the reward based on portfolio value change and penalties."""
        # Base reward: portfolio value change
        portfolio_change = self.portfolio_value - prev_portfolio_value
        
        # Sharpe ratio component (approximation using single step)
        portfolio_return = portfolio_change / prev_portfolio_value if prev_portfolio_value > 0 else 0
        
        # Transaction cost penalty (discourages frequent trading)
        transaction_penalty = -transaction_cost * 5  # Amplify the penalty
        
        # Combine rewards
        reward = portfolio_change + transaction_penalty
        
        return reward
    
    def render(self, mode='human'):
        """Render the environment's current state (for visualization)."""
        print(f"Step: {self.current_step}")
        print(f"Price: ${self.current_price:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Shares held: {self.shares_held}")
        print(f"Portfolio value: ${self.portfolio_value:.2f}")
        print(f"Total trades: {len(self.trades)}")
    
    def plot_performance(self, title="Trading Agent Performance"):
        """Plot the agent's performance compared to buy and hold strategy."""
        # Extract prices for the trading period
        prices = self.data['Open'].iloc[self.window_size:self.current_step+1].values
        
        # Calculate buy and hold performance
        initial_price = prices[0]
        buy_hold_value = [self.initial_balance / initial_price * price for price in prices]
        
        # Reconstruct agent's portfolio value history
        portfolio_values = []
        current_balance = self.initial_balance
        current_shares = 0
        
        for step, price in enumerate(prices):
            # Apply trades at this step
            for trade_type, trade_step, shares, trade_price in self.trades:
                adjusted_step = trade_step - self.window_size
                if adjusted_step == step:
                    if trade_type == 'BUY':
                        cost = shares * trade_price
                        fee = cost * self.transaction_fee_percent
                        current_balance -= (cost + fee)
                        current_shares += shares
                    elif trade_type == 'SELL':
                        revenue = shares * trade_price
                        fee = revenue * self.transaction_fee_percent
                        current_balance += (revenue - fee)
                        current_shares -= shares
            
            # Calculate portfolio value at this step
            portfolio_value = current_balance + current_shares * price
            portfolio_values.append(portfolio_value)
        
        # Calculate returns
        agent_returns = (portfolio_values[-1] / self.initial_balance - 1) * 100
        buy_hold_returns = (buy_hold_value[-1] / self.initial_balance - 1) * 100
        
        # Plot
        plt.figure(figsize=(14, 7))
        plt.plot(portfolio_values, label=f'Agent (Return: {agent_returns:.2f}%)')
        plt.plot(buy_hold_value, label=f'Buy & Hold (Return: {buy_hold_returns:.2f}%)')
        
        # Mark trades on the chart
        for trade_type, trade_step, shares, price in self.trades:
            adjusted_step = trade_step - self.window_size
            if adjusted_step >= 0 and adjusted_step < len(portfolio_values):
                marker = '^' if trade_type == 'BUY' else 'v'
                color = 'g' if trade_type == 'BUY' else 'r'
                plt.scatter(adjusted_step, portfolio_values[adjusted_step], marker=marker, color=color, s=100)
        
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        return plt

# Actor-Critic Network for PPO
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
            
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        shared_features = self.shared(state)
        
        # Actor: action probabilities
        action_probs = self.actor(shared_features)
        
        # Critic: state value
        state_value = self.critic(shared_features)
        
        return action_probs, state_value
        
    def act(self, state):
        """Select an action from the policy."""
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob

    def evaluate(self, states, actions):
        """Evaluate actions given states."""
        if isinstance(states, np.ndarray):
            states = torch.FloatTensor(states)
        if isinstance(actions, np.ndarray):
            actions = torch.LongTensor(actions)
            
        action_probs, state_values = self.forward(states)
        dist = Categorical(action_probs)
        
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        
        return action_log_probs, state_values, dist_entropy

# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.0003, 
                 gamma=0.99, clip_param=0.2, value_coef=0.5, entropy_coef=0.01):
        """
        PPO Agent implementation.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Size of hidden layers
            lr: Learning rate
            gamma: Discount factor
            clip_param: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
        """
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        
    def select_action(self, state):
        """Select an action from the current policy."""
        return self.policy.act(state)
    
    def update_policy(self, rollout):
        """Update policy using collected rollout data."""
        # Extract data from rollout
        states = torch.FloatTensor(np.array(rollout['states'])).to(self.device)
        actions = torch.LongTensor(np.array(rollout['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(rollout['log_probs'])).to(self.device)
        returns = torch.FloatTensor(np.array(rollout['returns'])).to(self.device)
        advantages = torch.FloatTensor(np.array(rollout['advantages'])).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get current policy evaluations
        new_log_probs, state_values, entropy = self.policy.evaluate(states, actions)
        
        # Calculate policy loss using clipped objective
        ratios = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value function loss
        value_loss = 0.5 * ((state_values.squeeze() - returns) ** 2).mean()
        
        # Entropy bonus (encourages exploration)
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)  # Gradient clipping
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def save(self, path):
        """Save the agent's policy."""
        torch.save(self.policy.state_dict(), path)
    
    def load(self, path):
        """Load a saved policy."""
        self.policy.load_state_dict(torch.load(path, map_location=self.device))

# Training function
def train_ppo(env, agent, num_episodes=100, update_interval=2048, num_updates=10,
              save_path=None, eval_interval=10):
    """
    Train a PPO agent on the stock trading environment.
    
    Args:
        env: Trading environment
        agent: PPO agent
        num_episodes: Number of episodes to train for
        update_interval: Number of steps between policy updates
        num_updates: Number of times to update policy per interval
        save_path: Path to save the trained model
        eval_interval: Episodes between evaluations
    """
    # Tracking metrics
    episode_rewards = []
    portfolio_values = []
    loss_history = {'policy': [], 'value': [], 'entropy': [], 'total': []}
    
    total_steps = 0
    best_reward = -np.inf
    
    for episode in range(1, num_episodes + 1):
        # Reset environment
        state = env.reset()
        done = False
        episode_reward = 0
        
        # Collect rollout data
        rollout = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        # Run episode
        with tqdm(desc=f"Episode {episode}/{num_episodes}", leave=False) as pbar:
            while not done:
                # Select action from policy
                action, log_prob = agent.select_action(state)
                
                # Get value estimate
                with torch.no_grad():
                    _, value = agent.policy(torch.FloatTensor(state))
                
                # Execute action in environment
                next_state, reward, done, info = env.step(action)
                
                # Store transition in rollout
                rollout['states'].append(state)
                rollout['actions'].append(action)
                rollout['rewards'].append(reward)
                rollout['values'].append(value.item())
                rollout['log_probs'].append(log_prob.item())
                rollout['dones'].append(done)
                
                # Update state and metrics
                state = next_state
                episode_reward += reward
                total_steps += 1
                
                pbar.update(1)
                pbar.set_postfix({'reward': episode_reward, 'portfolio': info['portfolio_value']})
                
                # Update policy if enough data is collected
                if total_steps % update_interval == 0:
                    # Compute returns and advantages
                    returns, advantages = compute_gae(
                        rollout['rewards'], 
                        rollout['values'], 
                        rollout['dones'], 
                        agent.gamma, 
                        0.95  # GAE lambda
                    )
                    
                    # Create full rollout dict with computed values
                    full_rollout = {
                        'states': rollout['states'],
                        'actions': rollout['actions'],
                        'log_probs': rollout['log_probs'],
                        'returns': returns,
                        'advantages': advantages
                    }
                    
                    # Update policy multiple times
                    for _ in range(num_updates):
                        losses = agent.update_policy(full_rollout)
                    
                    # Track losses
                    for key, value in losses.items():
                        if key in loss_history:
                            loss_history[key].append(value)
                    
                    # Clear rollout data
                    for k in rollout:
                        rollout[k] = []
        
        # Track episode metrics
        episode_rewards.append(episode_reward)
        portfolio_values.append(env.portfolio_value)
        
        print(f"Episode {episode}: Reward={episode_reward:.2f}, "
              f"Portfolio=${env.portfolio_value:.2f}, "
              f"Trades={len(env.trades)}")
        
        # Evaluate and potentially save model
        if episode % eval_interval == 0 or episode == num_episodes:
            # Calculate metrics over the last few episodes
            avg_reward = np.mean(episode_rewards[-eval_interval:])
            avg_portfolio = np.mean(portfolio_values[-eval_interval:])
            
            print(f"\nEvaluation after Episode {episode}:")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Average Portfolio Value: ${avg_portfolio:.2f}")
            
            # Save best model based on reward
            if avg_reward > best_reward and save_path:
                best_reward = avg_reward
                agent.save(save_path)
                print(f"Saved best model with avg reward: {best_reward:.2f}")
    
    return episode_rewards, portfolio_values, loss_history

# Compute Generalized Advantage Estimation (GAE)
def compute_gae(rewards, values, dones, gamma, lam):
    """
    Compute returns and advantages using GAE.
    
    Args:
        rewards: List of rewards
        values: List of value estimates
        dones: List of done flags
        gamma: Discount factor
        lam: GAE lambda parameter
    
    Returns:
        returns: Computed returns
        advantages: Computed advantages
    """
    # Add an additional value estimate for the final state
    values = np.append(values, values[-1] if not dones[-1] else 0)
    
    # Initialize arrays
    advantages = np.zeros_like(rewards, dtype=np.float32)
    returns = np.zeros_like(rewards, dtype=np.float32)
    
    # Initialize gae
    gae = 0
    
    # Compute backwards
    for t in reversed(range(len(rewards))):
        # Compute delta (TD error)
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        
        # Compute GAE
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        
        # Store advantage and return
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    
    return returns, advantages

# Evaluation function
def evaluate_agent(env, agent, num_episodes=5):
    """
    Evaluate the agent's performance.
    
    Args:
        env: Trading environment
        agent: Trained agent
        num_episodes: Number of episodes to evaluate
    
    Returns:
        mean_reward: Mean reward across episodes
        mean_portfolio: Mean final portfolio value
    """
    rewards = []
    portfolios = []
    trade_counts = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Select action (no exploration during evaluation)
            with torch.no_grad():
                action, _ = agent.select_action(state)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
        
        # Track metrics
        rewards.append(episode_reward)
        portfolios.append(env.portfolio_value)
        trade_counts.append(len(env.trades))
        
        print(f"Eval Episode {episode+1}: Reward={episode_reward:.2f}, "
              f"Portfolio=${env.portfolio_value:.2f}, "
              f"Trades={len(env.trades)}")
    
    # Calculate mean metrics
    mean_reward = np.mean(rewards)
    mean_portfolio = np.mean(portfolios)
    mean_trades = np.mean(trade_counts)
    
    print(f"\nEvaluation Results:")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Mean Portfolio Value: ${mean_portfolio:.2f}")
    print(f"Mean Number of Trades: {mean_trades:.2f}")
    
    # Plot performance for the last episode
    plt_obj = env.plot_performance(title="Agent Evaluation Performance")
    
    return mean_reward, mean_portfolio, plt_obj

# Main function to run the entire pipeline
def main():
    # Parameters
    window_size = 10
    initial_balance = 10000
    transaction_fee = 0.001  # 0.1% per transaction
    
    # Generate synthetic data
    data = generate_toy_stock_data(n_days=1000, volatility=0.015, trend=0.0003)
    
    # Define features to use
    features = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "sma_10", "sma_30", "rsi", "ema_12", "ema_26", "macd", "macd_signal"]
    
    # Split data into train and test
    train_data = data.iloc[:800].reset_index(drop=True)
    test_data = data.iloc[800:].reset_index(drop=True)

    # Create environments
    train_env = StockTradingEnv(
        train_data, 
        features=features,
        window_size=window_size,
        initial_balance=initial_balance,
        transaction_fee_percent=transaction_fee
    )
    
    test_env = StockTradingEnv(
        test_data, 
        features=features,
        window_size=window_size,
        initial_balance=initial_balance,
        transaction_fee_percent=transaction_fee
    )
    
    # Create PPO agent
    state_dim = train_env.state_dim
    action_dim = train_env.action_dim
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        lr=0.0003,
        gamma=0.99,
        clip_param=0.2,
        value_coef=0.5,
        entropy_coef=0.01
    )
    
    print(f"Starting training with:")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Training data: {len(train_data)} days")
    print(f"Test data: {len(test_data)} days")
    
    # Train the agent
    rewards, portfolios, losses = train_ppo(
        train_env,
        agent,
        num_episodes=50,
        update_interval=1024,
        num_updates=10,
        save_path="ppo_stock_trader.pth",
        eval_interval=5
    )
    
    # Plot training performance
    plt.figure(figsize=(14, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # Plot portfolio values
    plt.subplot(2, 2, 2)
    plt.plot(portfolios)
    plt.axhline(y=initial_balance, color='r', linestyle='--', label='Initial Balance')
    plt.title('Portfolio Values')
    plt.xlabel('Episode')
    plt.ylabel('Value ($)')
    plt.legend()
    plt.grid(True)
    
    # Plot losses
    plt.subplot(2, 2, 3)
    plt.plot(losses['policy'], label='Policy Loss')
    plt.plot(losses['value'], label='Value Loss')
    plt.title('Training Losses')
    plt.xlabel('Update')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(losses['entropy'], label='Entropy')
    plt.title('Entropy')
    plt.xlabel('Update')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_performance.png')
    
    print("Training completed. Evaluating on test data...")
    
    # Load best model
    agent.load("ppo_stock_trader.pth")
    
    # Evaluate on test data
    mean_reward, mean_portfolio, plot = evaluate_agent(test_env, agent, num_episodes=1)
    
    # Save evaluation plot
    plot.savefig('evaluation_performance.png')
    
    print("Complete pipeline executed successfully.")
    print(f"Final portfolio value: ${mean_portfolio:.2f} (Starting from ${initial_balance:.2f})")
    print(f"Return: {(mean_portfolio/initial_balance - 1) * 100:.2f}%")
    
    # Show final plots
    plt.show()

# Run the pipeline if executed as a script
if __name__ == "__main__":
    main()