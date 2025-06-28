"""
Enhanced PPO VANET Server
=========================

This script implements a PPO (Proximal Policy Optimization) reinforcement learning server for VANET optimization.

QUICK START:
1. For TRAINING: Set OPERATION_MODE = "TRAINING" below
2. For TESTING: Set OPERATION_MODE = "TESTING" below  
3. Run: python ppo_server.py

OUTPUT FILES:
- ppo_model.pth: Trained PPO model
- performance_results.xlsx: Comprehensive performance analysis (4 sheets)
- logs/: Directory with detailed debug logs and TensorBoard logs

MODES:
- TRAINING: Trains the PPO agent, saves model periodically
- TESTING: Uses pre-trained model with no exploration for evaluation
"""

import socket
import threading
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import entropy
from collections import defaultdict, deque
import random
import os
import time
import math
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical, Normal
from datetime import datetime
import sys

# ================== CONFIGURATION ==================
# CHANGE THIS TO SWITCH BETWEEN MODES
OPERATION_MODE = "TESTING"        # Options: "TRAINING" or "TESTING"

# ==============================
# Constants and Hyperparameters
# ==============================
MBL = 0.6  # Maximum Busy Load (target CBR from paper)
CBR_RANGE = (0.5, 0.8)  # Wider safety bounds as in MATLAB code
BUFFER_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.0003
EPS_CLIP = 0.2
K_EPOCHS = 10
HIDDEN_UNITS = 256
HOST = '127.0.0.1'
PORT = 5000
MODEL_SAVE_PATH = 'ppo_model.pth'
PERFORMANCE_LOG_PATH = 'performance_results.xlsx'
MODEL_SAVE_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 10

# Log Configuration
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_RECEIVED_PATH = os.path.join(LOG_DIR, 'received.log')
LOG_SENT_PATH = os.path.join(LOG_DIR, 'sent.log')
LOG_DEBUG_PATH = os.path.join(LOG_DIR, 'debug.log')

# Environment parameters from the paper
SAFETY_DISTANCE = 100  # meters
PATH_LOSS_EXPONENT = 2.5
SENSITIVITIES = {
    0: -92,   # MCS 0 sensitivity in dBm
    1: -90, 2: -88, 3: -86, 4: -84, 5: -82,
    6: -80, 7: -78, 8: -76, 9: -74, 10: -72
}

# Action space parameters (from paper)
POWER_MIN, POWER_MAX = 1, 30  # dBm
MCS_MIN, MCS_MAX = 0, 9  # Modulation and Coding Scheme levels
DEFAULT_BEACON_RATE = 10  # Hz (default value to return)

# Reward weights from the paper
WC = 2.0    # Channel load weight
WP = 0.25   # Power-sensitivity weight
WD = 0.1    # Data rate weight
WE = 0.8    # Data rate exponent

def log_message(message, log_file=None, print_stdout=True):
    """Enhanced logging function"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_msg = f"[{timestamp}] {message}"
    
    if print_stdout:
        print(formatted_msg)
    
    if log_file:
        with open(log_file, 'a') as f:
            f.write(formatted_msg + "\n")

# ==============================
# Neural Network Architectures
# ==============================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Shared feature extractor
        self.shared_fc1 = nn.Linear(state_dim, HIDDEN_UNITS)
        self.shared_fc2 = nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS)
        
        # Actor network
        self.actor_fc = nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS)
        self.actor_mu = nn.Linear(HIDDEN_UNITS, action_dim)
        self.actor_sigma = nn.Linear(HIDDEN_UNITS, action_dim)
        
        # Critic network
        self.critic_fc = nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS)
        self.value = nn.Linear(HIDDEN_UNITS, 1)

    def forward(self, state):
        # Shared features
        x = torch.relu(self.shared_fc1(state))
        x = torch.relu(self.shared_fc2(x))
        
        # Actor output
        x_actor = torch.relu(self.actor_fc(x))
        mu = torch.tanh(self.actor_mu(x_actor))  # Actions normalized to [-1, 1]
        sigma = torch.clamp(self.actor_sigma(x_actor), min=-20, max=2)
        sigma = torch.exp(sigma)
        sigma = torch.clamp(sigma, min=0.01, max=1.0)  # Ensure reasonable sigma range
        
        # Critic output
        x_critic = torch.relu(self.critic_fc(x))
        value = self.value(x_critic)
        
        return mu, sigma, value

# ==============================
# Enhanced Performance Metrics Tracker
# ==============================
class PerformanceTracker:
    def __init__(self):
        self.reset_metrics()
        self.writer = SummaryWriter(LOG_DIR)
        self.episode_data = []
        self.detailed_logs = []
        
    def reset_metrics(self):
        self.cumulative_reward = 0
        self.action_distribution = defaultdict(int)
        self.action_history = []
        self.cbr_values = []
        self.rewards = []
        self.state_history = []
        self.action_changes = []
        self.safety_constraint_violations = 0
        self.total_steps = 0
        self.losses = []
        self.policy_entropies = []
        self.value_estimates = []
        self.exploration_steps = 0
        self.exploitation_steps = 0
        
    def update_metrics(self, state, action, reward, next_state, is_exploration=False, policy_entropy=0, value_estimate=0):
        self.total_steps += 1
        self.cumulative_reward += reward
        self.rewards.append(reward)
        self.cbr_values.append(state[0])  # CBR from current state
        self.action_distribution[(round(action[0], 1), action[1])] += 1
        self.action_history.append(action)
        self.state_history.append(next_state)
        self.policy_entropies.append(policy_entropy)
        self.value_estimates.append(value_estimate)
        
        if is_exploration:
            self.exploration_steps += 1
        else:
            self.exploitation_steps += 1
        
        if len(self.action_history) > 1:
            self.action_changes.append(np.linalg.norm(
                np.array(self.action_history[-1]) - np.array(self.action_history[-2])
            ))
        
        if not (CBR_RANGE[0] <= next_state[0] <= CBR_RANGE[1]):
            self.safety_constraint_violations += 1
            
        # Detailed log for each step
        self.detailed_logs.append({
            'timestamp': datetime.now(),
            'cbr': state[0],
            'action_power': action[0],
            'action_mcs': action[1],
            'reward': reward,
            'next_cbr': next_state[0],
            'policy_entropy': policy_entropy,
            'value_estimate': value_estimate,
            'is_exploration': is_exploration
        })
    
    def record_loss(self, loss):
        self.losses.append(loss)
    
    def calculate_episode_metrics(self, episode_num, epsilon):
        """Calculate comprehensive episode metrics for PPO"""
        if not self.rewards:
            return {}
            
        metrics = {
            'episode': episode_num,
            'timestamp': datetime.now(),
            'total_steps': self.total_steps,
            'cumulative_reward': self.cumulative_reward,
            'average_reward': np.mean(self.rewards),
            'max_reward': max(self.rewards),
            'min_reward': min(self.rewards),
            'reward_std': np.std(self.rewards),
            
            # CBR Performance
            'avg_cbr': np.mean(self.cbr_values),
            'cbr_std': np.std(self.cbr_values),
            'cbr_in_range_rate': sum(1 for cbr in self.cbr_values if CBR_RANGE[0] <= cbr <= CBR_RANGE[1]) / len(self.cbr_values),
            'cbr_violation_rate': self.safety_constraint_violations / self.total_steps if self.total_steps > 0 else 0,
            'cbr_target_deviation': np.mean([abs(cbr - MBL) for cbr in self.cbr_values]),
            
            # PPO-Specific Metrics
            'policy_convergence': np.var([a[0] for a in self.action_history[-max(1, len(self.action_history)//10):]]) if self.action_history else 0,
            'action_jitter': np.mean(self.action_changes) if self.action_changes else 0,
            'avg_policy_entropy': np.mean(self.policy_entropies),
            'avg_value_estimate': np.mean(self.value_estimates),
            'value_estimate_std': np.std(self.value_estimates),
            
            # Action Analysis
            'avg_power': np.mean([a[0] for a in self.action_history]) if self.action_history else 0,
            'avg_mcs': np.mean([a[1] for a in self.action_history]) if self.action_history else 0,
            'power_std': np.std([a[0] for a in self.action_history]) if self.action_history else 0,
            'mcs_std': np.std([a[1] for a in self.action_history]) if self.action_history else 0,
            
            # Training Metrics
            'training_loss': np.mean(self.losses) if self.losses else 0,
            'loss_std': np.std(self.losses) if self.losses else 0,
            'exploration_rate': self.exploration_steps / (self.exploration_steps + self.exploitation_steps) if (self.exploration_steps + self.exploitation_steps) > 0 else 0,
            'exploration_steps': self.exploration_steps,
            'exploitation_steps': self.exploitation_steps,
            'current_epsilon': epsilon,
            
            # Performance Indicators
            'transmission_efficiency': self.cumulative_reward / (np.mean([a[0] for a in self.action_history]) + 1e-6) if self.action_history else 0,
            'safety_constraint_rate': 1 - (self.safety_constraint_violations / self.total_steps if self.total_steps > 0 else 0),
            'evaluation_success_rate': 1 if (1 - self.safety_constraint_violations / self.total_steps if self.total_steps > 0 else 0) > 0.95 else 0,
        }
        
        # Action distribution entropy
        action_counts = np.array(list(self.action_distribution.values()))
        if len(action_counts) > 0:
            action_probs = action_counts / np.sum(action_counts)
            metrics['exploration_entropy'] = entropy(action_probs)
            metrics['action_diversity'] = len(self.action_distribution) / max(1, self.total_steps)
        else:
            metrics['exploration_entropy'] = 0
            metrics['action_diversity'] = 0
        
        return metrics
    
    def log_performance(self, episode_num, epsilon):
        """Log performance and return metrics"""
        metrics = self.calculate_episode_metrics(episode_num, epsilon)
        if metrics:
            self.episode_data.append(metrics)
            
            # Log to TensorBoard
            for key, value in metrics.items():
                if key not in ['episode', 'timestamp'] and isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, episode_num)
                    
            if episode_num % PERFORMANCE_LOG_INTERVAL == 0:
                self.save_to_excel()
                
        return metrics
    
    def save_to_excel(self):
        """Save performance data to Excel with multiple sheets"""
        try:
            with pd.ExcelWriter(PERFORMANCE_LOG_PATH, engine='openpyxl', mode='w') as writer:
                # Episode Summary Sheet
                if self.episode_data:
                    episode_df = pd.DataFrame(self.episode_data)
                    episode_df.to_excel(writer, sheet_name='Episode_Summary', index=False)
                
                # Detailed Logs Sheet (last 1000 entries to avoid huge files)
                if self.detailed_logs:
                    detailed_df = pd.DataFrame(self.detailed_logs[-1000:])
                    detailed_df.to_excel(writer, sheet_name='Detailed_Logs', index=False)
                
                # PPO Performance Analysis
                if self.episode_data:
                    analysis_data = self._generate_performance_analysis()
                    analysis_df = pd.DataFrame([analysis_data])
                    analysis_df.to_excel(writer, sheet_name='PPO_Performance_Analysis', index=False)
                
                # Network Performance Analysis
                if self.episode_data:
                    network_analysis = self._analyze_network_performance()
                    network_df = pd.DataFrame([network_analysis])
                    network_df.to_excel(writer, sheet_name='Network_Performance', index=False)
                
            log_message(f"Performance data saved to {PERFORMANCE_LOG_PATH}", print_stdout=True)
            
        except Exception as e:
            log_message(f"Error saving to Excel: {e}", LOG_DEBUG_PATH, print_stdout=True)
    
    def _generate_performance_analysis(self):
        """Generate comprehensive PPO performance analysis"""
        if not self.episode_data:
            return {}
            
        df = pd.DataFrame(self.episode_data)
        
        return {
            'total_episodes': len(df),
            'total_training_steps': df['total_steps'].sum(),
            'avg_episode_length': df['total_steps'].mean(),
            'convergence_episode': self._detect_convergence(df),
            'learning_efficiency': self._calculate_learning_efficiency(df),
            'final_performance_score': self._calculate_final_performance_score(df),
            'reward_improvement_rate': self._calculate_improvement_rate(df, 'cumulative_reward'),
            'cbr_performance_improvement': self._calculate_improvement_rate(df, 'cbr_in_range_rate'),
            'policy_stability_score': 1 / (1 + df['policy_convergence'].mean()),
            'value_function_accuracy': 1 / (1 + df['value_estimate_std'].mean()),
            'exploration_efficiency': df['exploration_entropy'].mean(),
            'action_consistency': 1 - df['action_jitter'].mean(),
            'safety_performance': df['safety_constraint_rate'].mean(),
            'transmission_efficiency_trend': self._calculate_improvement_rate(df, 'transmission_efficiency'),
        }
    
    def _analyze_network_performance(self):
        """Analyze network-specific performance metrics"""
        if not self.episode_data:
            return {}
            
        df = pd.DataFrame(self.episode_data)
        
        return {
            'avg_cbr_performance': df['avg_cbr'].mean(),
            'cbr_stability': 1 - df['cbr_std'].mean(),
            'target_cbr_adherence': df['cbr_in_range_rate'].mean(),
            'avg_transmission_power': df['avg_power'].mean(),
            'power_efficiency': 1 / (df['avg_power'].mean() + 1e-6),
            'avg_mcs_level': df['avg_mcs'].mean(),
            'mcs_adaptation_rate': df['mcs_std'].mean(),
            'network_safety_score': df['safety_constraint_rate'].mean(),
            'overall_network_efficiency': df['transmission_efficiency'].mean(),
            'cbr_target_deviation': df['cbr_target_deviation'].mean(),
        }
    
    def _detect_convergence(self, df, window=50):
        """Detect convergence episode based on reward stability"""
        if len(df) < window:
            return None
            
        rewards = df['cumulative_reward'].values
        for i in range(window, len(rewards)):
            recent_std = np.std(rewards[i-window:i])
            if recent_std < np.std(rewards[:i]) * 0.1:
                return i
        return None
    
    def _calculate_learning_efficiency(self, df):
        """Calculate how efficiently the agent learns"""
        if len(df) < 10:
            return 0
            
        early_performance = df.head(10)['cumulative_reward'].mean()
        late_performance = df.tail(10)['cumulative_reward'].mean()
        
        if early_performance == 0:
            return float('inf') if late_performance > 0 else 0
            
        return (late_performance - early_performance) / len(df)
    
    def _calculate_final_performance_score(self, df):
        """Calculate final performance score (0-1)"""
        if len(df) < 10:
            return 0
            
        recent_episodes = df.tail(10)
        cbr_score = recent_episodes['cbr_in_range_rate'].mean()
        reward_score = min(1.0, max(0.0, (recent_episodes['cumulative_reward'].mean() + 100) / 200))
        
        return (cbr_score + reward_score) / 2
    
    def _calculate_improvement_rate(self, df, column):
        """Calculate improvement rate for a given metric"""
        if len(df) < 2:
            return 0
            
        values = df[column].values
        return np.polyfit(range(len(values)), values, 1)[0]

# ==============================
# Enhanced PPO Agent
# ==============================
class PPOAgent:
    def __init__(self, state_dim, action_dim, training_mode=True):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=BUFFER_SIZE)
        self.training_mode = training_mode
        self.performance = PerformanceTracker()
        self.episode_count = 0
        
        # Exploration parameters
        self.epsilon = 1.0 if training_mode else 0.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        
        if not training_mode or (training_mode and os.path.exists(MODEL_SAVE_PATH)):
            try:
                self.load_model(MODEL_SAVE_PATH)
                if not training_mode:
                    log_message("Testing mode: Pre-trained model loaded successfully", print_stdout=True)
            except Exception as e:
                if not training_mode:
                    log_message(f"WARNING: Testing mode but failed to load model: {e}", print_stdout=True)
                else:
                    log_message(f"Training mode: Starting fresh (no existing model found)", print_stdout=True)

    def calculate_reward(self, cbr, power, mcs):
        """
        Calculate reward according to the paper's reward function:
        r = ω_c*g(CBR) - ω_p*|(S_r + l) - p| - ω_d*(d)^ω_e
        """
        try:
            # Ensure inputs are real numbers
            cbr = float(np.real(cbr))
            power = float(np.real(power))
            mcs = int(np.real(mcs))
            
            # Channel load term (g(CBR))
            cbr_deviation = cbr - MBL
            g_cbr = -np.sign(cbr_deviation) * cbr
            
            # Special reward when CBR is within target range
            if abs(cbr_deviation) <= 0.025:
                g_cbr += 10.0
            else:
                g_cbr -= 0.1
            
            # Power-sensitivity term
            mcs_int = max(0, min(10, int(round(mcs))))  # Clamp MCS to valid range
            sensitivity = SENSITIVITIES.get(mcs_int, -80)
            path_loss = 10 * PATH_LOSS_EXPONENT * np.log10(max(1.0, SAFETY_DISTANCE))
            power_term = abs((sensitivity + path_loss) - power)
            
            # Data rate term
            data_rate_term = (mcs_int + 1) ** WE
            
            # Combined reward with weights from paper
            reward = WC * g_cbr - WP * power_term - WD * data_rate_term
            
            # Ensure reward is real and finite
            reward = float(np.real(reward))
            if not np.isfinite(reward):
                reward = -10.0  # Default penalty for invalid states
                
            return np.clip(reward, -100.0, 100.0)  # Clip to reasonable range
            
        except Exception as e:
            log_message(f"Error in reward calculation: {e}", LOG_DEBUG_PATH)
            return -10.0  # Default penalty

    def select_action(self, state):
        """Enhanced action selection with better exploration tracking"""
        state_tensor = torch.FloatTensor(state)
        mu, sigma, value = self.policy(state_tensor)
        
        # Ensure mu and sigma are properly bounded
        mu = torch.clamp(mu, -1.0, 1.0)  # Ensure output is in [-1, 1]
        sigma = torch.clamp(sigma, 0.01, 1.0)  # Ensure positive sigma
        
        is_exploration = False
        if self.training_mode and random.random() < self.epsilon:
            action = torch.FloatTensor(2).uniform_(-1, 1)
            is_exploration = True
        else:
            dist = torch.distributions.Normal(mu, sigma)
            action = dist.sample()
            # Clamp sampled action to [-1, 1]
            action = torch.clamp(action, -1.0, 1.0)
            
        # Calculate log probability and entropy
        dist = torch.distributions.Normal(mu, sigma)
        logprob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum()

        # Denormalize actions to their respective ranges with validation
        power = self._denormalize(action[0].item(), POWER_MIN, POWER_MAX)
        mcs = self._denormalize(action[1].item(), MCS_MIN, MCS_MAX, is_int=True)
        
        # Additional validation
        power = np.clip(power, POWER_MIN, POWER_MAX)
        mcs = np.clip(int(mcs), MCS_MIN, MCS_MAX)
        
        return np.array([power, mcs]), logprob.item(), is_exploration, entropy.item(), value.item()

    def _denormalize(self, value, min_val, max_val, is_int=False):
        """Denormalize from [-1, 1] to [min_val, max_val] with proper clamping."""
        # First clamp the normalized value to [-1, 1] range
        value = np.clip(value, -1.0, 1.0)
        
        # Denormalize to target range
        denorm = min_val + (value + 1) * (max_val - min_val) / 2
        
        # Final clamping to ensure bounds
        denorm = np.clip(denorm, min_val, max_val)
        
        return round(denorm) if is_int else denorm

    def update_policy(self):
        if len(self.memory) < BATCH_SIZE:
            return 0
        
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, logprobs, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays first to avoid warnings
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        old_logprobs = torch.FloatTensor(np.array(logprobs)).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones, dtype=np.float32)).unsqueeze(1)
        
        with torch.no_grad():
            _, _, values = self.policy(states)
            _, _, next_values = self.policy(next_states)
            advantages = rewards + GAMMA * next_values * (1 - dones) - values
        
        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        for _ in range(K_EPOCHS):
            mu, sigma, values = self.policy(states)
            dist = torch.distributions.Normal(mu, sigma)
            new_logprobs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().mean()
            
            ratios = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (rewards + GAMMA * next_values * (1 - dones) - values).pow(2).mean()
            loss = policy_loss + value_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        if self.training_mode:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return total_loss / K_EPOCHS

    def save_model(self, path):
        try:
            torch.save({
                'policy_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'episode_count': self.episode_count,
                'performance_data': {
                    'episode_data': self.performance.episode_data,
                    'detailed_logs': self.performance.detailed_logs[-100:]  # Save last 100 detailed logs
                }
            }, path)
            log_message(f"Model saved to {path}", print_stdout=True)
        except Exception as e:
            log_message(f"Error saving model: {e}", LOG_DEBUG_PATH, print_stdout=True)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', 0.0)
        self.episode_count = checkpoint.get('episode_count', 0)
        
        # Load performance data if available
        if 'performance_data' in checkpoint:
            perf_data = checkpoint['performance_data']
            if 'episode_data' in perf_data:
                self.performance.episode_data = perf_data['episode_data']
        
        log_message(f"Model loaded from {path}", print_stdout=True)

    def end_episode(self):
        if self.training_mode:
            self.episode_count += 1
            metrics = self.performance.log_performance(self.episode_count, self.epsilon)
            
            if self.episode_count % MODEL_SAVE_INTERVAL == 0:
                self.save_model(MODEL_SAVE_PATH)
            
            self.performance.reset_metrics()
            return metrics
        return {}

# ==============================
# Enhanced RL Server
# ==============================
class RLServer:
    def __init__(self, host, port, training_mode=True):
        self.host = host
        self.port = port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(5)
        self.training_mode = training_mode
        self.agent = PPOAgent(state_dim=1, action_dim=2, training_mode=training_mode)
        self.running = True
        
        mode_str = "TRAINING" if training_mode else "TESTING"
        log_message(f"PPO Server started in {mode_str} mode on {host}:{port}", print_stdout=True)

    def receive_message_with_header(self, conn):
        """Receive message with 4-byte length header (compatible with simulation script)"""
        try:
            # Receive 4-byte header
            header_data = b''
            while len(header_data) < 4:
                chunk = conn.recv(4 - len(header_data))
                if not chunk:
                    return None
                header_data += chunk
            
            # Parse message length
            message_length = int.from_bytes(header_data, byteorder='little')
            
            # Receive actual message
            message_data = b''
            while len(message_data) < message_length:
                chunk = conn.recv(min(message_length - len(message_data), 8192))
                if not chunk:
                    return None
                message_data += chunk
            
            return message_data.decode('utf-8')
            
        except Exception as e:
            log_message(f"Error receiving message: {e}", LOG_DEBUG_PATH)
            return None

    def send_message_with_header(self, conn, message):
        """Send message with 4-byte length header (compatible with simulation script)"""
        try:
            message_bytes = message.encode('utf-8')
            message_length = len(message_bytes)
            
            # Send length header
            header = message_length.to_bytes(4, byteorder='little')
            conn.sendall(header)
            
            # Send message
            conn.sendall(message_bytes)
            return True
            
        except Exception as e:
            log_message(f"Error sending message: {e}", LOG_DEBUG_PATH)
            return False

    def start(self):
        """Start the server"""
        try:
            log_message("Server listening for connections...", print_stdout=True)
            while self.running:
                try:
                    conn, addr = self.server.accept()
                    # Handle each client in a separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client, 
                        args=(conn, addr),
                        daemon=True
                    )
                    client_thread.start()
                except Exception as e:
                    if self.running:
                        log_message(f"Error accepting connection: {e}", LOG_DEBUG_PATH, print_stdout=True)
        except Exception as e:
            if self.running:
                log_message(f"Server error: {e}", LOG_DEBUG_PATH, print_stdout=True)
        finally:
            self.stop()

    def handle_client(self, conn, addr):
        """Enhanced client handler with proper protocol support"""
        log_message(f"Client connected from {addr}", print_stdout=True)
        
        try:
            while self.running:
                # Receive message with header protocol
                message_str = self.receive_message_with_header(conn)
                if not message_str:
                    break
                
                log_message(f"Received data from {addr}: {message_str[:100]}...", LOG_RECEIVED_PATH)
                
                try:
                    # Parse vehicle data
                    batch_data = json.loads(message_str)
                    log_message(f"Processing {len(batch_data)} vehicles", print_stdout=True)
                    
                    responses = {}
                    
                    for veh_id, vehicle_data in batch_data.items():
                        response = self.process_vehicle(veh_id, vehicle_data)
                        if response:
                            responses[veh_id] = response
                    
                    # FIXED: Wrap responses in "vehicles" key to match simulation script expectation
                    response_dict = {"vehicles": responses}
                    response_str = json.dumps(response_dict)
                    
                    if self.send_message_with_header(conn, response_str):
                        log_message(f"Sent response to {addr}: {len(responses)} vehicles", LOG_SENT_PATH)
                    else:
                        break
                        
                    # Training episode management
                    if self.training_mode and len(self.agent.memory) >= BATCH_SIZE:
                        loss = self.agent.update_policy()
                        self.agent.performance.record_loss(loss)
                        
                        # End episode if we've processed enough steps
                        if self.agent.performance.total_steps >= 100:
                            metrics = self.agent.end_episode()
                            if metrics:
                                log_message(f"Episode {self.agent.episode_count} completed. Reward: {metrics.get('cumulative_reward', 0):.2f}, CBR Rate: {metrics.get('cbr_in_range_rate', 0):.3f}", print_stdout=True)
                
                except json.JSONDecodeError as e:
                    log_message(f"JSON decode error: {e}", LOG_DEBUG_PATH, print_stdout=True)
                    break
                except Exception as e:
                    log_message(f"Error processing vehicle data: {e}", LOG_DEBUG_PATH, print_stdout=True)
                    break
        
        except Exception as e:
            log_message(f"Error in handle_client: {e}", LOG_DEBUG_PATH, print_stdout=True)
        finally:
            try:
                conn.close()
            except:
                pass
            log_message(f"Client {addr} disconnected", print_stdout=True)

    def process_vehicle(self, veh_id, vehicle_data):
        """Process individual vehicle data and return PPO response"""
        try:
            # Extract vehicle parameters with defaults
            cbr = float(vehicle_data.get("CBR", 0.5))
            current_power = float(vehicle_data.get("transmissionPower", 20))
            current_mcs = int(vehicle_data.get("MCS", 5))
            current_beacon = float(vehicle_data.get("beaconRate", DEFAULT_BEACON_RATE))
            
            # Handle NaN values and validate ranges
            if math.isnan(cbr) or cbr < 0 or cbr > 1:
                cbr = 0.5
            
            current_power = np.clip(current_power, POWER_MIN, POWER_MAX)
            current_mcs = np.clip(current_mcs, MCS_MIN, MCS_MAX)
            
            # Create state vector (just CBR for this PPO implementation)
            state = [cbr]
            
            # Select action (power and MCS)
            action, logprob, is_exploration, entropy, value = self.agent.select_action(state)
            new_power, new_mcs = action
            
            # Validate action outputs
            if not (POWER_MIN <= new_power <= POWER_MAX):
                log_message(f"Invalid power {new_power} for vehicle {veh_id}, clamping to range", LOG_DEBUG_PATH)
                new_power = np.clip(new_power, POWER_MIN, POWER_MAX)
                
            if not (MCS_MIN <= new_mcs <= MCS_MAX):
                log_message(f"Invalid MCS {new_mcs} for vehicle {veh_id}, clamping to range", LOG_DEBUG_PATH)
                new_mcs = np.clip(int(new_mcs), MCS_MIN, MCS_MAX)
            
            # Prepare response (include default beacon rate)
            response = {
                "transmissionPower": float(new_power),
                "MCS": int(new_mcs),
                "beaconRate": DEFAULT_BEACON_RATE  # Always return default
            }
            
            # Training logic
            if self.training_mode:
                # Estimate next CBR (simplified model)
                cbr_change = (new_power - current_power) * 0.01 + (new_mcs - current_mcs) * 0.005
                next_cbr = np.clip(cbr + cbr_change, 0.0, 1.0)
                
                # Calculate reward using the paper's reward function
                reward = self.agent.calculate_reward(next_cbr, new_power, new_mcs)
                
                # Validate reward
                if not np.isfinite(reward) or np.iscomplex(reward):
                    log_message(f"Invalid reward {reward} for vehicle {veh_id}, using default", LOG_DEBUG_PATH)
                    reward = -10.0
                
                done = bool(next_cbr < CBR_RANGE[0] or next_cbr > CBR_RANGE[1])  # Convert to regular bool
                
                # Store transition
                self.agent.memory.append((
                    state,
                    action.tolist(),  # Convert numpy array to list
                    logprob,
                    float(reward),  # Ensure reward is float
                    [next_cbr],  # next state
                    done
                ))
                
                # Update metrics
                self.agent.performance.update_metrics(
                    state, action, float(reward), [next_cbr], 
                    is_exploration, entropy, value
                )
            
            log_message(f"Vehicle {veh_id}: CBR {cbr:.3f}, Power {current_power:.3f}->{new_power:.3f}, MCS {current_mcs}->{new_mcs}, Action: [{action[0]:.3f}, {action[1]:.3f}]", LOG_DEBUG_PATH)
            log_message(f"SENDING to {veh_id}: Power={new_power:.6f}, MCS={new_mcs}, BeaconRate={DEFAULT_BEACON_RATE}", LOG_DEBUG_PATH)
            return response
            
        except Exception as e:
            log_message(f"Error processing vehicle {veh_id}: {e}", LOG_DEBUG_PATH, print_stdout=True)
            # Return safe default values
            return {
                "transmissionPower": 20.0,
                "MCS": 5,
                "beaconRate": DEFAULT_BEACON_RATE
            }

    def stop(self):
        """Stop the server and save final results"""
        log_message("Stopping PPO server...", print_stdout=True)
        self.running = False
        
        try:
            self.server.close()
        except:
            pass
        
        if self.training_mode:
            self.agent.save_model(MODEL_SAVE_PATH)
            # Force save final performance data
            self.agent.performance.save_to_excel()
            log_message("Final performance data saved", print_stdout=True)
        
        # Close TensorBoard writer
        self.agent.performance.writer.close()
        log_message("PPO Server stopped", print_stdout=True)

# ================== Main Execution ==================
def main():
    # Validate operation mode
    if OPERATION_MODE.upper() not in ["TRAINING", "TESTING"]:
        print(f"ERROR: Invalid OPERATION_MODE '{OPERATION_MODE}'. Must be 'TRAINING' or 'TESTING'")
        sys.exit(1)
    
    training_mode = (OPERATION_MODE.upper() == "TRAINING")
    
    print("="*60)
    print(f"PPO VANET SERVER")
    print(f"Host: {HOST}:{PORT}")
    print(f"Mode: {OPERATION_MODE.upper()}")
    if training_mode:
        print(f"Learning Rate: {LEARNING_RATE}")
        print(f"Batch Size: {BATCH_SIZE}")
        print(f"Model will be saved every {MODEL_SAVE_INTERVAL} episodes")
    else:
        print(f"Using pre-trained model: {MODEL_SAVE_PATH}")
    print("="*60)
    
    # Initialize server
    rl_server = RLServer(HOST, PORT, training_mode=training_mode)
    
    try:
        rl_server.start()
    except KeyboardInterrupt:
        log_message("Received interrupt signal", print_stdout=True)
    finally:
        rl_server.stop()

if __name__ == "__main__":
    main()