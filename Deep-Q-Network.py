"""
DUAL-AGENT DEEP Q-NETWORK (DQN) VANET SERVER
============================================

Complete DQN implementation with:
- Neural networks instead of Q-tables
- Continuous state space (no discretization)
- Experience replay buffer
- Target networks for stability
- Dual-agent architecture (MAC + PHY)
- Improved reward functions with quadratic penalties
- Evidence-based density categorization
- Sectoral antenna support

Key improvements over Q-table version:
- Better generalization across similar states
- More sample efficient learning
- Handles continuous state space naturally
- Advanced DQN techniques (double DQN, target networks)
"""

import socket
import threading
import numpy as np
import json
import random
import os
import time
import math
from datetime import datetime
from collections import defaultdict, deque
import pandas as pd
from scipy.stats import entropy
import sys
import logging

# DQN specific imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ================== CONFIGURATION ==================
OPERATION_MODE = "TRAINING"        # Options: "TRAINING" or "TESTING"
ANTENNA_TYPE = "SECTORAL"          # Options: "SECTORAL" or "OMNIDIRECTIONAL"

# ================== Constants ==================
CBR_TARGET = 0.4                   # Better latency/PDR performance
CBR_RANGE = (0.35, 0.45)          # Acceptable CBR range
SINR_TARGET = 12.0                 # Fixed SINR target
SINR_GOOD_THRESHOLD = 12.0         # Threshold for diminishing returns

# DQN Hyperparameters
LEARNING_RATE = 0.001              # Neural network learning rate
GAMMA = 0.95                       # Discount factor
EPSILON = 1.0                      # Initial exploration rate
EPSILON_DECAY = 0.9995             # Exploration decay
MIN_EPSILON = 0.1                  # Minimum exploration
BATCH_SIZE = 64                    # Replay buffer batch size
REPLAY_BUFFER_SIZE = 100000        # Experience replay buffer size
TARGET_UPDATE_FREQ = 1000          # Target network update frequency
MEMORY_MIN_SIZE = 1000             # Minimum experiences before training

HOST = '127.0.0.1'
PORT = 5000

# Parameter ranges (same as before)
POWER_MIN = 1
POWER_MAX = 30
BEACON_MIN = 1
BEACON_MAX = 20

# File paths
MODEL_PREFIX = f"{ANTENNA_TYPE.lower()}_dual_agent_dqn"
MAC_MODEL_PATH = f'{MODEL_PREFIX}_mac_model.pth'
PHY_MODEL_PATH = f'{MODEL_PREFIX}_phy_model.pth'
PERFORMANCE_LOG_PATH = f'{MODEL_PREFIX}_performance.xlsx'
MODEL_SAVE_INTERVAL = 50
PERFORMANCE_LOG_INTERVAL = 10

# ================== Action Spaces (Same as before) ==================
MAC_ACTIONS = [
    (0, 0), (1, 0), (-1, 0), (2, 0), (-2, 0), (3, 0), (-3, 0), (5, 0), (-5, 0),
    (0, 1), (0, -1), (0, 2), (0, -2), (1, 1), (1, -1), (-1, 1), (-1, -1),
    (2, 1), (-2, -1), (10, 0), (-10, 0), (0, 5), (0, -5),
]

PHY_ACTIONS = [0, 1, -1, 2, -2, 3, -3, 5, -5, 10, -10, 15, -15]

MAC_ACTION_DIM = len(MAC_ACTIONS)
PHY_ACTION_DIM = len(PHY_ACTIONS)

# State dimensions (continuous now)
MAC_STATE_DIM = 5  # [CBR, SINR, beacon, MCS, neighbors]
PHY_STATE_DIM = 4  # [CBR, SINR, power, neighbors]

# ================== Logging Setup ==================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_DEBUG_PATH = os.path.join(LOG_DIR, f'{MODEL_PREFIX}_debug.log')

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_DEBUG_PATH, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("DUAL-AGENT DQN INITIALIZED")
logger.info(f"CBR Target: {CBR_TARGET}")
logger.info(f"SINR Target: {SINR_TARGET} dB (fixed)")
logger.info(f"Antenna Type: {ANTENNA_TYPE}")
logger.info(f"Using PyTorch: {torch.__version__}")

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ================== Helper Functions (Keep from previous version) ==================
def get_neighbor_category(neighbor_count, antenna_type="OMNIDIRECTIONAL"):
    """Evidence-based density categorization"""
    if antenna_type.upper() == "SECTORAL":
        if neighbor_count <= 13:
            return "LOW"
        elif neighbor_count <= 26:
            return "MEDIUM"
        elif neighbor_count <= 40:
            return "HIGH"
        else:
            return "VERY_HIGH"
    else:
        if neighbor_count <= 10:
            return "LOW"
        elif neighbor_count <= 20:
            return "MEDIUM"
        elif neighbor_count <= 30:
            return "HIGH"
        else:
            return "VERY_HIGH"

def get_expected_sinr_range(neighbor_count, antenna_type="OMNIDIRECTIONAL"):
    """Get expected SINR range based on density category and antenna type"""
    category = get_neighbor_category(neighbor_count, antenna_type)
    
    if antenna_type.upper() == "SECTORAL":
        ranges = {
            "LOW": (18, 35),
            "MEDIUM": (11, 25),
            "HIGH": (5, 17),
            "VERY_HIGH": (-2, 13)
        }
    else:
        ranges = {
            "LOW": (15, 30),
            "MEDIUM": (8, 20),
            "HIGH": (2, 12),
            "VERY_HIGH": (-5, 8)
        }
    
    return ranges.get(category, (8, 20))

def get_density_multiplier(neighbor_count, antenna_type="OMNIDIRECTIONAL"):
    """Density-based reward multiplier"""
    category = get_neighbor_category(neighbor_count, antenna_type)
    multipliers = {
        "LOW": 0.8,
        "MEDIUM": 1.0,
        "HIGH": 1.4,
        "VERY_HIGH": 1.8
    }
    return multipliers.get(category, 1.0)

# ================== Improved Reward Functions ==================
def calculate_sinr_reward(sinr, power_norm, neighbors, antenna_type="OMNIDIRECTIONAL"):
    """Improved SINR reward with quadratic penalties and logarithmic scaling"""
    SINR_TARGET = 12.0
    
    # Phase 1: Below target
    if sinr < SINR_TARGET:
        base_reward = 10.0 * (sinr / SINR_TARGET)
        # Logarithmic neighbor penalty
        neighbor_penalty = -2.0 * math.log(1 + neighbors) / math.log(1 + 30)
        sinr_reward = base_reward + neighbor_penalty
    else:
        # Phase 2: Above target - diminishing returns
        base_reward = 10.0
        excess_sinr = sinr - SINR_TARGET
        diminishing_reward = 5.0 * math.sqrt(excess_sinr / 10.0)
        sinr_reward = base_reward + diminishing_reward
        sinr_reward = min(sinr_reward, 18.0)
    
    # Quadratic power penalty when SINR is sufficient
    if sinr >= SINR_TARGET:
        if power_norm > 0.6:
            power_penalty = -8.0 * ((power_norm - 0.6) ** 2)
            sinr_reward += power_penalty
        elif power_norm <= 0.4:
            efficiency_bonus = 4.0 * ((0.4 - power_norm) ** 2)
            sinr_reward += efficiency_bonus
    
    # Logarithmic neighbor impact for excessive SINR
    if sinr > 20.0:
        neighbor_impact_penalty = -3.0 * math.log(1 + neighbors) / math.log(1 + 50) * (sinr - 20.0) / 10.0
        sinr_reward += neighbor_impact_penalty
    
    return np.clip(sinr_reward, -15, 20)

# ================== DQN Neural Networks ==================
class DQN(nn.Module):
    """Deep Q-Network architecture"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256, 128]):
        super(DQN, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, state):
        return self.network(state)

# ================== Experience Replay Buffer ==================
class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(states).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(next_states).to(device),
            torch.BoolTensor(dones).to(device)
        )
    
    def __len__(self):
        return len(self.buffer)

# ================== DQN Agent Base Class ==================
class DQNAgent:
    """Base DQN agent with common functionality"""
    
    def __init__(self, state_dim, action_dim, lr=LEARNING_RATE):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = EPSILON
        
        # Networks
        self.q_network = DQN(state_dim, action_dim).to(device)
        self.target_network = DQN(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.memory = ReplayBuffer(REPLAY_BUFFER_SIZE)
        
        # Update target network
        self.update_target_network()
        self.training_step = 0
        
        logger.info(f"Initialized DQN Agent - State: {state_dim}, Actions: {action_dim}")
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def normalize_state(self, state):
        """Normalize state values for better neural network performance"""
        # Override in subclasses
        return np.array(state, dtype=np.float32)
    
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor)
            return q_values.max(1)[1].item()
    
    def store_experience(self, state, action, reward, next_state, done=False):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self):
        """Train the DQN using experience replay"""
        if len(self.memory) < MEMORY_MIN_SIZE:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            # Double DQN: use main network to select action, target network to evaluate
            next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (GAMMA * next_q_values * ~dones.unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % TARGET_UPDATE_FREQ == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY
        
        return loss.item()
    
    def save_model(self, filepath):
        """Save model weights"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filepath)
    
    def load_model(self, filepath):
        """Load model weights"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', MIN_EPSILON)
            self.training_step = checkpoint.get('training_step', 0)
            logger.info(f"Loaded model from {filepath}")
            return True
        return False

# ================== MAC DQN Agent ==================
class MACDQNAgent(DQNAgent):
    """MAC Agent using DQN - Controls beacon rate and MCS"""
    
    def __init__(self):
        super().__init__(MAC_STATE_DIM, MAC_ACTION_DIM)
    
    def normalize_state(self, cbr, sinr, beacon, mcs, neighbors):
        """Normalize MAC state for neural network"""
        # Normalize to roughly [-1, 1] or [0, 1] range
        normalized_state = np.array([
            cbr,                                    # Already 0-1
            sinr / 50.0,                           # 0-50 → 0-1
            (beacon - BEACON_MIN) / (BEACON_MAX - BEACON_MIN),  # 1-20 → 0-1
            mcs / 9.0,                             # 0-9 → 0-1
            min(neighbors / 50.0, 1.0)             # Cap at 50 neighbors
        ], dtype=np.float32)
        
        return normalized_state
    
    def select_action(self, state_indices, neighbor_count, antenna_type, training=True):
        """Select MAC action with density awareness"""
        # Use base DQN action selection
        action_idx = super().select_action(state_indices, training)
        
        # Apply density-aware exploration during training
        if training and random.random() < 0.1:  # 10% chance for guided exploration
            density_category = get_neighbor_category(neighbor_count, antenna_type)
            
            if density_category in ["HIGH", "VERY_HIGH"]:
                # Prefer conservative beacon actions
                conservative_actions = [i for i, (b, m) in enumerate(MAC_ACTIONS) if b <= 1]
                if conservative_actions:
                    action_idx = random.choice(conservative_actions)
            elif density_category == "LOW":
                # Can use higher beacon rates
                aggressive_actions = [i for i, (b, m) in enumerate(MAC_ACTIONS) if b >= -1]
                if aggressive_actions:
                    action_idx = random.choice(aggressive_actions)
        
        return action_idx
    
    def calculate_reward(self, cbr, sinr, beacon, mcs, neighbors, next_cbr, next_beacon, next_mcs, antenna_type):
        """MAC reward with improved mathematical formulations"""
        # Primary CBR optimization
        cbr_error = abs(cbr - CBR_TARGET)
        cbr_reward = 10.0 * (1 - math.tanh(25 * cbr_error))
        
        # Logarithmic beacon optimization
        optimal_beacon_factor = 1.0 - (0.3 * math.log(1 + neighbors) / math.log(1 + 40))
        optimal_beacon = 12.0 * optimal_beacon_factor
        
        if antenna_type.upper() == "SECTORAL":
            optimal_beacon *= 1.1
        
        # Quadratic beacon penalty
        beacon_error = abs(beacon - optimal_beacon)
        beacon_reward = -2.0 * (beacon_error / 5.0) ** 2
        
        # MCS optimization with logarithmic neighbor consideration
        optimal_mcs_factor = 1.0 - (0.4 * math.log(1 + neighbors) / math.log(1 + 50))
        optimal_mcs = 8.0 * optimal_mcs_factor
        
        if antenna_type.upper() == "SECTORAL":
            optimal_mcs *= 1.2
        
        # Quadratic MCS penalty
        mcs_error = abs(mcs - optimal_mcs)
        mcs_reward = -1.5 * (mcs_error / 3.0) ** 2
        
        # Logarithmic smoothness penalty
        beacon_change = abs(next_beacon - beacon)
        mcs_change = abs(next_mcs - mcs)
        smoothness_penalty = -1.0 * (math.log(1 + beacon_change) + math.log(1 + mcs_change))
        
        total_reward = cbr_reward + beacon_reward + mcs_reward + smoothness_penalty
        
        return np.clip(total_reward, -20, 20)

# ================== PHY DQN Agent ==================
class PHYDQNAgent(DQNAgent):
    """PHY Agent using DQN - Controls transmission power"""
    
    def __init__(self):
        super().__init__(PHY_STATE_DIM, PHY_ACTION_DIM)
    
    def normalize_state(self, cbr, sinr, power, neighbors):
        """Normalize PHY state for neural network"""
        normalized_state = np.array([
            cbr,                                    # Already 0-1
            sinr / 50.0,                           # 0-50 → 0-1
            (power - POWER_MIN) / (POWER_MAX - POWER_MIN),  # 1-30 → 0-1
            min(neighbors / 50.0, 1.0)             # Cap at 50 neighbors
        ], dtype=np.float32)
        
        return normalized_state
    
    def select_action(self, state_indices, neighbor_count, current_sinr, antenna_type, training=True):
        """Select PHY action with SINR awareness"""
        action_idx = super().select_action(state_indices, training)
        
        # SINR-aware exploration during training
        if training and random.random() < 0.1:
            expected_min, expected_max = get_expected_sinr_range(neighbor_count, antenna_type)
            
            if current_sinr < expected_min:
                # Need more power
                power_up_actions = [i for i, p in enumerate(PHY_ACTIONS) if p >= 1]
                if power_up_actions:
                    action_idx = random.choice(power_up_actions)
            elif current_sinr > expected_max:
                # Can reduce power
                power_down_actions = [i for i, p in enumerate(PHY_ACTIONS) if p <= -1]
                if power_down_actions:
                    action_idx = random.choice(power_down_actions)
        
        return action_idx
    
    def calculate_reward(self, cbr, sinr, power, neighbors, next_sinr, next_power, antenna_type):
        """PHY reward with improved mathematical formulations"""
        power_norm = (power - POWER_MIN) / (POWER_MAX - POWER_MIN)
        
        # Primary SINR reward
        sinr_reward = calculate_sinr_reward(sinr, power_norm, neighbors, antenna_type)
        
        # Power efficiency with logarithmic neighbor consideration
        base_power_need = 0.3 + (0.4 * math.log(1 + neighbors) / math.log(1 + 40))
        
        if antenna_type.upper() == "SECTORAL":
            base_power_need *= 0.8
        
        # Quadratic power efficiency reward/penalty
        power_deviation = power_norm - base_power_need
        
        if abs(power_deviation) <= 0.1:
            power_efficiency_reward = 3.0
        else:
            power_efficiency_reward = -4.0 * (power_deviation ** 2)
        
        # Logarithmic neighbor impact penalty
        if power_norm > 0.7:
            neighbor_impact_penalty = -2.0 * math.log(1 + neighbors) / math.log(1 + 30) * (power_norm - 0.7) ** 2
            power_efficiency_reward += neighbor_impact_penalty
        
        # Logarithmic smoothness penalty
        power_change = abs(next_power - power)
        smoothness_penalty = -0.5 * math.log(1 + power_change)
        
        total_reward = sinr_reward + power_efficiency_reward + smoothness_penalty
        
        return np.clip(total_reward, -20, 20)

# ================== Performance Tracking (Updated) ==================
class DualAgentDQNPerformanceMetrics:
    def __init__(self):
        self.reset_metrics()
        self.episode_data = []
        
    def reset_metrics(self):
        self.mac_rewards = []
        self.phy_rewards = []
        self.joint_rewards = []
        self.cbr_values = []
        self.sinr_values = []
        self.mac_losses = []
        self.phy_losses = []
        self.neighbor_counts = []
        self.mac_actions = []
        self.phy_actions = []
        
    def add_step(self, mac_reward, phy_reward, cbr, sinr, neighbors, mac_action, phy_action, mac_loss=0, phy_loss=0):
        self.mac_rewards.append(mac_reward)
        self.phy_rewards.append(phy_reward)
        self.joint_rewards.append(0.5 * mac_reward + 0.5 * phy_reward)
        self.cbr_values.append(cbr)
        self.sinr_values.append(sinr)
        self.neighbor_counts.append(neighbors)
        self.mac_actions.append(mac_action)
        self.phy_actions.append(phy_action)
        self.mac_losses.append(mac_loss)
        self.phy_losses.append(phy_loss)
    
    def calculate_episode_metrics(self, episode_num):
        """Calculate episode statistics"""
        if not self.mac_rewards:
            return {}
            
        metrics = {
            'episode': episode_num,
            'timestamp': datetime.now(),
            'total_steps': len(self.mac_rewards),
            
            # Reward metrics
            'avg_mac_reward': np.mean(self.mac_rewards),
            'avg_phy_reward': np.mean(self.phy_rewards),
            'avg_joint_reward': np.mean(self.joint_rewards),
            'cumulative_joint_reward': sum(self.joint_rewards),
            
            # Performance metrics
            'avg_cbr': np.mean(self.cbr_values),
            'cbr_in_range_rate': sum(1 for cbr in self.cbr_values if CBR_RANGE[0] <= cbr <= CBR_RANGE[1]) / len(self.cbr_values),
            'avg_sinr': np.mean(self.sinr_values),
            'sinr_above_12_rate': sum(1 for sinr in self.sinr_values if sinr >= 12) / len(self.sinr_values),
            
            # DQN specific metrics
            'avg_mac_loss': np.mean(self.mac_losses) if self.mac_losses else 0,
            'avg_phy_loss': np.mean(self.phy_losses) if self.phy_losses else 0,
            
            # Density analysis
            'avg_neighbors': np.mean(self.neighbor_counts),
            'low_density_rate': sum(1 for n in self.neighbor_counts if get_neighbor_category(n, ANTENNA_TYPE) == "LOW") / len(self.neighbor_counts),
            'medium_density_rate': sum(1 for n in self.neighbor_counts if get_neighbor_category(n, ANTENNA_TYPE) == "MEDIUM") / len(self.neighbor_counts),
            'high_density_rate': sum(1 for n in self.neighbor_counts if get_neighbor_category(n, ANTENNA_TYPE) == "HIGH") / len(self.neighbor_counts),
            'very_high_density_rate': sum(1 for n in self.neighbor_counts if get_neighbor_category(n, ANTENNA_TYPE) == "VERY_HIGH") / len(self.neighbor_counts),
            
            # Action diversity
            'mac_action_entropy': entropy(np.bincount(self.mac_actions, minlength=MAC_ACTION_DIM)),
            'phy_action_entropy': entropy(np.bincount(self.phy_actions, minlength=PHY_ACTION_DIM)),
        }
        
        return metrics
    
    def log_performance(self, episode_num):
        """Log and save performance metrics"""
        metrics = self.calculate_episode_metrics(episode_num)
        if metrics:
            self.episode_data.append(metrics)
            
            if episode_num % PERFORMANCE_LOG_INTERVAL == 0:
                self.save_to_excel()
        return metrics
    
    def save_to_excel(self):
        """Save performance data to Excel"""
        try:
            with pd.ExcelWriter(PERFORMANCE_LOG_PATH, engine='openpyxl', mode='w') as writer:
                if self.episode_data:
                    episode_df = pd.DataFrame(self.episode_data)
                    episode_df.to_excel(writer, sheet_name='Episode_Summary', index=False)
                
                # DQN analysis
                analysis_data = {
                    'Metric': ['MAC Avg Reward', 'PHY Avg Reward', 'Joint Avg Reward',
                               'CBR Performance', 'SINR Performance', 'MAC Training Loss',
                               'PHY Training Loss', 'Action Diversity (MAC)', 'Action Diversity (PHY)'],
                    'Value': [
                        np.mean([d['avg_mac_reward'] for d in self.episode_data[-10:]]),
                        np.mean([d['avg_phy_reward'] for d in self.episode_data[-10:]]),
                        np.mean([d['avg_joint_reward'] for d in self.episode_data[-10:]]),
                        np.mean([d['cbr_in_range_rate'] for d in self.episode_data[-10:]]),
                        np.mean([d['sinr_above_12_rate'] for d in self.episode_data[-10:]]),
                        np.mean([d['avg_mac_loss'] for d in self.episode_data[-10:]]),
                        np.mean([d['avg_phy_loss'] for d in self.episode_data[-10:]]),
                        np.mean([d['mac_action_entropy'] for d in self.episode_data[-10:]]),
                        np.mean([d['phy_action_entropy'] for d in self.episode_data[-10:]])
                    ]
                }
                analysis_df = pd.DataFrame(analysis_data)
                analysis_df.to_excel(writer, sheet_name='DQN_Analysis', index=False)
                
            logger.info(f"Performance data saved to {PERFORMANCE_LOG_PATH}")
            
        except Exception as e:
            logger.error(f"Error saving to Excel: {e}")

# ================== Dual-Agent DQN Implementation ==================
class DualAgentDQN:
    def __init__(self, training_mode=True):
        self.training_mode = training_mode
        self.mac_agent = MACDQNAgent()
        self.phy_agent = PHYDQNAgent()
        self.performance = DualAgentDQNPerformanceMetrics()
        self.episode_count = 0
        
        # Load pre-trained models if they exist
        self.load_models()
        
        # Set agents to evaluation mode if testing
        if not training_mode:
            self.mac_agent.epsilon = 0.0
            self.phy_agent.epsilon = 0.0
    
    def process_vehicle(self, veh_id, veh_info):
        """Process vehicle with DQN agents"""
        try:
            # Extract current state
            cbr = float(veh_info.get("CBR", 0.4))
            sinr = float(veh_info.get("SINR", veh_info.get("SNR", 20)))
            neighbors = int(veh_info.get("neighbors", 10))
            current_power = float(veh_info.get("transmissionPower", 15))
            current_beacon = float(veh_info.get("beaconRate", 10))
            current_mcs = int(veh_info.get("MCS", 5))
            
            antenna_type = ANTENNA_TYPE
            
            # Validate and clamp inputs
            current_power = np.clip(current_power, POWER_MIN, POWER_MAX)
            current_beacon = np.clip(current_beacon, BEACON_MIN, BEACON_MAX)
            current_mcs = np.clip(current_mcs, 0, 9)
            cbr = np.clip(cbr, 0.0, 1.0)
            sinr = np.clip(sinr, 0, 50)
            neighbors = max(0, neighbors)
            
            # Handle NaN values
            if not np.isfinite(cbr):
                cbr = 0.4
            if not np.isfinite(sinr):
                sinr = 15.0
            if not np.isfinite(current_power):
                current_power = 15.0
            if not np.isfinite(current_beacon):
                current_beacon = 10.0
            
            # Normalize states for neural networks
            mac_state = self.mac_agent.normalize_state(cbr, sinr, current_beacon, current_mcs, neighbors)
            phy_state = self.phy_agent.normalize_state(cbr, sinr, current_power, neighbors)
            
            # Select actions
            mac_action_idx = self.mac_agent.select_action(mac_state, neighbors, antenna_type, self.training_mode)
            phy_action_idx = self.phy_agent.select_action(phy_state, neighbors, sinr, antenna_type, self.training_mode)
            
            # Apply actions
            beacon_delta, mcs_delta = MAC_ACTIONS[mac_action_idx]
            power_delta = PHY_ACTIONS[phy_action_idx]
            
            new_beacon = np.clip(current_beacon + beacon_delta, BEACON_MIN, BEACON_MAX)
            new_mcs = np.clip(current_mcs + mcs_delta, 0, 9)
            new_power = np.clip(current_power + power_delta, POWER_MIN, POWER_MAX)
            
            # Training updates
            if self.training_mode:
                # Simulate next state (simplified)
                next_cbr = cbr + random.uniform(-0.05, 0.05)
                next_cbr = np.clip(next_cbr, 0, 1)
                next_sinr = sinr + random.uniform(-2, 2)
                
                # Calculate rewards
                mac_reward = self.mac_agent.calculate_reward(
                    cbr, sinr, current_beacon, current_mcs, neighbors,
                    next_cbr, new_beacon, new_mcs, antenna_type
                )
                
                phy_reward = self.phy_agent.calculate_reward(
                    cbr, sinr, current_power, neighbors,
                    next_sinr, new_power, antenna_type
                )
                
                # Next states
                next_mac_state = self.mac_agent.normalize_state(next_cbr, next_sinr, new_beacon, new_mcs, neighbors)
                next_phy_state = self.phy_agent.normalize_state(next_cbr, next_sinr, new_power, neighbors)
                
                # Store experiences
                self.mac_agent.store_experience(mac_state, mac_action_idx, mac_reward, next_mac_state)
                self.phy_agent.store_experience(phy_state, phy_action_idx, phy_reward, next_phy_state)
                
                # Train networks
                mac_loss = self.mac_agent.train()
                phy_loss = self.phy_agent.train()
                
                # Track performance
                self.performance.add_step(
                    mac_reward, phy_reward, cbr, sinr, neighbors,
                    mac_action_idx, phy_action_idx, mac_loss, phy_loss
                )
            
            # Enhanced logging
            if veh_id.endswith('0'):
                density_cat = get_neighbor_category(neighbors, antenna_type)
                expected_sinr = get_expected_sinr_range(neighbors, antenna_type)
                logger.info(f"Vehicle {veh_id} [{antenna_type}][{density_cat}]: "
                           f"CBR={cbr:.3f}, SINR={sinr:.1f}dB, Neighbors={neighbors}")
                logger.info(f"  Expected SINR: {expected_sinr[0]}-{expected_sinr[1]} dB")
                logger.info(f"  MAC: Beacon {current_beacon:.0f}->{new_beacon:.0f}Hz, MCS {current_mcs}->{new_mcs}")
                logger.info(f"  PHY: Power {current_power:.0f}->{new_power:.0f}dBm")
                if self.training_mode:
                    logger.info(f"  Epsilon: MAC={self.mac_agent.epsilon:.3f}, PHY={self.phy_agent.epsilon:.3f}")
            
            return {
                "transmissionPower": int(new_power),
                "beaconRate": int(new_beacon),
                "MCS": int(new_mcs)
            }
            
        except Exception as e:
            logger.error(f"Error processing vehicle {veh_id}: {e}")
            return {
                "transmissionPower": 15,
                "beaconRate": 10,
                "MCS": 5
            }
    
    def end_episode(self):
        """End current episode and save metrics"""
        if self.training_mode:
            self.episode_count += 1
            metrics = self.performance.log_performance(self.episode_count)
            
            if metrics:
                logger.info(f"Episode {self.episode_count}: "
                           f"MAC reward={metrics['avg_mac_reward']:.3f}, "
                           f"PHY reward={metrics['avg_phy_reward']:.3f}, "
                           f"CBR in range={metrics['cbr_in_range_rate']:.2%}, "
                           f"Losses: MAC={metrics['avg_mac_loss']:.4f}, PHY={metrics['avg_phy_loss']:.4f}")
            
            if self.episode_count % MODEL_SAVE_INTERVAL == 0:
                self.save_models()
            
            self.performance.reset_metrics()
    
    def save_models(self):
        """Save DQN models"""
        try:
            self.mac_agent.save_model(MAC_MODEL_PATH)
            self.phy_agent.save_model(PHY_MODEL_PATH)
            logger.info(f"Models saved: {MAC_MODEL_PATH}, {PHY_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Load pre-trained DQN models"""
        try:
            if self.mac_agent.load_model(MAC_MODEL_PATH):
                logger.info("MAC DQN model loaded successfully")
            if self.phy_agent.load_model(PHY_MODEL_PATH):
                logger.info("PHY DQN model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")

# ================== Server Implementation (Same structure) ==================
class DualAgentDQNServer:
    def __init__(self, host, port, training_mode=True):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(5)
        self.dual_agent = DualAgentDQN(training_mode=training_mode)
        self.training_mode = training_mode
        self.running = True
        
        mode_str = "TRAINING" if training_mode else "TESTING"
        logger.info(f"Dual-Agent DQN Server started in {mode_str} mode on {host}:{port}")

    def receive_message_with_header(self, conn):
        try:
            header_data = b''
            while len(header_data) < 4:
                chunk = conn.recv(4 - len(header_data))
                if not chunk:
                    return None
                header_data += chunk
            
            message_length = int.from_bytes(header_data, byteorder='little')
            
            message_data = b''
            while len(message_data) < message_length:
                chunk = conn.recv(min(message_length - len(message_data), 8192))
                if not chunk:
                    return None
                message_data += chunk
            
            return message_data.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None

    def send_message_with_header(self, conn, message):
        try:
            message_bytes = message.encode('utf-8')
            message_length = len(message_bytes)
            
            header = message_length.to_bytes(4, byteorder='little')
            conn.sendall(header)
            conn.sendall(message_bytes)
            return True
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False

    def handle_client(self, conn, addr):
        logger.info(f"Client connected from {addr}")
        
        try:
            while self.running:
                message_str = self.receive_message_with_header(conn)
                if not message_str:
                    break
                
                try:
                    batch_data = json.loads(message_str)
                    logger.info(f"Processing {len(batch_data)} vehicles")
                    
                    responses = {}
                    
                    for veh_id, veh_info in batch_data.items():
                        response = self.dual_agent.process_vehicle(veh_id, veh_info)
                        if response:
                            responses[veh_id] = response
                    
                    response_dict = {"vehicles": responses}
                    response_str = json.dumps(response_dict)
                    
                    if self.send_message_with_header(conn, response_str):
                        logger.info(f"Sent response to {addr}: {len(responses)} vehicles")
                    else:
                        break
                    
                    if self.training_mode and len(self.dual_agent.performance.mac_rewards) >= 100:
                        self.dual_agent.end_episode()
                
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error processing vehicle data: {e}")
                    break
        
        except Exception as e:
            logger.error(f"Error in handle_client: {e}")
        finally:
            try:
                conn.close()
            except:
                pass
            logger.info(f"Client {addr} disconnected")

    def start(self):
        try:
            logger.info("Dual-Agent DQN Server listening for connections...")
            while self.running:
                try:
                    conn, addr = self.server.accept()
                    client_thread = threading.Thread(
                        target=self.handle_client, 
                        args=(conn, addr),
                        daemon=True
                    )
                    client_thread.start()
                except Exception as e:
                    if self.running:
                        logger.error(f"Error accepting connection: {e}")
        except Exception as e:
            if self.running:
                logger.error(f"Server error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        logger.info("Stopping Dual-Agent DQN server...")
        self.running = False
        
        try:
            self.server.close()
        except:
            pass
        
        if self.training_mode:
            self.dual_agent.save_models()
            self.dual_agent.performance.save_to_excel()
            logger.info("Final models and performance data saved")
        
        logger.info("Dual-Agent DQN server stopped")

# ================== Main Execution ==================
def main():
    if OPERATION_MODE.upper() not in ["TRAINING", "TESTING"]:
        print(f"ERROR: Invalid OPERATION_MODE '{OPERATION_MODE}'. Must be 'TRAINING' or 'TESTING'")
        sys.exit(1)
    
    if ANTENNA_TYPE.upper() not in ["SECTORAL", "OMNIDIRECTIONAL"]:
        print(f"ERROR: Invalid ANTENNA_TYPE '{ANTENNA_TYPE}'. Must be 'SECTORAL' or 'OMNIDIRECTIONAL'")
        sys.exit(1)
    
    training_mode = (OPERATION_MODE.upper() == "TRAINING")
    
    print("="*80)
    print(" DUAL-AGENT DEEP Q-NETWORK (DQN) VANET SERVER")
    print(f"Host: {HOST}:{PORT}")
    print(f"Mode: {OPERATION_MODE.upper()}")
    print(f"Antenna Type: {ANTENNA_TYPE.upper()}")
    print(f"Device: {device}")
    print("="*40)
    print("DQN ARCHITECTURE:")
    print("  • MAC DQN: [CBR, SINR, beacon, MCS, neighbors] → beacon/MCS actions")
    print("  • PHY DQN: [CBR, SINR, power, neighbors] → power actions")
    print("  • Neural Networks: 256→256→128 hidden layers")
    print("  • Experience Replay: 100k buffer, batch size 64")
    print("  • Target Networks: Updated every 1000 steps")
    print("  • Double DQN: Reduced overestimation bias")
    print("="*40)
    print("IMPROVEMENTS OVER Q-TABLE:")
    print("  ✓ Continuous state space (no discretization loss)")
    print("  ✓ Better generalization between similar states")
    print("  ✓ More sample efficient learning")
    print("  ✓ Advanced DQN techniques")
    print("  ✓ GPU acceleration support")
    print("="*40)
    print(f"CBR Target: {CBR_TARGET} (optimized for latency/PDR)")
    print(f"SINR Target: {SINR_TARGET} dB (fixed)")
    print("Reward Functions: Quadratic penalties + logarithmic scaling")
    if training_mode:
        print(f"Learning Rate: {LEARNING_RATE}")
        print(f"Initial Epsilon: {EPSILON}")
        print(f"Models will be saved every {MODEL_SAVE_INTERVAL} episodes")
    else:
        print("Using pre-trained DQN models")
    print("="*80)
    
    # Initialize server
    dqn_server = DualAgentDQNServer(HOST, PORT, training_mode=training_mode)
    
    try:
        dqn_server.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        dqn_server.stop()

if __name__ == "__main__":
    main()