"""
FIXED CBR-Centered Q-Learning VANET Server
==========================================

This script implements a PROPERLY DESIGNED Q-learning reinforcement learning server for VANET optimization.
The reward function is now CORRECTLY designed to target CBR = 0.65.

QUICK START:
1. For TRAINING: Set OPERATION_MODE = "TRAINING" below
2. For TESTING: Set OPERATION_MODE = "TESTING" below  
3. Configure CBR settings in the CONFIGURATION section
4. Run: python qlearning_server.py

CRITICAL FIXES:
- Reward function now PROPERLY targets CBR = 0.65
- Massive penalties for extreme CBR values (like 0.995)
- Rewards for moving closer to target from ANY direction
- Fixed exploration to avoid high-power bias

OUTPUT FILES:
- q_learning_model.npy: Trained Q-table model
- performance_results.xlsx: Comprehensive performance analysis (4 sheets)
- logs/: Directory with detailed debug logs
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
from collections import defaultdict
import pandas as pd
from scipy.stats import entropy
import sys

# ================== MAIN CONFIGURATION ==================
# CHANGE THIS TO SWITCH BETWEEN MODES
OPERATION_MODE = "TRAINING"        # Options: "TRAINING" or "TESTING"

# ================== CBR CONFIGURATION ==================
#  MAIN CBR SETTINGS - Change these to adjust CBR behavior
CBR_TARGET = 0.65                 #  Primary target CBR (0.0 to 1.0)
CBR_TOLERANCE = 0.05              # ±tolerance around target for acceptable range
CBR_EXTREME_HIGH = 0.9            # CBR above this gets massive penalties
CBR_HIGH_WARNING = 0.8            # CBR above this gets high penalties  
CBR_EXTREME_LOW = 0.3             # CBR below this gets penalties
CBR_WIDE_ACCEPTABLE_TOLERANCE = 0.1  # Wider acceptable range for partial rewards

# Automatically calculated CBR ranges (don't change these)
CBR_RANGE = (CBR_TARGET - CBR_TOLERANCE, CBR_TARGET + CBR_TOLERANCE)
CBR_WIDE_RANGE = (CBR_TARGET - CBR_WIDE_ACCEPTABLE_TOLERANCE, CBR_TARGET + CBR_WIDE_ACCEPTABLE_TOLERANCE)

# File naming based on CBR target (automatically generated)
CBR_TARGET_STR = f"{CBR_TARGET:.2f}".replace('.', '_')
MODEL_SAVE_PATH = f'q_learning_model_cbr_fixed_{CBR_TARGET_STR}.npy'
PERFORMANCE_LOG_PATH = f'performance_results_cbr_fixed_{CBR_TARGET_STR}.xlsx'

# ================== REWARD FUNCTION CONFIGURATION ==================
REWARD_DISTANCE_SCALE = 200       # Scale factor for distance-from-target reward
REWARD_TARGET_ZONE = 50           # Bonus for being in target CBR range
REWARD_WIDE_ZONE = 20             # Bonus for being in wider acceptable range
REWARD_EXTREME_HIGH_PENALTY = 200 # Penalty multiplier for extreme high CBR
REWARD_HIGH_PENALTY = 100         # Penalty multiplier for high CBR
REWARD_LOW_PENALTY = 50           # Penalty multiplier for low CBR
REWARD_STABILITY_SCALE = 0.5      # Scale factor for stability penalty
REWARD_EFFICIENCY_SCALE = 0.2     # Scale factor for efficiency reward

# ================== OTHER CONSTANTS ==================
BUFFER_SIZE = 100000               # Replay buffer size
LEARNING_RATE = 0.15               # Slightly higher learning rate
DISCOUNT_FACTOR = 0.95             # Slightly lower discount for faster learning
EPSILON = 1.0                      # Initial exploration rate
EPSILON_DECAY = 0.9995             # Slow decay for thorough exploration
MIN_EPSILON = 0.1                  # High minimum exploration
HOST = '127.0.0.1'                 # Server IP
PORT = 5000                        # Server port

# Power and beacon ranges
POWER_MIN = 1                      # Minimum power (dBm) 
POWER_MAX = 30                     # Maximum power (dBm)
BEACON_MIN = 1                     # Minimum beacon rate (Hz)
BEACON_MAX = 20                    # Maximum beacon rate (Hz)

# Intervals
MODEL_SAVE_INTERVAL = 50           # Save model every N episodes
PERFORMANCE_LOG_INTERVAL = 10      # Log performance every N episodes

# FIXED: State discretization
CBR_BINS = np.linspace(0.0, 1.0, 21)          # 21 points = 20 bins for finer CBR resolution
NEIGHBORS_BINS = np.linspace(0, 50, 11)       # 11 points = 10 bins for neighbors (0-50)
SNR_BINS = np.linspace(0, 50, 11)             # 11 points = 10 bins for SNR (0-50 dB)
POWER_BINS = np.linspace(POWER_MIN, POWER_MAX, 16)  # 16 points = 15 bins for power

# State dimensions
CBR_STATES = 20    # 20 discrete CBR states (finer resolution)
NEIGHBORS_STATES = 10   # 10 discrete neighbor states  
SNR_STATES = 10     # 10 discrete SNR states
POWER_STATES = 15   # 15 discrete power bins
BEACON_STATES = 20  # 20 discrete beacon levels (1-20)

# Discrete beacon levels
DISCRETE_BEACON_LEVELS = list(range(1, 21))  # [1, 2, 3, ..., 20]

# FIXED: More conservative action space
ACTION_SPACE = [
    (1, 1),    # Action 0: Power +1dBm, Beacon +1 level
    (1, -1),   # Action 1: Power +1dBm, Beacon -1 level
    (1, 0),    # Action 2: Power +1dBm, Beacon no change
    (-1, 1),   # Action 3: Power -1dBm, Beacon +1 level
    (-1, -1),  # Action 4: Power -1dBm, Beacon -1 level
    (-1, 0),   # Action 5: Power -1dBm, Beacon no change
    (2, 2),    # Action 6: Power +2dBm, Beacon +2 levels
    (2, -2),   # Action 7: Power +2dBm, Beacon -2 levels
    (2, 0),    # Action 8: Power +2dBm, Beacon no change
    (-2, 2),   # Action 9: Power -2dBm, Beacon +2 levels
    (-2, -2),  # Action 10: Power -2dBm, Beacon -2 levels
    (-2, 0),   # Action 11: Power -2dBm, Beacon no change
    (3, 1),    # Action 12: Power +3dBm, Beacon +1 level
    (-3, 1),   # Action 13: Power -3dBm, Beacon +1 level
    (3, -1),   # Action 14: Power +3dBm, Beacon -1 level
    (-3, -1),  # Action 15: Power -3dBm, Beacon -1 level
    (0, 3),    # Action 16: Power no change, Beacon +3 levels
    (0, -3),   # Action 17: Power no change, Beacon -3 levels
    (0, 2),    # Action 18: Power no change, Beacon +2 levels
    (0, -2),   # Action 19: Power no change, Beacon -2 levels
    (0, 1),    # Action 20: Power no change, Beacon +1 level
    (0, -1),   # Action 21: Power no change, Beacon -1 level
    (0, 0)     # Action 22: No change
]
ACTION_DIM = len(ACTION_SPACE)

# IEEE 802.11bd MCS mapping (0-9)
SNR_TO_MCS = {
    (0, 5): 0, (5, 10): 1, (10, 15): 2, (15, 20): 3,
    (20, 25): 4, (25, 30): 5, (30, 35): 6, (35, 40): 7,
    (40, 45): 8, (45, float('inf')): 9  # MCS 9 for SNR >= 45 dB
}

# Initialize Q-table
STATE_DIM = (CBR_STATES, NEIGHBORS_STATES, SNR_STATES, POWER_STATES, BEACON_STATES)
q_table = np.zeros(STATE_DIM + (ACTION_DIM,), dtype=np.float32)

# Log Configuration
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_RECEIVED_PATH = os.path.join(LOG_DIR, 'received.log')
LOG_SENT_PATH = os.path.join(LOG_DIR, 'sent.log')
LOG_DEBUG_PATH = os.path.join(LOG_DIR, 'debug.log')

print(f" CBR-CENTERED Q-LEARNING: Q-table shape: {q_table.shape}")
print(f" CBR-CENTERED Q-LEARNING: State dimensions: CBR={CBR_STATES}, Neighbors={NEIGHBORS_STATES}, SNR={SNR_STATES}, Power={POWER_STATES}, Beacon={BEACON_STATES}")
print(f" CBR-CENTERED Q-LEARNING: Target CBR = {CBR_TARGET}")
print(f" CBR-CENTERED Q-LEARNING: CBR Range = {CBR_RANGE} (tolerance = ±{CBR_TOLERANCE})")
print(f" CBR-CENTERED Q-LEARNING: Reward function FIXED to target CBR properly")

# ================== Helper Functions ==================
def get_beacon_index(beacon_value):
    """Get index of discrete beacon level (1-20 -> 0-19)."""
    beacon_value = int(round(float(beacon_value)))
    beacon_value = max(1, min(20, beacon_value))  # Clamp to valid range
    return beacon_value - 1  # Convert to 0-based index

def get_beacon_from_index(beacon_index):
    """Get beacon value from index (0-19 -> 1-20)."""
    beacon_index = max(0, min(len(DISCRETE_BEACON_LEVELS) - 1, beacon_index))
    return DISCRETE_BEACON_LEVELS[beacon_index]

def discretize(value, bins, max_states):
    """Discretize a continuous value into a bin index with proper bounds checking."""
    # Handle NaN or infinite values
    if not np.isfinite(value):
        value = 0.0
    
    # Clamp value to bin range
    value = np.clip(value, bins[0], bins[-1])
    
    # Find appropriate bin
    bin_idx = np.digitize(value, bins) - 1
    
    # Ensure index is within valid range [0, max_states-1]
    return max(0, min(max_states - 1, bin_idx))

def adjust_mcs_based_on_snr(snr):
    """Return MCS level based on SNR thresholds with validation (0-9 range)."""
    try:
        # Validate and clamp SNR input
        snr = float(np.real(snr))
        if not np.isfinite(snr):
            snr = 20.0  # Default SNR
        snr = np.clip(snr, 0, 100)  # Reasonable SNR range
        
        for (snr_min, snr_max), mcs in SNR_TO_MCS.items():
            if snr_min <= snr < snr_max:
                return max(0, min(9, int(mcs)))  # Ensure MCS is in valid range 0-9
        return 0  # Default to lowest MCS
    except Exception as e:
        log_message(f"Error in MCS calculation: {e}", LOG_DEBUG_PATH)
        return 0  # Safe default

def log_message(message, log_file=None, print_stdout=True):
    """Enhanced logging function"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_msg = f"[{timestamp}] {message}"
    
    if print_stdout:
        print(formatted_msg)
    
    if log_file:
        with open(log_file, 'a') as f:
            f.write(formatted_msg + "\n")

# ================== Performance Tracking ==================
class PerformanceMetrics:
    def __init__(self):
        self.reset_metrics()
        self.episode_data = []
        self.detailed_logs = []
        
    def reset_metrics(self):
        self.rewards = []
        self.cbr_values = []
        self.action_history = []
        self.power_history = []
        self.beacon_history = []
        self.epsilon_history = []
        self.q_values = []
        self.exploration_steps = 0
        self.exploitation_steps = 0
        
    def update_metrics(self, state, action, reward, next_state, epsilon, q_value, is_exploration):
        self.rewards.append(reward)
        self.cbr_values.append(state[2])  # CBR from current state
        self.action_history.append(action)
        self.power_history.append(state[0])
        self.beacon_history.append(state[1])
        self.epsilon_history.append(epsilon)
        self.q_values.append(q_value)
        
        if is_exploration:
            self.exploration_steps += 1
        else:
            self.exploitation_steps += 1
            
        # Detailed log for each step
        self.detailed_logs.append({
            'timestamp': datetime.now(),
            'power': state[0],
            'beacon_rate': state[1],
            'cbr': state[2],
            'neighbors': state[3],
            'snr': state[4],
            'action': action,
            'reward': reward,
            'q_value': q_value,
            'epsilon': epsilon,
            'is_exploration': is_exploration
        })
    
    def calculate_episode_metrics(self, episode_num):
        """Calculate comprehensive episode metrics"""
        if not self.rewards:
            return {}
            
        metrics = {
            'episode': episode_num,
            'timestamp': datetime.now(),
            'total_steps': len(self.rewards),
            'cumulative_reward': sum(self.rewards),
            'average_reward': np.mean(self.rewards),
            'max_reward': max(self.rewards),
            'min_reward': min(self.rewards),
            'reward_std': np.std(self.rewards),
            
            # CBR Performance
            'avg_cbr': np.mean(self.cbr_values),
            'cbr_std': np.std(self.cbr_values),
            'cbr_in_range_rate': sum(1 for cbr in self.cbr_values if CBR_RANGE[0] <= cbr <= CBR_RANGE[1]) / len(self.cbr_values),
            'cbr_violation_rate': sum(1 for cbr in self.cbr_values if not CBR_RANGE[0] <= cbr <= CBR_RANGE[1]) / len(self.cbr_values),
            'cbr_target_deviation': np.mean([abs(cbr - CBR_TARGET) for cbr in self.cbr_values]),
            
            # Action Analysis
            'action_entropy': entropy(np.bincount(self.action_history, minlength=ACTION_DIM)),
            'action_jitter': np.mean(np.abs(np.diff(self.action_history))) if len(self.action_history) > 1 else 0,
            'most_used_action': np.argmax(np.bincount(self.action_history, minlength=ACTION_DIM)),
            'action_diversity': len(np.unique(self.action_history)) / ACTION_DIM,
            
            # Power and Beacon Analysis
            'avg_power': np.mean(self.power_history),
            'power_std': np.std(self.power_history),
            'power_range': max(self.power_history) - min(self.power_history),
            'avg_beacon_rate': np.mean(self.beacon_history),
            'beacon_rate_std': np.std(self.beacon_history),
            'beacon_range': max(self.beacon_history) - min(self.beacon_history),
            
            # Q-Learning Specific Metrics
            'avg_q_value': np.mean(self.q_values),
            'q_value_std': np.std(self.q_values),
            'max_q_value': max(self.q_values),
            'min_q_value': min(self.q_values),
            'exploration_rate': self.exploration_steps / (self.exploration_steps + self.exploitation_steps) if (self.exploration_steps + self.exploitation_steps) > 0 else 0,
            'exploration_steps': self.exploration_steps,
            'exploitation_steps': self.exploitation_steps,
            'final_epsilon': self.epsilon_history[-1] if self.epsilon_history else 0,
        }
        
        return metrics
    
    def log_performance(self, episode_num):
        """Log performance and return metrics"""
        metrics = self.calculate_episode_metrics(episode_num)
        if metrics:
            self.episode_data.append(metrics)
            
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
                
                # Q-Learning Performance Analysis
                if self.episode_data:
                    analysis_data = self._generate_performance_analysis()
                    analysis_df = pd.DataFrame([analysis_data])
                    analysis_df.to_excel(writer, sheet_name='RL_Performance_Analysis', index=False)
                
                # Q-Table Statistics
                qtable_stats = self._analyze_qtable()
                qtable_df = pd.DataFrame([qtable_stats])
                qtable_df.to_excel(writer, sheet_name='Q_Table_Analysis', index=False)
                
            log_message(f"Performance data saved to {PERFORMANCE_LOG_PATH}", print_stdout=True)
            
        except Exception as e:
            log_message(f"Error saving to Excel: {e}", LOG_DEBUG_PATH, print_stdout=True)
    
    def _generate_performance_analysis(self):
        """Generate comprehensive RL performance analysis"""
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
            'exploration_exploitation_balance': df['exploration_rate'].mean(),
            'action_diversity_trend': df['action_diversity'].mean(),
            'cbr_stability_score': 1 - df['cbr_std'].mean(),  # Lower std = higher stability
            'reward_stability_score': 1 / (1 + df['reward_std'].mean()),  # Lower volatility = higher stability
            'power_exploration_range': df['power_range'].mean(),
            'beacon_exploration_range': df['beacon_range'].mean(),
        }
    
    def _detect_convergence(self, df, window=50):
        """Detect convergence episode based on reward stability"""
        if len(df) < window:
            return None
            
        rewards = df['cumulative_reward'].values
        for i in range(window, len(rewards)):
            recent_std = np.std(rewards[i-window:i])
            if recent_std < np.std(rewards[:i]) * 0.1:  # 10% of historical std
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
        reward_score = min(1.0, max(0.0, (recent_episodes['cumulative_reward'].mean() + 100) / 200))  # Normalize to 0-1
        
        return (cbr_score + reward_score) / 2
    
    def _calculate_improvement_rate(self, df, column):
        """Calculate improvement rate for a given metric"""
        if len(df) < 2:
            return 0
            
        values = df[column].values
        return np.polyfit(range(len(values)), values, 1)[0]  # Linear trend slope
    
    def _analyze_qtable(self):
        """Analyze Q-table statistics"""
        global q_table
        
        non_zero_count = np.count_nonzero(q_table)
        total_entries = q_table.size
        
        return {
            'total_q_entries': total_entries,
            'non_zero_entries': non_zero_count,
            'q_table_coverage': non_zero_count / total_entries,
            'max_q_value': np.max(q_table),
            'min_q_value': np.min(q_table),
            'mean_q_value': np.mean(q_table),
            'q_value_std': np.std(q_table),
            'q_value_range': np.max(q_table) - np.min(q_table),
        }

# ================== Q-Learning Agent ==================
class QLearningAgent:
    def __init__(self, training_mode=True, load_model=True):
        self.training_mode = training_mode
        self.epsilon = EPSILON if training_mode else 0.0  # No exploration in testing mode
        self.performance = PerformanceMetrics()
        self.episode_count = 0
        
        # Exploration tracking
        self.state_visit_counts = defaultdict(int)
        self.power_exploration_counts = [0] * 30   # Track power level visits (1-30)
        self.cbr_performance_history = []  # Track CBR performance over time
        
        # Handle model loading with dimension checking
        if load_model and os.path.exists(MODEL_SAVE_PATH):
            global q_table
            try:
                loaded_q_table = np.load(MODEL_SAVE_PATH)
                expected_shape = STATE_DIM + (ACTION_DIM,)
                
                if loaded_q_table.shape == expected_shape:
                    q_table = loaded_q_table
                    log_message(f"Loaded pre-trained model from {MODEL_SAVE_PATH} with shape {loaded_q_table.shape}", print_stdout=True)
                else:
                    log_message(f"Model shape mismatch! Expected {expected_shape}, got {loaded_q_table.shape}. Creating new Q-table.", print_stdout=True)
                    q_table = np.zeros(expected_shape, dtype=np.float32)
                    
            except Exception as e:
                log_message(f"Error loading model: {e}, initializing new Q-table", print_stdout=True)
                q_table = np.zeros(STATE_DIM + (ACTION_DIM,), dtype=np.float32)
        elif not training_mode:
            log_message("WARNING: Testing mode but no pre-trained model found!", print_stdout=True)
    
    def calculate_reward(self, current_state, next_state):
        """FIXED: Configurable CBR-centered reward function"""
        try:
            current_cbr = float(np.real(current_state[2]))
            next_cbr = float(np.real(next_state[2]))
            next_power = float(np.real(next_state[0]))
            next_beacon = float(np.real(next_state[1]))
            
            # Validate inputs
            current_cbr = np.clip(current_cbr, 0.0, 1.0)
            next_cbr = np.clip(next_cbr, 0.0, 1.0)
            next_power = np.clip(next_power, POWER_MIN, POWER_MAX)
            next_beacon = np.clip(next_beacon, BEACON_MIN, BEACON_MAX)
            
            # 1. Distance from target reward (primary component)
            current_distance = abs(current_cbr - CBR_TARGET)
            next_distance = abs(next_cbr - CBR_TARGET)
            distance_improvement = current_distance - next_distance
            distance_reward = distance_improvement * REWARD_DISTANCE_SCALE
            
            # 2. Target zone reward (being in the sweet spot)
            if CBR_RANGE[0] <= next_cbr <= CBR_RANGE[1]:
                target_zone_reward = REWARD_TARGET_ZONE  # Perfect target range
            elif CBR_WIDE_RANGE[0] <= next_cbr <= CBR_WIDE_RANGE[1]:
                target_zone_reward = REWARD_WIDE_ZONE  # Wider acceptable range
            else:
                target_zone_reward = 0
            
            # 3. Extreme CBR penalties (catastrophic failure cases)
            extreme_penalty = 0
            if next_cbr > CBR_EXTREME_HIGH:  # Catastrophic high CBR
                extreme_penalty = -REWARD_EXTREME_HIGH_PENALTY * (next_cbr - CBR_EXTREME_HIGH)
            elif next_cbr > CBR_HIGH_WARNING:  # High CBR
                extreme_penalty = -REWARD_HIGH_PENALTY * (next_cbr - CBR_HIGH_WARNING)
            elif next_cbr < CBR_EXTREME_LOW:  # Too low CBR (network underutilized)
                extreme_penalty = -REWARD_LOW_PENALTY * (CBR_EXTREME_LOW - next_cbr)
            
            # 4. Stability reward (small penalty for large parameter changes)
            power_change = abs(next_power - current_state[0])
            beacon_change = abs(next_beacon - current_state[1])
            stability_penalty = -(power_change + beacon_change) * REWARD_STABILITY_SCALE
            
            # 5. Efficiency reward (prefer lower power when possible)
            if CBR_RANGE[0] <= next_cbr <= CBR_RANGE[1]:
                # Only apply efficiency reward when CBR is good
                efficiency_reward = (30 - next_power) * REWARD_EFFICIENCY_SCALE
            else:
                efficiency_reward = 0
            
            # Combine all components
            total_reward = (distance_reward + target_zone_reward + 
                          extreme_penalty + stability_penalty + efficiency_reward)
            
            # Ensure reward is finite
            if not np.isfinite(total_reward):
                total_reward = -50.0
                
            # Debug logging for critical cases
            if next_cbr > CBR_HIGH_WARNING or abs(distance_improvement) > 0.1:
                log_message(f"REWARD DEBUG: CBR {current_cbr:.3f}->{next_cbr:.3f}, Distance: {current_distance:.3f}->{next_distance:.3f}, Improvement: {distance_improvement:.3f}, Total Reward: {total_reward:.1f}", LOG_DEBUG_PATH)
            
            return np.clip(total_reward, -500, 500)  # Prevent extreme rewards
            
        except Exception as e:
            log_message(f"Error in reward calculation: {e}", LOG_DEBUG_PATH)
            return -50.0  # Default penalty
    
    def select_action(self, state):
        """Enhanced epsilon-greedy action selection with CBR-aware exploration"""
        try:
            # Validate and clean state inputs
            if len(state) < 5:
                log_message(f"Invalid state length {len(state)}, using defaults", LOG_DEBUG_PATH)
                state = [15, 10, CBR_TARGET, 20, 20]  # default state with configured CBR target
            
            # Ensure all state values are valid
            power = np.clip(float(np.real(state[0])), POWER_MIN, POWER_MAX)
            beacon = float(np.real(state[1]))
            cbr = np.clip(float(np.real(state[2])), 0.0, 1.0)
            neighbors = max(0, int(np.real(state[3])))
            snr = np.clip(float(np.real(state[4])), 0, 50)
            
            # Handle NaN or infinite values
            if not all(np.isfinite([power, beacon, cbr, neighbors, snr])):
                log_message("Invalid state values detected, using defaults", LOG_DEBUG_PATH)
                power, beacon, cbr, neighbors, snr = 15, 10, CBR_TARGET, 20, 20
            
            # Track power usage for analysis
            if self.training_mode and 1 <= int(power) <= 30:
                self.power_exploration_counts[int(power) - 1] += 1
            
            # Discretize state variables
            cbr_idx = discretize(cbr, CBR_BINS, CBR_STATES)
            neighbors_idx = discretize(neighbors, NEIGHBORS_BINS, NEIGHBORS_STATES) 
            snr_idx = discretize(snr, SNR_BINS, SNR_STATES)
            power_idx = discretize(power, POWER_BINS, POWER_STATES)
            beacon_idx = get_beacon_index(beacon)
            
            # Track state visits
            state_key = (cbr_idx, neighbors_idx, snr_idx, power_idx, beacon_idx)
            self.state_visit_counts[state_key] += 1
            
            # Debug: Log discretization occasionally
            if random.random() < 0.01:  # 1% of the time
                log_message(f"State: CBR {cbr:.3f}->{cbr_idx}, Neighbors {neighbors}->{neighbors_idx}, SNR {snr:.1f}->{snr_idx}, Power {power:.1f}->{power_idx}, Beacon {beacon:.1f}->{beacon_idx}", LOG_DEBUG_PATH)
            
            # Get Q-values with bounds checking
            try:
                q_values = q_table[cbr_idx, neighbors_idx, snr_idx, power_idx, beacon_idx].copy()
            except IndexError as e:
                log_message(f"Index error in Q-table access: {e}", LOG_DEBUG_PATH)
                # Use safe default indices
                cbr_idx = min(cbr_idx, CBR_STATES - 1)
                neighbors_idx = min(neighbors_idx, NEIGHBORS_STATES - 1)
                snr_idx = min(snr_idx, SNR_STATES - 1)
                power_idx = min(power_idx, POWER_STATES - 1)
                beacon_idx = min(beacon_idx, BEACON_STATES - 1)
                q_values = q_table[cbr_idx, neighbors_idx, snr_idx, power_idx, beacon_idx].copy()
            
            # CBR-aware exploration strategy
            is_exploration = False
            if self.training_mode:
                # Increase exploration when CBR is far from target
                cbr_distance = abs(cbr - CBR_TARGET)
                adaptive_epsilon = min(1.0, self.epsilon + cbr_distance * 2)  # Boost exploration when CBR is bad
                
                if random.random() < adaptive_epsilon:
                    # CBR-aware action selection during exploration
                    if cbr > CBR_HIGH_WARNING:  # CBR too high, bias toward power reduction
                        power_reduction_actions = [3, 4, 5, 9, 10, 11, 13, 15]  # Actions that reduce power
                        action = random.choice(power_reduction_actions)
                        log_message(f"HIGH CBR EXPLORATION: CBR={cbr:.3f}, chose power reduction action {action}", LOG_DEBUG_PATH)
                    elif cbr < CBR_EXTREME_LOW:  # CBR too low, bias toward power increase
                        power_increase_actions = [0, 1, 2, 6, 7, 8, 12, 14]  # Actions that increase power
                        action = random.choice(power_increase_actions)
                        log_message(f"LOW CBR EXPLORATION: CBR={cbr:.3f}, chose power increase action {action}", LOG_DEBUG_PATH)
                    else:
                        action = random.randint(0, ACTION_DIM - 1)  # Normal random exploration
                    is_exploration = True
                else:
                    action = np.argmax(q_values)
            else:
                action = np.argmax(q_values)
                
            q_value = q_values[action] if len(q_values) > action else 0.0
            return action, q_value, is_exploration
            
        except Exception as e:
            log_message(f"Error in action selection: {e}", LOG_DEBUG_PATH)
            return 0, 0.0, True
    
    def apply_action(self, current_power, current_beacon, action_idx):
        """Apply action with proper bounds checking"""
        try:
            # Validate inputs
            current_power = np.clip(float(current_power), POWER_MIN, POWER_MAX)
            current_beacon = float(current_beacon)
            action_idx = max(0, min(ACTION_DIM - 1, int(action_idx)))
            
            # Get current beacon index
            current_beacon_idx = get_beacon_index(current_beacon)
            
            # Get action changes
            power_change_dbm, beacon_level_change = ACTION_SPACE[action_idx]
            
            # Apply power change
            new_power = current_power + power_change_dbm
            new_power = np.clip(new_power, POWER_MIN, POWER_MAX)
            
            # Apply beacon change
            new_beacon_idx = current_beacon_idx + beacon_level_change
            new_beacon_idx = max(0, min(len(DISCRETE_BEACON_LEVELS) - 1, new_beacon_idx))
            new_beacon = get_beacon_from_index(new_beacon_idx)
            
            return int(new_power), int(new_beacon)
            
        except Exception as e:
            log_message(f"Error in apply_action: {e}", LOG_DEBUG_PATH)
            return 15, 10  # Safe middle values
    
    def update_q_table(self, state, action, reward, next_state):
        """Enhanced Q-learning update with proper validation"""
        try:
            # Validate inputs
            if len(state) < 5 or len(next_state) < 5:
                log_message("Invalid state dimensions for Q-table update", LOG_DEBUG_PATH)
                return
                
            # Ensure reward is valid
            reward = float(np.real(reward))
            if not np.isfinite(reward):
                reward = -50.0
                
            # Validate action
            action = max(0, min(ACTION_DIM - 1, int(action)))
            
            # Current state discretization
            power = np.clip(float(np.real(state[0])), POWER_MIN, POWER_MAX)
            beacon = float(np.real(state[1]))
            cbr = np.clip(float(np.real(state[2])), 0.0, 1.0)
            neighbors = max(0, int(np.real(state[3])))
            snr = np.clip(float(np.real(state[4])), 0, 50)
            
            cbr_idx = discretize(cbr, CBR_BINS, CBR_STATES)
            neighbors_idx = discretize(neighbors, NEIGHBORS_BINS, NEIGHBORS_STATES)
            snr_idx = discretize(snr, SNR_BINS, SNR_STATES)
            power_idx = discretize(power, POWER_BINS, POWER_STATES)
            beacon_idx = get_beacon_index(beacon)
            
            # Next state discretization
            next_power = np.clip(float(np.real(next_state[0])), POWER_MIN, POWER_MAX)
            next_beacon = float(np.real(next_state[1]))
            next_cbr = np.clip(float(np.real(next_state[2])), 0.0, 1.0)
            next_neighbors = max(0, int(np.real(next_state[3])))
            next_snr = np.clip(float(np.real(next_state[4])), 0, 50)
            
            next_cbr_idx = discretize(next_cbr, CBR_BINS, CBR_STATES)
            next_neighbors_idx = discretize(next_neighbors, NEIGHBORS_BINS, NEIGHBORS_STATES)
            next_snr_idx = discretize(next_snr, SNR_BINS, SNR_STATES)
            next_power_idx = discretize(next_power, POWER_BINS, POWER_STATES)
            next_beacon_idx = get_beacon_index(next_beacon)
            
            # Q-learning update with bounds checking
            try:
                current_q = q_table[cbr_idx, neighbors_idx, snr_idx, power_idx, beacon_idx][action]
                max_next_q = np.max(q_table[next_cbr_idx, next_neighbors_idx, next_snr_idx, next_power_idx, next_beacon_idx])
            except IndexError as e:
                log_message(f"Index error in Q-table update: {e}", LOG_DEBUG_PATH)
                return
            
            # Q-learning update rule
            new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q - current_q)
            
            # Ensure Q-value is finite
            if np.isfinite(new_q):
                q_table[cbr_idx, neighbors_idx, snr_idx, power_idx, beacon_idx][action] = new_q
                
                # Log significant Q-value updates
                if abs(reward) > 50 or abs(new_q - current_q) > 10:
                    log_message(f"SIGNIFICANT Q-UPDATE: S={cbr:.3f},{neighbors},{snr:.1f},{power:.1f},{beacon}, A={action}, R={reward:.1f}, Q: {current_q:.2f}->{new_q:.2f}", LOG_DEBUG_PATH)
            else:
                log_message(f"Invalid Q-value {new_q}, skipping update", LOG_DEBUG_PATH)
            
        except Exception as e:
            log_message(f"Error in Q-table update: {e}", LOG_DEBUG_PATH)
    
    def decay_epsilon(self):
        if self.training_mode:
            self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)
    
    def save_model(self):
        try:
            np.save(MODEL_SAVE_PATH, q_table)
            log_message(f"Model saved to {MODEL_SAVE_PATH} with shape {q_table.shape}", print_stdout=True)
        except Exception as e:
            log_message(f"Error saving model: {e}", LOG_DEBUG_PATH, print_stdout=True)
    
    def end_episode(self):
        if self.training_mode:
            self.episode_count += 1
            metrics = self.performance.log_performance(self.episode_count)
            
            # Log CBR performance and exploration statistics
            if self.episode_count % 10 == 0:
                avg_cbr = np.mean(self.performance.cbr_values) if self.performance.cbr_values else 0
                self.cbr_performance_history.append(avg_cbr)
                
                # Power exploration analysis
                total_power_visits = sum(self.power_exploration_counts)
                if total_power_visits > 0:
                    low_power_visits = sum(self.power_exploration_counts[:15])    # 1-15 dBm
                    high_power_visits = sum(self.power_exploration_counts[15:])   # 16-30 dBm
                    low_power_pct = (low_power_visits / total_power_visits) * 100
                    high_power_pct = (high_power_visits / total_power_visits) * 100
                    
                    log_message(f"EPISODE {self.episode_count}: Avg CBR={avg_cbr:.3f} (target={CBR_TARGET}), Power exploration - Low(1-15): {low_power_pct:.1f}%, High(16-30): {high_power_pct:.1f}%, Epsilon: {self.epsilon:.3f}", print_stdout=True)
                
                # CBR performance trend
                if len(self.cbr_performance_history) >= 5:
                    recent_trend = np.mean(self.cbr_performance_history[-5:])
                    early_trend = np.mean(self.cbr_performance_history[:5]) if len(self.cbr_performance_history) >= 10 else recent_trend
                    trend_direction = "IMPROVING" if recent_trend < early_trend and recent_trend < CBR_HIGH_WARNING else "DEGRADING" if recent_trend > CBR_HIGH_WARNING else "STABLE"
                    log_message(f"CBR TREND: {trend_direction} (recent avg: {recent_trend:.3f})", print_stdout=True)
            
            if self.episode_count % MODEL_SAVE_INTERVAL == 0:
                self.save_model()
            
            self.performance.reset_metrics()
            return metrics
        return {}

# ================== Enhanced RL Server ==================
class RLServer:
    def __init__(self, host, port, training_mode=True):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(5)
        self.agent = QLearningAgent(training_mode=training_mode, load_model=True)
        self.training_mode = training_mode
        self.running = True
        
        mode_str = "TRAINING" if training_mode else "TESTING"
        log_message(f"CBR-Centered RL Server started in {mode_str} mode on {host}:{port}", print_stdout=True)

    def receive_message_with_header(self, conn):
        """Receive message with 4-byte length header"""
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
        """Send message with 4-byte length header"""
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

    def handle_client(self, conn, addr):
        """Enhanced client handler with CBR monitoring"""
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
                    cbr_values = []  # Track CBR values for this batch
                    
                    for veh_id, veh_info in batch_data.items():
                        response = self.process_vehicle(veh_id, veh_info)
                        if response:
                            responses[veh_id] = response
                            # Collect CBR values for monitoring
                            if "CBR" in veh_info:
                                cbr_values.append(float(veh_info["CBR"]))
                    
                    # Log batch CBR statistics
                    if cbr_values:
                        avg_cbr = np.mean(cbr_values)
                        max_cbr = max(cbr_values)
                        min_cbr = min(cbr_values)
                        if avg_cbr > CBR_HIGH_WARNING or max_cbr > CBR_EXTREME_HIGH:  # Alert for high CBR
                            log_message(f"⚠️  HIGH CBR DETECTED: Avg={avg_cbr:.3f}, Max={max_cbr:.3f}, Min={min_cbr:.3f} (Target={CBR_TARGET})", print_stdout=True)
                    
                    # Send response
                    response_dict = {"vehicles": responses}
                    response_str = json.dumps(response_dict)
                    
                    if self.send_message_with_header(conn, response_str):
                        log_message(f"Sent response to {addr}: {len(responses)} vehicles", LOG_SENT_PATH)
                    else:
                        break
                        
                    # Training episode management
                    if self.training_mode and len(self.agent.performance.rewards) >= 100:
                        self.agent.decay_epsilon()
                        metrics = self.agent.end_episode()
                        if metrics:
                            log_message(f"Episode {self.agent.episode_count} completed. Avg CBR: {metrics.get('avg_cbr', 0):.3f}, Cumulative Reward: {metrics.get('cumulative_reward', 0):.1f}, CBR in range: {metrics.get('cbr_in_range_rate', 0):.3f}", print_stdout=True)
                
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

    def process_vehicle(self, veh_id, veh_info):
        """Process individual vehicle with CBR-focused optimization"""
        try:
            # Extract vehicle parameters with validation
            current_power = float(veh_info.get("transmissionPower", 15))
            current_beacon = float(veh_info.get("beaconRate", 10))
            current_snr = float(veh_info.get("SINR", veh_info.get("SNR", 20)))
            neighbors = int(veh_info.get("neighbors", 20))
            cbr = float(veh_info.get("CBR", CBR_TARGET))
            
            # Handle sectoral antenna data if present
            if "front_power" in veh_info and "rear_power" in veh_info:
                front_power = float(veh_info.get("front_power", current_power))
                rear_power = float(veh_info.get("rear_power", current_power))
                current_power = (front_power + rear_power) / 2
            
            # Validate and clamp all inputs
            current_power = np.clip(current_power, POWER_MIN, POWER_MAX)
            current_beacon = np.clip(current_beacon, BEACON_MIN, BEACON_MAX)
            current_snr = np.clip(current_snr, 0, 50)
            neighbors = max(0, neighbors)
            cbr = np.clip(cbr, 0.0, 1.0)
            
            # Handle NaN or infinite values
            if math.isnan(cbr) or not np.isfinite(cbr):
                cbr = CBR_TARGET  # Default to configured target
            if math.isnan(current_snr) or not np.isfinite(current_snr):
                current_snr = 20.0
            if math.isnan(current_power) or not np.isfinite(current_power):
                current_power = 15.0  # Conservative default
            if math.isnan(current_beacon) or not np.isfinite(current_beacon):
                current_beacon = 10.0
            
            # Ensure beacon is from discrete levels
            current_beacon = get_beacon_from_index(get_beacon_index(current_beacon))
            
            # Create state vector
            current_state = [current_power, current_beacon, cbr, neighbors, current_snr]
            
            # Select action
            action_idx, q_value, is_exploration = self.agent.select_action(current_state)
            new_power, new_beacon = self.agent.apply_action(current_power, current_beacon, action_idx)
            
            # Additional validation of outputs
            new_power = np.clip(new_power, POWER_MIN, POWER_MAX)
            new_beacon = np.clip(new_beacon, BEACON_MIN, BEACON_MAX)
            
            # Create next state for training
            next_state = [new_power, new_beacon, cbr, neighbors, current_snr]
            
            # Training updates
            if self.training_mode:
                reward = self.agent.calculate_reward(current_state, next_state)
                
                # Validate reward
                if not np.isfinite(reward):
                    reward = -50.0
                
                self.agent.update_q_table(current_state, action_idx, reward, next_state)
                self.agent.performance.update_metrics(
                    current_state, action_idx, float(reward), next_state, 
                    self.agent.epsilon, q_value, is_exploration
                )
            
            # Prepare response
            response = {
                "transmissionPower": int(new_power),
                "beaconRate": int(new_beacon),
                "MCS": adjust_mcs_based_on_snr(current_snr)
            }
            
            # Add sectoral antenna support
            if "antenna_type" in veh_info and veh_info["antenna_type"] == "SECTORAL":
                response["front_power"] = int(new_power)
                response["rear_power"] = int(new_power)
                response["side_power_static"] = veh_info.get("side_power_static", 10)
            
            # Log interesting cases
            if cbr > CBR_EXTREME_HIGH or abs(new_power - current_power) > 5:
                log_message(f"Vehicle {veh_id}: Power {current_power:.1f}->{new_power}, Beacon {current_beacon:.1f}->{new_beacon}, CBR {cbr:.3f}, Action {action_idx}, Exploration: {is_exploration}", LOG_DEBUG_PATH)
            
            return response
            
        except Exception as e:
            log_message(f"Error processing vehicle {veh_id}: {e}", LOG_DEBUG_PATH, print_stdout=True)
            # Return conservative default values
            return {
                "transmissionPower": 15,  # Conservative power
                "beaconRate": 10,         # Middle beacon rate
                "MCS": 4
            }

    def start(self):
        """Start the server"""
        try:
            log_message("CBR-Centered RL Server listening for connections...", print_stdout=True)
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
    
    def stop(self):
        """Stop the server and save final results"""
        log_message("Stopping CBR-Centered RL server...", print_stdout=True)
        self.running = False
        
        try:
            self.server.close()
        except:
            pass
        
        if self.training_mode:
            self.agent.save_model()
            # Force save final performance data
            self.agent.performance.save_to_excel()
            log_message("Final performance data saved", print_stdout=True)
        
        log_message("CBR-Centered RL server stopped", print_stdout=True)

# ================== Main Execution ==================
def main():
    # Validate operation mode
    if OPERATION_MODE.upper() not in ["TRAINING", "TESTING"]:
        print(f"ERROR: Invalid OPERATION_MODE '{OPERATION_MODE}'. Must be 'TRAINING' or 'TESTING'")
        sys.exit(1)
    
    training_mode = (OPERATION_MODE.upper() == "TRAINING")
    
    print("="*80)
    print(f" CBR-CENTERED Q-LEARNING VANET SERVER")
    print(f"Host: {HOST}:{PORT}")
    print(f"Mode: {OPERATION_MODE.upper()}")
    print(f"  CBR CONFIGURATION:")
    print(f"   Target CBR: {CBR_TARGET}")
    print(f"   Acceptable Range: {CBR_RANGE[0]:.3f} - {CBR_RANGE[1]:.3f} (±{CBR_TOLERANCE} tolerance)")
    print(f"   Wide Range: {CBR_WIDE_RANGE[0]:.3f} - {CBR_WIDE_RANGE[1]:.3f} (±{CBR_WIDE_ACCEPTABLE_TOLERANCE} tolerance)")
    print(f"   Extreme High Threshold: {CBR_EXTREME_HIGH}")
    print(f"   High Warning Threshold: {CBR_HIGH_WARNING}")
    print(f"   Extreme Low Threshold: {CBR_EXTREME_LOW}")
    print(f"Power Range: {POWER_MIN}-{POWER_MAX} dBm (continuous, {POWER_STATES} bins)")
    print(f"Beacon Range: {BEACON_MIN}-{BEACON_MAX} Hz (discrete)")
    print(f"State Dimensions: {STATE_DIM}")
    print(f"Q-table Shape: {q_table.shape}")
    print(f"Action Space Size: {ACTION_DIM}")
    print(f" Files: Model={MODEL_SAVE_PATH}, Results={PERFORMANCE_LOG_PATH}")
    print("  REWARD FUNCTION FIXES:")
    print("  • Massive penalties for extreme CBR (configurable thresholds)")
    print(f"  • Rewards for moving closer to {CBR_TARGET} from ANY direction")  
    print(f"  • Target zone bonuses for CBR in {CBR_RANGE[0]:.2f}-{CBR_RANGE[1]:.2f} range")
    print("  • CBR-aware exploration (more exploration when CBR is bad)")
    print("  • Conservative action space to prevent wild swings")
    if training_mode:
        print(f"Learning Rate: {LEARNING_RATE}")
        print(f"Initial Epsilon: {EPSILON}")
        print(f"Epsilon Decay: {EPSILON_DECAY}")
        print(f"Min Epsilon: {MIN_EPSILON}")
        print(f"Model will be saved every {MODEL_SAVE_INTERVAL} episodes")
        print("  USING CONFIGURABLE MODEL NAME!")
    else:
        print(f"Using pre-trained model: {MODEL_SAVE_PATH}")
    print("="*80)
    
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
