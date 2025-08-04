"""
DUAL-AGENT Q-Learning VANET Server
==================================

This script implements a SEPARATED MAC/PHY Q-learning architecture for VANET optimization.
Based on the dual-agent SAC approach with:
- MAC Agent: Controls beacon rate and MCS (CBR-focused)
- PHY Agent: Controls transmission power (SINR-focused)
- Neighbor-aware optimization
- Centralized learning manager

QUICK START:
1. For TRAINING: Set OPERATION_MODE = "TRAINING" below
2. For TESTING: Set OPERATION_MODE = "TESTING" below  
3. Run: python dual_qlearning_server.py
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

# ================== CONFIGURATION (REPLACE THE EXISTING CONFIGURATION SECTION) ==================
# CHANGE THIS TO SWITCH BETWEEN MODES
OPERATION_MODE = "TESTING"        # Options: "TRAINING" or "TESTING"

# NEW: Antenna Type Configuration (doesn't require VANET script changes)
ANTENNA_TYPE = "SECTORAL"          # Options: "SECTORAL" or "OMNIDIRECTIONAL"
# This setting tells the dual-agent system how to optimize internally
# The VANET script interface remains exactly the same

# ================== Constants ==================
CBR_TARGET = 0.65                  # Target CBR for optimal performance
CBR_RANGE = (0.6, 0.7)             # Acceptable CBR range
BUFFER_SIZE = 100000               # Experience buffer size
LEARNING_RATE = 0.15               # Q-learning rate
DISCOUNT_FACTOR = 0.95             # Future reward discount
EPSILON = 1.0                      # Initial exploration rate
EPSILON_DECAY = 0.9995             # Exploration decay
MIN_EPSILON = 0.1                  # Minimum exploration
HOST = '127.0.0.1'                 # Server IP
PORT = 5000                        # Server port

# Power and beacon ranges
POWER_MIN = 1                      # Minimum power (dBm) 
POWER_MAX = 30                     # Maximum power (dBm)
BEACON_MIN = 1                     # Minimum beacon rate (Hz)
BEACON_MAX = 20                    # Maximum beacon rate (Hz)

# File paths - Include antenna type in filename for separate models
MODEL_PREFIX = f"{ANTENNA_TYPE.lower()}_dual_agent"
MAC_MODEL_PATH = f'{MODEL_PREFIX}_mac_qlearning_model.npy'
PHY_MODEL_PATH = f'{MODEL_PREFIX}_phy_qlearning_model.npy'
PERFORMANCE_LOG_PATH = f'{MODEL_PREFIX}_performance.xlsx'
MODEL_SAVE_INTERVAL = 50           # Save model every N episodes
PERFORMANCE_LOG_INTERVAL = 10      # Log performance every N episodes


# ================== State Discretization ==================
# MAC Agent State: [CBR, SINR, beacon, mcs, neighbors] - 5D
CBR_BINS = np.linspace(0.0, 1.0, 21)          # 20 bins for CBR
SINR_BINS = np.linspace(0, 50, 11)            # 10 bins for SINR
BEACON_BINS = np.arange(1, 21)                # Discrete beacon values 1-20
MCS_BINS = np.arange(0, 10)                   # Discrete MCS values 0-9
NEIGHBORS_BINS = np.linspace(0, 50, 11)       # 10 bins for neighbors

# PHY Agent State: [CBR, SINR, power, neighbors] - 4D
POWER_BINS = np.arange(1, 31)                 # Discrete power values 1-30

# State dimensions
MAC_STATE_DIM = (20, 10, 20, 10, 10)  # CBR, SINR, beacon(1-20), MCS(0-9), neighbors
PHY_STATE_DIM = (20, 10, 30, 10)      # CBR, SINR, power(1-30), neighbors

# ================== Action Spaces (SAME AS BEFORE) ==================
# MAC Agent Actions: [beacon_delta, mcs_delta]
MAC_ACTIONS = [
    (0, 0),    # No change
    (1, 0),    # Beacon +1
    (-1, 0),   # Beacon -1
    (2, 0),    # Beacon +2
    (-2, 0),   # Beacon -2
    (3, 0),    # Beacon +3
    (-3, 0),   # Beacon -3
    (5, 0),    # Beacon +5
    (-5, 0),   # Beacon -5
    (0, 1),    # MCS +1
    (0, -1),   # MCS -1
    (0, 2),    # MCS +2
    (0, -2),   # MCS -2
    (1, 1),    # Beacon +1, MCS +1
    (1, -1),   # Beacon +1, MCS -1
    (-1, 1),   # Beacon -1, MCS +1
    (-1, -1),  # Beacon -1, MCS -1
    (2, 1),    # Beacon +2, MCS +1
    (-2, -1),  # Beacon -2, MCS -1
    (10, 0),   # Beacon +10 (large jump)
    (-10, 0),  # Beacon -10 (large jump)
    (0, 5),    # MCS +5 (large jump)
    (0, -5),   # MCS -5 (large jump)
]

# PHY Agent Actions: [power_delta]
PHY_ACTIONS = [
    0,    # No change
    1,    # Power +1
    -1,   # Power -1
    2,    # Power +2
    -2,   # Power -2
    3,    # Power +3
    -3,   # Power -3
    5,    # Power +5
    -5,   # Power -5
    10,   # Power +10 (large jump)
    -10,  # Power -10 (large jump)
    15,   # Power +15 (very large jump)
    -15,  # Power -15 (very large jump)
]

MAC_ACTION_DIM = len(MAC_ACTIONS)
PHY_ACTION_DIM = len(PHY_ACTIONS)

# Initialize Q-tables FIRST
mac_q_table = np.zeros(MAC_STATE_DIM + (MAC_ACTION_DIM,), dtype=np.float32)
phy_q_table = np.zeros(PHY_STATE_DIM + (PHY_ACTION_DIM,), dtype=np.float32)

# ================== Logging Setup (MOVE TO AFTER Q-table initialization) ==================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_DEBUG_PATH = os.path.join(LOG_DIR, f'{MODEL_PREFIX}_debug.log')

# Configure logging with UTF-8 encoding to prevent Unicode errors
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_DEBUG_PATH, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# NOW the logger can access the Q-tables safely
logger.info("DUAL-AGENT Q-LEARNING INITIALIZED")
logger.info(f"Antenna Type: {ANTENNA_TYPE}")
logger.info(f"MAC Q-table shape: {mac_q_table.shape} (5D state -> 2D action)")
logger.info(f"PHY Q-table shape: {phy_q_table.shape} (4D state -> 1D action)")

# ================== Helper Functions ==================
def discretize(value, bins):
    """Discretize a continuous value into a bin index"""
    if not np.isfinite(value):
        value = 0.0
    
    # For discrete values (beacon, mcs, power), directly map to index
    if len(bins) <= 30 and bins[0] == 1 and bins[-1] in [20, 30]:  # Beacon or Power
        value = int(np.clip(value, bins[0], bins[-1]))
        return value - bins[0]  # Convert to 0-based index
    elif len(bins) == 10 and bins[0] == 0 and bins[-1] == 9:  # MCS
        value = int(np.clip(value, 0, 9))
        return value
    else:  # Continuous values (CBR, SINR, neighbors)
        value = np.clip(value, bins[0], bins[-1])
        bin_idx = np.digitize(value, bins) - 1
        return max(0, min(len(bins) - 2, bin_idx))

# ================== Enhanced Helper Functions (REPLACE EXISTING) ==================
# ================== Enhanced Helper Functions (REPLACE EXISTING) ==================
def get_neighbor_category(neighbor_count, antenna_type="OMNIDIRECTIONAL"):
    """Enhanced neighbor density categorization with internal antenna awareness"""
    # Internal logic - doesn't change input/output interface
    if antenna_type.upper() == "SECTORAL":
        # Sectoral antennas can handle higher densities better
        if neighbor_count <= 3:
            return "VERY_LOW"
        elif neighbor_count <= 8:
            return "LOW"
        elif neighbor_count <= 15:
            return "MEDIUM"
        elif neighbor_count <= 22:
            return "HIGH"
        elif neighbor_count <= 30:
            return "VERY_HIGH"
        else:
            return "EXTREME"
    else:
        # Original omnidirectional thresholds
        if neighbor_count <= 2:
            return "VERY_LOW"
        elif neighbor_count <= 5:
            return "LOW"
        elif neighbor_count <= 10:
            return "MEDIUM"
        elif neighbor_count <= 15:
            return "HIGH"
        elif neighbor_count <= 20:
            return "VERY_HIGH"
        else:
            return "EXTREME"

def discretize(value, bins):
    """Discretize a continuous value into a bin index - SAME AS ORIGINAL"""
    if not np.isfinite(value):
        value = 0.0
    
    # For discrete values (beacon, mcs, power), directly map to index
    if len(bins) <= 30 and bins[0] == 1 and bins[-1] in [20, 30]:  # Beacon or Power
        value = int(np.clip(value, bins[0], bins[-1]))
        return value - bins[0]  # Convert to 0-based index
    elif len(bins) == 10 and bins[0] == 0 and bins[-1] == 9:  # MCS
        value = int(np.clip(value, 0, 9))
        return value
    else:  # Continuous values (CBR, SINR, neighbors)
        value = np.clip(value, bins[0], bins[-1])
        bin_idx = np.digitize(value, bins) - 1
        return max(0, min(len(bins) - 2, bin_idx))

def get_density_multiplier(neighbor_count, antenna_type="OMNIDIRECTIONAL"):
    """Get density-based multiplier for reward calculation"""
    density_cat = get_neighbor_category(neighbor_count, antenna_type)
    multipliers = {
        "VERY_LOW": 0.5,
        "LOW": 0.8, 
        "MEDIUM": 1.0,
        "HIGH": 1.3,
        "VERY_HIGH": 1.6,
        "EXTREME": 2.0
    }
    return multipliers.get(density_cat, 1.0)

def analyze_antenna_efficiency(antenna_type, neighbors, cbr, sinr):
    """Analyze antenna efficiency based on conditions"""
    if antenna_type.upper() == "SECTORAL":
        # Sectoral antennas should perform better in high density
        if neighbors > 10:
            efficiency_bonus = 0.2  # 20% bonus in high density
            if cbr < 0.7 and sinr > 10:  # Good performance metrics
                efficiency_bonus = 0.3
        else:
            efficiency_bonus = 0.1  # Small bonus in low density
    else:
        # Omnidirectional baseline
        efficiency_bonus = 0.0
        if neighbors <= 5:  # Omnidirectional might be better in very low density
            efficiency_bonus = 0.1
    
    return efficiency_bonus

# ================== Separated Agent Classes ==================

class MACAgent:
    """MAC Agent - Controls beacon rate and MCS with CBR focus"""
    
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon
        self.q_table = mac_q_table
        self.state_visit_counts = defaultdict(int)
        
    def get_state_indices(self, cbr, sinr, beacon, mcs, neighbors):
        """Convert continuous state to discrete indices"""
        cbr_idx = discretize(cbr, CBR_BINS)
        sinr_idx = discretize(sinr, SINR_BINS)
        beacon_idx = discretize(beacon, BEACON_BINS)
        mcs_idx = discretize(mcs, MCS_BINS)
        neighbors_idx = discretize(neighbors, NEIGHBORS_BINS)
        
        return (cbr_idx, sinr_idx, beacon_idx, mcs_idx, neighbors_idx)
    
    def select_action(self, state_indices, neighbor_count, antenna_type="OMNIDIRECTIONAL"):
        """Select action with internal antenna awareness but same interface"""
        # Track state visits
        self.state_visit_counts[state_indices] += 1
        
        # Internal density classification (doesn't change input)
        density_category = get_neighbor_category(neighbor_count, antenna_type)
        adaptive_epsilon = self.epsilon
        
        # Internal antenna-aware exploration (doesn't change interface)
        if antenna_type.upper() == "SECTORAL" and density_category in ["HIGH", "VERY_HIGH", "EXTREME"]:
            adaptive_epsilon = min(1.0, self.epsilon * 1.3)  # More exploration for sectoral in high density
        
        # Enhanced exploration for full discrete range coverage
        if random.random() < adaptive_epsilon:
            # 20% chance for random jump to any discrete value (during high exploration)
            if self.epsilon > 0.5 and random.random() < 0.2:
                # Direct jump to random discrete values
                if random.random() < 0.5:  # 50% chance to explore beacon
                    target_beacon = random.randint(1, 20)
                    current_beacon = state_indices[2] + 1  # Convert back from index
                    beacon_delta = target_beacon - current_beacon
                    # Find closest action
                    best_action = 0
                    best_diff = float('inf')
                    for i, (b, m) in enumerate(MAC_ACTIONS):
                        if abs(b - beacon_delta) < best_diff:
                            best_diff = abs(b - beacon_delta)
                            best_action = i
                    return best_action
                else:  # 50% chance to explore MCS
                    target_mcs = random.randint(0, 9)
                    current_mcs = state_indices[3]
                    mcs_delta = target_mcs - current_mcs
                    # Find closest action
                    best_action = 0
                    best_diff = float('inf')
                    for i, (b, m) in enumerate(MAC_ACTIONS):
                        if abs(m - mcs_delta) < best_diff:
                            best_diff = abs(m - mcs_delta)
                            best_action = i
                    return best_action
            
            # Antenna-aware biased exploration (internal logic)
            if antenna_type.upper() == "SECTORAL":
                if density_category in ["HIGH", "VERY_HIGH", "EXTREME"]:
                    # Sectoral antennas can be more aggressive in high density
                    preferred_actions = [i for i, (b, m) in enumerate(MAC_ACTIONS) if b <= 1]
                    if preferred_actions and random.random() < 0.8:
                        return random.choice(preferred_actions)
                elif density_category in ["VERY_LOW", "LOW"]:
                    # Can use higher beacon rates with sectoral
                    preferred_actions = [i for i, (b, m) in enumerate(MAC_ACTIONS) if b >= 0]
                    if preferred_actions and random.random() < 0.7:
                        return random.choice(preferred_actions)
            else:
                # Omnidirectional strategy (more conservative)
                if density_category in ["HIGH", "VERY_HIGH", "EXTREME"]:
                    preferred_actions = [i for i, (b, m) in enumerate(MAC_ACTIONS) if b <= 0]
                    if preferred_actions and random.random() < 0.7:
                        return random.choice(preferred_actions)
                elif density_category in ["VERY_LOW", "LOW"]:
                    preferred_actions = [i for i, (b, m) in enumerate(MAC_ACTIONS) if b >= 0]
                    if preferred_actions and random.random() < 0.7:
                        return random.choice(preferred_actions)
            
            return random.randint(0, MAC_ACTION_DIM - 1)
        else:
            # Exploit
            return np.argmax(self.q_table[state_indices])
    
    def calculate_reward(self, cbr, sinr, beacon, mcs, neighbors, next_cbr, next_beacon, next_mcs, antenna_type="OMNIDIRECTIONAL"):
        """MAC-specific reward with internal antenna awareness"""
        # Primary: CBR optimization
        cbr_error = abs(cbr - CBR_TARGET)
        cbr_reward = 10.0 * (1 - math.tanh(20 * cbr_error))
        
        # Internal antenna-aware optimization (doesn't change interface)
        density_category = get_neighbor_category(neighbors, antenna_type)
        beacon_reward = 0.0
        
        # Enhanced beacon reward based on antenna type (internal logic)
        if antenna_type.upper() == "SECTORAL":
            # Sectoral antennas can handle higher densities better
            if density_category in ["HIGH", "VERY_HIGH", "EXTREME"]:
                if beacon <= 6:  # More conservative for sectoral in high density
                    beacon_reward = 4.0
                elif beacon > 12:
                    beacon_reward = -2.0
            elif density_category in ["VERY_LOW", "LOW"]:
                if beacon >= 10:  # Can use higher rates with sectoral
                    beacon_reward = 3.0
                elif beacon < 4:
                    beacon_reward = -2.0
        else:
            # Original omnidirectional logic
            if density_category in ["HIGH", "VERY_HIGH", "EXTREME"]:
                if beacon <= 8:
                    beacon_reward = 3.0
                elif beacon > 15:
                    beacon_reward = -3.0
            elif density_category in ["VERY_LOW", "LOW"]:
                if beacon >= 12:
                    beacon_reward = 2.0
                elif beacon < 5:
                    beacon_reward = -2.0
        
        # MCS efficiency reward with antenna awareness
        mcs_reward = 0.0
        if antenna_type.upper() == "SECTORAL":
            # Sectoral can use higher MCS in high density due to better interference management
            if density_category in ["HIGH", "VERY_HIGH", "EXTREME"] and mcs <= 5:
                mcs_reward = 2.5
            elif density_category in ["VERY_LOW", "LOW"] and mcs >= 6:
                mcs_reward = 2.5
        else:
            # Original omnidirectional MCS logic
            if density_category in ["HIGH", "VERY_HIGH", "EXTREME"] and mcs <= 4:
                mcs_reward = 2.0
            elif density_category in ["VERY_LOW", "LOW"] and mcs >= 6:
                mcs_reward = 2.0
        
        # Action smoothness penalty
        beacon_change = abs(next_beacon - beacon)
        mcs_change = abs(next_mcs - mcs)
        smoothness_penalty = -0.5 * (beacon_change + mcs_change)
        
        total_reward = cbr_reward + beacon_reward + mcs_reward + smoothness_penalty
        
        return np.clip(total_reward, -20, 20)
    
    def update_q_table(self, state_indices, action, reward, next_state_indices):
        """Update Q-table using Q-learning rule - SAME AS ORIGINAL"""
        current_q = self.q_table[state_indices][action]
        max_next_q = np.max(self.q_table[next_state_indices])
        
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q - current_q)
        self.q_table[state_indices][action] = new_q
        
        return new_q - current_q  # TD error

# ================== Enhanced PHY Agent==================
# ================== Simplified PHY Agent (REPLACE EXISTING CLASS) ==================
class PHYAgent:
    """PHY Agent - Controls transmission power with SINR focus"""
    
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon
        self.q_table = phy_q_table
        self.state_visit_counts = defaultdict(int)
        
    def get_state_indices(self, cbr, sinr, power, neighbors):
        """Convert continuous state to discrete indices - SAME AS ORIGINAL"""
        cbr_idx = discretize(cbr, CBR_BINS)
        sinr_idx = discretize(sinr, SINR_BINS)
        power_idx = discretize(power, POWER_BINS)
        neighbors_idx = discretize(neighbors, NEIGHBORS_BINS)
        
        return (cbr_idx, sinr_idx, power_idx, neighbors_idx)
    
    def select_action(self, state_indices, neighbor_count, current_sinr, antenna_type="OMNIDIRECTIONAL"):
        """Select action with internal antenna awareness but same interface"""
        # Track state visits
        self.state_visit_counts[state_indices] += 1
        
        # Internal antenna-aware adaptation
        density_category = get_neighbor_category(neighbor_count, antenna_type)
        adaptive_epsilon = self.epsilon
        
        # Internal logic for antenna-specific exploration
        if antenna_type.upper() == "SECTORAL":
            if current_sinr < 8.0:  # Poor SINR with sectoral (can be more aggressive)
                adaptive_epsilon = min(1.0, self.epsilon * 1.4)
        else:
            if current_sinr < 6.0:  # Poor SINR with omnidirectional
                adaptive_epsilon = min(1.0, self.epsilon * 1.3)
        
        if random.random() < adaptive_epsilon:
            # 20% chance for random jump to any discrete power value (during high exploration)
            if self.epsilon > 0.5 and random.random() < 0.2:
                # Direct jump to random discrete power
                target_power = random.randint(1, 30)
                current_power = state_indices[2] + 1  # Convert back from index
                power_delta = target_power - current_power
                # Find closest action
                best_action = 0
                best_diff = float('inf')
                for i, p in enumerate(PHY_ACTIONS):
                    if abs(p - power_delta) < best_diff:
                        best_diff = abs(p - power_delta)
                        best_action = i
                return best_action
            
            # Internal antenna-aware biased exploration
            if antenna_type.upper() == "SECTORAL":
                # Sectoral can be more power-efficient
                if density_category in ["HIGH", "VERY_HIGH", "EXTREME"] and current_sinr > 12:
                    # Can reduce power more aggressively with sectoral
                    preferred_actions = [i for i, p in enumerate(PHY_ACTIONS) if p <= -1]
                    if preferred_actions and random.random() < 0.8:
                        return random.choice(preferred_actions)
                elif density_category in ["VERY_LOW", "LOW"] and current_sinr < 12:
                    # Might need more power for coverage
                    preferred_actions = [i for i, p in enumerate(PHY_ACTIONS) if p >= 1]
                    if preferred_actions and random.random() < 0.7:
                        return random.choice(preferred_actions)
            else:
                # Original omnidirectional strategy
                if density_category in ["HIGH", "VERY_HIGH", "EXTREME"] and current_sinr > 10:
                    preferred_actions = [i for i, p in enumerate(PHY_ACTIONS) if p <= 0]
                    if preferred_actions and random.random() < 0.7:
                        return random.choice(preferred_actions)
                elif density_category in ["VERY_LOW", "LOW"] and current_sinr < 10:
                    preferred_actions = [i for i, p in enumerate(PHY_ACTIONS) if p >= 0]
                    if preferred_actions and random.random() < 0.7:
                        return random.choice(preferred_actions)
            
            return random.randint(0, PHY_ACTION_DIM - 1)
        else:
            # Exploit
            return np.argmax(self.q_table[state_indices])
    
    def calculate_reward(self, cbr, sinr, power, neighbors, next_sinr, next_power, antenna_type="OMNIDIRECTIONAL"):
        """PHY-specific reward with internal antenna awareness"""
        # Adaptive SINR target based on antenna type (internal logic)
        if antenna_type.upper() == "SECTORAL":
            # Sectoral antennas can achieve better SINR performance
            base_sinr_target = 15.0 - (neighbors / 12) * 4.0  # Better performance in high density
        else:
            # Original omnidirectional target
            base_sinr_target = 14.0 - (neighbors / 10) * 4.0
        
        sinr_target = max(8.0, base_sinr_target)  # Minimum reasonable target
        
        # SINR optimization reward
        sinr_reward = 8.0 * math.tanh((sinr - sinr_target) / 5.0)
        
        # Power efficiency reward with antenna awareness
        power_norm = (power - POWER_MIN) / (POWER_MAX - POWER_MIN)
        density_category = get_neighbor_category(neighbors, antenna_type)
        
        power_reward = 0.0
        if antenna_type.upper() == "SECTORAL":
            # Sectoral antennas can be more power efficient
            if density_category in ["HIGH", "VERY_HIGH", "EXTREME"]:
                if power_norm <= 0.4:  # Very efficient with sectoral
                    power_reward = 5.0
                elif power_norm > 0.8:
                    power_reward = -2.0
            elif density_category in ["VERY_LOW", "LOW"]:
                if 0.3 <= power_norm <= 0.7:
                    power_reward = 3.0
                elif power_norm < 0.2:
                    power_reward = -1.0
        else:
            # Original omnidirectional efficiency logic
            if density_category in ["HIGH", "VERY_HIGH", "EXTREME"]:
                if power_norm <= 0.5:
                    power_reward = 4.0
                elif power_norm > 0.8:
                    power_reward = -2.0
            elif density_category in ["VERY_LOW", "LOW"]:
                if 0.3 <= power_norm <= 0.8:
                    power_reward = 2.0
                elif power_norm < 0.2:
                    power_reward = -1.0
        
        # SINR-power efficiency bonus
        if next_sinr >= 12.0 and power_norm <= 0.6:
            power_reward += 3.0  # Bonus for efficient operation
        
        # Action smoothness
        power_change = abs(next_power - power)
        smoothness_penalty = -0.3 * power_change
        
        total_reward = sinr_reward + power_reward + smoothness_penalty
        
        return np.clip(total_reward, -20, 20)
    
    def update_q_table(self, state_indices, action, reward, next_state_indices):
        """Update Q-table using Q-learning rule - SAME AS ORIGINAL"""
        current_q = self.q_table[state_indices][action]
        max_next_q = np.max(self.q_table[next_state_indices])
        
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q - current_q)
        self.q_table[state_indices][action] = new_q
        
        return new_q - current_q  # TD error

# ================== Centralized Learning Manager ==================
class CentralizedLearningManager:
    """Manages coordinated learning between MAC and PHY agents"""
    
    def __init__(self, mac_agent, phy_agent):
        self.mac_agent = mac_agent
        self.phy_agent = phy_agent
        self.experience_buffer = deque(maxlen=BUFFER_SIZE)
        self.update_counter = 0
        self.batch_size = 32
        
    def add_experience(self, experience):
        """Add experience to buffer"""
        self.experience_buffer.append(experience)
        self.update_counter += 1
        
        # Batch update every N experiences
        if self.update_counter % self.batch_size == 0 and len(self.experience_buffer) >= self.batch_size:
            self.perform_batch_update()
    
    def perform_batch_update(self):
        """Perform batch Q-learning updates"""
        # Sample batch from buffer
        batch = random.sample(self.experience_buffer, self.batch_size)
        
        mac_td_errors = []
        phy_td_errors = []
        
        for exp in batch:
            # Update MAC agent
            mac_td = self.mac_agent.update_q_table(
                exp['mac_state'], exp['mac_action'], 
                exp['mac_reward'], exp['next_mac_state']
            )
            mac_td_errors.append(abs(mac_td))
            
            # Update PHY agent
            phy_td = self.phy_agent.update_q_table(
                exp['phy_state'], exp['phy_action'],
                exp['phy_reward'], exp['next_phy_state']
            )
            phy_td_errors.append(abs(phy_td))
        
        # Log average TD errors
        if self.update_counter % 100 == 0:
            logger.info(f"Batch update {self.update_counter//self.batch_size}: "
                       f"MAC TD error: {np.mean(mac_td_errors):.4f}, "
                       f"PHY TD error: {np.mean(phy_td_errors):.4f}")

# ================== Performance Tracking ==================
class DualAgentPerformanceMetrics:
    def __init__(self):
        self.reset_metrics()
        self.episode_data = []
        
    def reset_metrics(self):
        self.mac_rewards = []
        self.phy_rewards = []
        self.joint_rewards = []
        self.cbr_values = []
        self.sinr_values = []
        self.mac_actions = []
        self.phy_actions = []
        self.neighbor_counts = []
        
    def add_step(self, mac_reward, phy_reward, cbr, sinr, neighbors, mac_action, phy_action):
        self.mac_rewards.append(mac_reward)
        self.phy_rewards.append(phy_reward)
        self.joint_rewards.append(0.5 * mac_reward + 0.5 * phy_reward)
        self.cbr_values.append(cbr)
        self.sinr_values.append(sinr)
        self.neighbor_counts.append(neighbors)
        self.mac_actions.append(mac_action)
        self.phy_actions.append(phy_action)
    
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
            'sinr_above_10_rate': sum(1 for sinr in self.sinr_values if sinr >= 10) / len(self.sinr_values),
            
            # Density analysis
            'avg_neighbors': np.mean(self.neighbor_counts),
            'high_density_rate': sum(1 for n in self.neighbor_counts if n > 8) / len(self.neighbor_counts),
            
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
                # Episode summary
                if self.episode_data:
                    episode_df = pd.DataFrame(self.episode_data)
                    episode_df.to_excel(writer, sheet_name='Episode_Summary', index=False)
                
                # Dual agent analysis
                analysis_data = {
                    'Metric': ['MAC Avg Reward', 'PHY Avg Reward', 'Joint Avg Reward',
                               'CBR Performance', 'SINR Performance', 'High Density Adaptation'],
                    'Value': [
                        np.mean([d['avg_mac_reward'] for d in self.episode_data[-10:]]),
                        np.mean([d['avg_phy_reward'] for d in self.episode_data[-10:]]),
                        np.mean([d['avg_joint_reward'] for d in self.episode_data[-10:]]),
                        np.mean([d['cbr_in_range_rate'] for d in self.episode_data[-10:]]),
                        np.mean([d['sinr_above_10_rate'] for d in self.episode_data[-10:]]),
                        np.mean([d['high_density_rate'] for d in self.episode_data[-10:]])
                    ]
                }
                analysis_df = pd.DataFrame(analysis_data)
                analysis_df.to_excel(writer, sheet_name='Dual_Agent_Analysis', index=False)
                
            logger.info(f"Performance data saved to {PERFORMANCE_LOG_PATH}")
            
        except Exception as e:
            logger.error(f"Error saving to Excel: {e}")

# ================== Dual-Agent Q-Learning Implementation ==================
# ================== Dual-Agent Q-Learning Implementation (REPLACE ENTIRE CLASS) ==================
class DualAgentQLearning:
    def __init__(self, training_mode=True):
        self.training_mode = training_mode
        self.mac_agent = MACAgent(epsilon=EPSILON if training_mode else 0.0)
        self.phy_agent = PHYAgent(epsilon=EPSILON if training_mode else 0.0)
        self.centralized_manager = CentralizedLearningManager(self.mac_agent, self.phy_agent)
        self.performance = DualAgentPerformanceMetrics()
        self.episode_count = 0
        
        # Load pre-trained models if they exist
        self.load_models()
    
    def process_vehicle(self, veh_id, veh_info):
        """Process vehicle with same input/output interface as original Q-learning"""
        try:
            # Extract current state - SAME AS ORIGINAL Q-LEARNING
            cbr = float(veh_info.get("CBR", 0.65))
            sinr = float(veh_info.get("SINR", veh_info.get("SNR", 20)))
            neighbors = int(veh_info.get("neighbors", 5))
            current_power = float(veh_info.get("transmissionPower", 15))  # Same as original
            current_beacon = float(veh_info.get("beaconRate", 10))
            current_mcs = int(veh_info.get("MCS", 5))
            
            # Use the configured antenna type (no changes needed to VANET script)
            antenna_type = ANTENNA_TYPE  # From configuration
            
            # Validate and clamp inputs - SAME AS ORIGINAL
            current_power = np.clip(current_power, POWER_MIN, POWER_MAX)
            current_beacon = np.clip(current_beacon, BEACON_MIN, BEACON_MAX)
            current_mcs = np.clip(current_mcs, 0, 9)
            cbr = np.clip(cbr, 0.0, 1.0)
            sinr = np.clip(sinr, 0, 50)
            neighbors = max(0, neighbors)
            
            # Handle NaN values - SAME AS ORIGINAL
            if not np.isfinite(cbr):
                cbr = 0.65
            if not np.isfinite(sinr):
                sinr = 20.0
            if not np.isfinite(current_power):
                current_power = 15.0
            if not np.isfinite(current_beacon):
                current_beacon = 10.0
            
            # Special handling for sectoral antenna power
            # If sectoral, treat transmissionPower as composite of front+rear
            if antenna_type == "SECTORAL":
                # Assume transmissionPower is the total/average power for sectoral
                # This allows the same interface while optimizing for sectoral characteristics
                effective_power = current_power  # Use as-is, but optimize knowing it's sectoral
            else:
                effective_power = current_power
            
            # Get MAC state indices
            mac_state_indices = self.mac_agent.get_state_indices(
                cbr, sinr, current_beacon, current_mcs, neighbors
            )
            
            # Get PHY state indices
            phy_state_indices = self.phy_agent.get_state_indices(
                cbr, sinr, effective_power, neighbors
            )
            
            # Select actions with configured antenna awareness
            mac_action_idx = self.mac_agent.select_action(mac_state_indices, neighbors, antenna_type)
            phy_action_idx = self.phy_agent.select_action(phy_state_indices, neighbors, sinr, antenna_type)
            
            # Apply actions
            beacon_delta, mcs_delta = MAC_ACTIONS[mac_action_idx]
            power_delta = PHY_ACTIONS[phy_action_idx]
            
            new_beacon = np.clip(current_beacon + beacon_delta, BEACON_MIN, BEACON_MAX)
            new_mcs = np.clip(current_mcs + mcs_delta, 0, 9)
            new_power = np.clip(effective_power + power_delta, POWER_MIN, POWER_MAX)
            
            # Training updates
            if self.training_mode:
                # Simulate next state
                next_cbr = cbr + random.uniform(-0.05, 0.05)
                next_cbr = np.clip(next_cbr, 0, 1)
                next_sinr = sinr + random.uniform(-2, 2)
                
                # Calculate rewards with antenna awareness
                mac_reward = self.mac_agent.calculate_reward(
                    cbr, sinr, current_beacon, current_mcs, neighbors,
                    next_cbr, new_beacon, new_mcs, antenna_type
                )
                
                phy_reward = self.phy_agent.calculate_reward(
                    cbr, sinr, effective_power, neighbors,
                    next_sinr, new_power, antenna_type
                )
                
                # Get next state indices
                next_mac_state = self.mac_agent.get_state_indices(
                    next_cbr, next_sinr, new_beacon, new_mcs, neighbors
                )
                next_phy_state = self.phy_agent.get_state_indices(
                    next_cbr, next_sinr, new_power, neighbors
                )
                
                # Store experience
                experience = {
                    'mac_state': mac_state_indices,
                    'mac_action': mac_action_idx,
                    'mac_reward': mac_reward,
                    'next_mac_state': next_mac_state,
                    'phy_state': phy_state_indices,
                    'phy_action': phy_action_idx,
                    'phy_reward': phy_reward,
                    'next_phy_state': next_phy_state
                }
                
                self.centralized_manager.add_experience(experience)
                self.performance.add_step(
                    mac_reward, phy_reward, cbr, sinr, neighbors,
                    mac_action_idx, phy_action_idx
                )
                
                # Decay exploration
                self.mac_agent.epsilon = max(MIN_EPSILON, self.mac_agent.epsilon * EPSILON_DECAY)
                self.phy_agent.epsilon = max(MIN_EPSILON, self.phy_agent.epsilon * EPSILON_DECAY)
            
            # Enhanced logging with antenna type info
            if veh_id.endswith('0'):  # Log every 10th vehicle
                density_cat = get_neighbor_category(neighbors, antenna_type)
                logger.info(f"Vehicle {veh_id} [{antenna_type}][{density_cat}]: "
                           f"CBR={cbr:.3f}, SINR={sinr:.1f}dB, Neighbors={neighbors}")
                logger.info(f"  MAC: Beacon {current_beacon:.0f}->{new_beacon:.0f}Hz, "
                           f"MCS {current_mcs}->{new_mcs}")
                logger.info(f"  PHY: Power {effective_power:.0f}->{new_power:.0f}dBm")
            
            # Return EXACT SAME FORMAT as original Q-learning
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
                           f"CBR in range={metrics['cbr_in_range_rate']:.2%}")
            
            if self.episode_count % MODEL_SAVE_INTERVAL == 0:
                self.save_models()
            
            self.performance.reset_metrics()
    
    def save_models(self):
        """Save Q-tables"""
        try:
            np.save(MAC_MODEL_PATH, self.mac_agent.q_table)
            np.save(PHY_MODEL_PATH, self.phy_agent.q_table)
            logger.info(f"Models saved: {MAC_MODEL_PATH}, {PHY_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Load pre-trained Q-tables"""
        try:
            if os.path.exists(MAC_MODEL_PATH):
                loaded_mac = np.load(MAC_MODEL_PATH)
                if loaded_mac.shape == self.mac_agent.q_table.shape:
                    self.mac_agent.q_table = loaded_mac
                    logger.info(f"Loaded MAC model from {MAC_MODEL_PATH}")
            
            if os.path.exists(PHY_MODEL_PATH):
                loaded_phy = np.load(PHY_MODEL_PATH)
                if loaded_phy.shape == self.phy_agent.q_table.shape:
                    self.phy_agent.q_table = loaded_phy
                    logger.info(f"Loaded PHY model from {PHY_MODEL_PATH}")
                    
        except Exception as e:
            logger.error(f"Error loading models: {e}")

# ================== Server Implementation ==================
class DualAgentRLServer:
    def __init__(self, host, port, training_mode=True):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(5)
        self.dual_agent = DualAgentQLearning(training_mode=training_mode)
        self.training_mode = training_mode
        self.running = True
        
        mode_str = "TRAINING" if training_mode else "TESTING"
        logger.info(f"Dual-Agent RL Server started in {mode_str} mode on {host}:{port}")

    def receive_message_with_header(self, conn):
        """Receive message with 4-byte length header"""
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
        """Send message with 4-byte length header"""
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
        """Handle client connection"""
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
                    
                    # End episode periodically during training
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
        """Start the server"""
        try:
            logger.info("Dual-Agent RL Server listening for connections...")
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
        """Stop the server"""
        logger.info("Stopping Dual-Agent RL server...")
        self.running = False
        
        try:
            self.server.close()
        except:
            pass
        
        if self.training_mode:
            self.dual_agent.save_models()
            self.dual_agent.performance.save_to_excel()
            logger.info("Final models and performance data saved")
        
        logger.info("Dual-Agent RL server stopped")

# ================== Main Execution ==================
# ================== Main Execution (REPLACE THE main() FUNCTION) ==================
def main():
    # Validate operation mode
    if OPERATION_MODE.upper() not in ["TRAINING", "TESTING"]:
        print(f"ERROR: Invalid OPERATION_MODE '{OPERATION_MODE}'. Must be 'TRAINING' or 'TESTING'")
        sys.exit(1)
    
    # Validate antenna type
    if ANTENNA_TYPE.upper() not in ["SECTORAL", "OMNIDIRECTIONAL"]:
        print(f"ERROR: Invalid ANTENNA_TYPE '{ANTENNA_TYPE}'. Must be 'SECTORAL' or 'OMNIDIRECTIONAL'")
        sys.exit(1)
    
    training_mode = (OPERATION_MODE.upper() == "TRAINING")
    
    print("="*80)
    print(" DUAL-AGENT Q-LEARNING VANET SERVER")
    print(f"Host: {HOST}:{PORT}")
    print(f"Mode: {OPERATION_MODE.upper()}")
    print(f"Antenna Type: {ANTENNA_TYPE.upper()}")
    print(f"Target CBR: {CBR_TARGET}")
    print("Architecture:")
    print("  • MAC Agent: Controls beacon rate & MCS (CBR-focused)")
    print("  • PHY Agent: Controls transmission power (SINR-focused)")
    print("  • Antenna-aware optimization")
    print("  • Centralized learning manager")
    print("Discrete Parameter Ranges:")
    print("  • Power: 1-30 dBm (30 discrete values)")
    print("  • Beacon Rate: 1-20 Hz (20 discrete values)")
    print("  • MCS: 0-9 (10 discrete values)")
    print("Enhanced Features:")
    print("  • Large action jumps (±10, ±15) for full range coverage")
    print("  • Random jumps to any discrete value during high exploration")
    print("  • Antenna-aware biased exploration")
    print("  • Density-aware neighbor classification")
    if ANTENNA_TYPE.upper() == "SECTORAL":
        print("  • Sectoral antenna optimizations:")
        print("    - Better high-density performance")
        print("    - More power-efficient operation")
        print("    - Enhanced interference management")
    else:
        print("  • Omnidirectional antenna optimizations:")
        print("    - Balanced coverage optimization")
        print("    - Conservative power management")
    if training_mode:
        print(f"Learning Rate: {LEARNING_RATE}")
        print(f"Initial Epsilon: {EPSILON}")
        print(f"Models will be saved every {MODEL_SAVE_INTERVAL} episodes")
    else:
        print("Using pre-trained models")
    print(f"Model files: {MAC_MODEL_PATH}, {PHY_MODEL_PATH}")
    print("="*80)
    
    # Initialize server
    rl_server = DualAgentRLServer(HOST, PORT, training_mode=training_mode)
    
    try:
        rl_server.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        rl_server.stop()

if __name__ == "__main__":
    main()
