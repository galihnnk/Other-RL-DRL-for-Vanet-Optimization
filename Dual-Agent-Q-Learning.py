"""
DUAL-AGENT Q-LEARNING WITH SCIENTIFIC DENSITY CATEGORIZATION by Galih Nugraha Nurkahfi, galih.nugraha.nurkahfi@brin.go.id
=========================================================================================================================
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

# ================== CONFIGURATION ==================
OPERATION_MODE = "TESTING"        # Options: "TRAINING" or "TESTING"
ANTENNA_TYPE = "SECTORAL"          # Options: "SECTORAL" or "OMNIDIRECTIONAL"

# ================== Constants ==================
CBR_TARGET = 0.4                   # Better latency/PDR performance
CBR_RANGE = (0.35, 0.45)          # Acceptable CBR range
SINR_TARGET = 12.0                 # Fixed SINR target
SINR_GOOD_THRESHOLD = 12.0         # Threshold for diminishing returns

BUFFER_SIZE = 100000
LEARNING_RATE = 0.15
DISCOUNT_FACTOR = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.1
HOST = '127.0.0.1'
PORT = 5000

# Power and beacon ranges
POWER_MIN = 1
POWER_MAX = 30
BEACON_MIN = 1
BEACON_MAX = 20

# File paths
MODEL_PREFIX = f"{ANTENNA_TYPE.lower()}_dual_agent"
MAC_MODEL_PATH = f'{MODEL_PREFIX}_mac_qlearning_model.npy'
PHY_MODEL_PATH = f'{MODEL_PREFIX}_phy_qlearning_model.npy'
PERFORMANCE_LOG_PATH = f'{MODEL_PREFIX}_performance.xlsx'
MODEL_SAVE_INTERVAL = 50
PERFORMANCE_LOG_INTERVAL = 10

# ================== DENSITY CATEGORIZATION ==================
# Based on VANET system with antenna awareness
def get_neighbor_category(neighbor_count, antenna_type="OMNIDIRECTIONAL"):
    """
    REVISED: Evidence-based density categorization aligned with VANET system
    
    Base thresholds from VANET research:
    - LOW: ≤10 neighbors (Expected SINR: 15-30 dB)
    - MEDIUM: ≤20 neighbors (Expected SINR: 8-20 dB) 
    - HIGH: ≤30 neighbors (Expected SINR: 2-12 dB)
    - VERY_HIGH: >30 neighbors (Expected SINR: -5 to 8 dB)
    
    Antenna awareness: Sectoral antennas can handle ~20-30% higher density
    due to spatial filtering and reduced interference.
    """
    if antenna_type.upper() == "SECTORAL":
        # Sectoral can handle higher densities due to spatial filtering
        if neighbor_count <= 13:      # ~30% higher than 10
            return "LOW"
        elif neighbor_count <= 26:    # ~30% higher than 20
            return "MEDIUM"
        elif neighbor_count <= 40:    # ~33% higher than 30
            return "HIGH"
        else:
            return "VERY_HIGH"
    else:
        # Omnidirectional - use VANET system directly
        if neighbor_count <= 10:
            return "LOW"
        elif neighbor_count <= 20:
            return "MEDIUM"
        elif neighbor_count <= 30:
            return "HIGH"
        else:
            return "VERY_HIGH"

def get_expected_sinr_range(neighbor_count, antenna_type="OMNIDIRECTIONAL"):
    """
    Get expected SINR range based on density category and antenna type
    
    Base ranges from VANET system, adjusted for antenna characteristics
    """
    category = get_neighbor_category(neighbor_count, antenna_type)
    
    if antenna_type.upper() == "SECTORAL":
        # Sectoral antennas achieve ~3-5 dB better SINR due to directional gain
        ranges = {
            "LOW": (18, 35),        # +3 dB from (15-30)
            "MEDIUM": (11, 25),     # +3-5 dB from (8-20)
            "HIGH": (5, 17),        # +3-5 dB from (2-12) 
            "VERY_HIGH": (-2, 13)   # +3-5 dB from (-5 to 8)
        }
    else:
        # Omnidirectional - use VANET ranges directly
        ranges = {
            "LOW": (15, 30),
            "MEDIUM": (8, 20),
            "HIGH": (2, 12),
            "VERY_HIGH": (-5, 8)
        }
    
    return ranges.get(category, (8, 20))

def get_density_multiplier(neighbor_count, antenna_type="OMNIDIRECTIONAL"):
    """Density-based reward multiplier using revised categories"""
    category = get_neighbor_category(neighbor_count, antenna_type)
    multipliers = {
        "LOW": 0.8,
        "MEDIUM": 1.0,        # Baseline
        "HIGH": 1.4,
        "VERY_HIGH": 1.8
    }
    return multipliers.get(category, 1.0)

# ================== State Discretization ==================
CBR_BINS = np.linspace(0.0, 1.0, 21)
SINR_BINS = np.linspace(0, 50, 11)
BEACON_BINS = np.arange(1, 21)
MCS_BINS = np.arange(0, 10)
NEIGHBORS_BINS = np.linspace(0, 50, 11)
POWER_BINS = np.arange(1, 31)

MAC_STATE_DIM = (20, 10, 20, 10, 10)
PHY_STATE_DIM = (20, 10, 30, 10)

# ================== Action Spaces ==================
MAC_ACTIONS = [
    (0, 0), (1, 0), (-1, 0), (2, 0), (-2, 0), (3, 0), (-3, 0), (5, 0), (-5, 0),
    (0, 1), (0, -1), (0, 2), (0, -2), (1, 1), (1, -1), (-1, 1), (-1, -1),
    (2, 1), (-2, -1), (10, 0), (-10, 0), (0, 5), (0, -5),
]

PHY_ACTIONS = [0, 1, -1, 2, -2, 3, -3, 5, -5, 10, -10, 15, -15]

MAC_ACTION_DIM = len(MAC_ACTIONS)
PHY_ACTION_DIM = len(PHY_ACTIONS)

# Initialize Q-tables
mac_q_table = np.zeros(MAC_STATE_DIM + (MAC_ACTION_DIM,), dtype=np.float32)
phy_q_table = np.zeros(PHY_STATE_DIM + (PHY_ACTION_DIM,), dtype=np.float32)

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

logger.info("DUAL-AGENT Q-LEARNING - REVISED DENSITY SYSTEM")
logger.info(f"CBR Target: {CBR_TARGET}")
logger.info(f"SINR Target: {SINR_TARGET} dB (fixed)")
logger.info(f"Antenna Type: {ANTENNA_TYPE}")
logger.info("Density Categories: LOW(≤10/13), MEDIUM(≤20/26), HIGH(≤30/40), VERY_HIGH(>30/40)")

# ================== Helper Functions ==================
def discretize(value, bins):
    """Discretize a continuous value into a bin index"""
    if not np.isfinite(value):
        value = 0.0
    
    if len(bins) <= 30 and bins[0] == 1 and bins[-1] in [20, 30]:
        value = int(np.clip(value, bins[0], bins[-1]))
        return value - bins[0]
    elif len(bins) == 10 and bins[0] == 0 and bins[-1] == 9:
        value = int(np.clip(value, 0, 9))
        return value
    else:
        value = np.clip(value, bins[0], bins[-1])
        bin_idx = np.digitize(value, bins) - 1
        return max(0, min(len(bins) - 2, bin_idx))
    
def get_optimal_parameters(neighbors, antenna_type="OMNIDIRECTIONAL"):
    """
    NEW HELPER FUNCTION: Calculate optimal parameters based on neighbor density
    
    Uses logarithmic scaling for smooth parameter adjustment
    """
    
    # Logarithmic scaling factors
    beacon_factor = 1.0 - (0.3 * math.log(1 + neighbors) / math.log(1 + 40))
    mcs_factor = 1.0 - (0.4 * math.log(1 + neighbors) / math.log(1 + 50))
    power_factor = 0.3 + (0.4 * math.log(1 + neighbors) / math.log(1 + 40))
    
    # Base parameters
    optimal_beacon = 12.0 * beacon_factor
    optimal_mcs = 8.0 * mcs_factor
    optimal_power_norm = power_factor
    
    # Antenna adjustments
    if antenna_type.upper() == "SECTORAL":
        optimal_beacon *= 1.1      # 10% higher beacon
        optimal_mcs *= 1.2         # 20% higher MCS
        optimal_power_norm *= 0.8  # 20% lower power need
    
    return {
        'beacon': optimal_beacon,
        'mcs': optimal_mcs,
        'power_norm': optimal_power_norm
    }

# ================== Enhanced SINR Reward Function ==================
def calculate_sinr_reward(sinr, power_norm, neighbors, antenna_type="OMNIDIRECTIONAL"):
    """
    REPLACE THE EXISTING calculate_sinr_reward FUNCTION WITH THIS IMPROVED VERSION
    
    Key improvements:
    - Quadratic power penalty (much stronger deterrent)
    - Logarithmic neighbor scaling (smoother than categorical)
    - Maintains fixed SINR target and two-phase structure
    """
    SINR_TARGET = 12.0
    
    # Phase 1: Below target - aggressive improvement needed
    if sinr < SINR_TARGET:
        base_reward = 10.0 * (sinr / SINR_TARGET)
        
        # IMPROVED: Logarithmic neighbor penalty (smoother than categorical bins)
        neighbor_penalty = -2.0 * math.log(1 + neighbors) / math.log(1 + 30)
        sinr_reward = base_reward + neighbor_penalty
        
    else:
        # Phase 2: Above target - diminishing returns
        base_reward = 10.0
        excess_sinr = sinr - SINR_TARGET
        diminishing_reward = 5.0 * math.sqrt(excess_sinr / 10.0)
        sinr_reward = base_reward + diminishing_reward
        sinr_reward = min(sinr_reward, 18.0)
    
    # IMPROVED: Quadratic power penalty when SINR is sufficient
    if sinr >= SINR_TARGET:
        if power_norm > 0.6:
            # Quadratic penalty grows MUCH faster than linear
            power_penalty = -8.0 * ((power_norm - 0.6) ** 2)
            sinr_reward += power_penalty
        elif power_norm <= 0.4:
            # Quadratic efficiency bonus
            efficiency_bonus = 4.0 * ((0.4 - power_norm) ** 2)
            sinr_reward += efficiency_bonus
    
    # IMPROVED: Logarithmic neighbor impact for excessive SINR
    if sinr > 20.0:
        neighbor_impact_penalty = -3.0 * math.log(1 + neighbors) / math.log(1 + 50) * (sinr - 20.0) / 10.0
        sinr_reward += neighbor_impact_penalty
    
    return np.clip(sinr_reward, -15, 20)

# ================== REVISED Agent Classes ==================
class MACAgent:
    """MAC Agent with revised density awareness"""
    
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon
        self.q_table = mac_q_table
        self.state_visit_counts = defaultdict(int)
        
    def get_state_indices(self, cbr, sinr, beacon, mcs, neighbors):
        cbr_idx = discretize(cbr, CBR_BINS)
        sinr_idx = discretize(sinr, SINR_BINS)
        beacon_idx = discretize(beacon, BEACON_BINS)
        mcs_idx = discretize(mcs, MCS_BINS)
        neighbors_idx = discretize(neighbors, NEIGHBORS_BINS)
        
        return (cbr_idx, sinr_idx, beacon_idx, mcs_idx, neighbors_idx)
    
    def select_action(self, state_indices, neighbor_count, antenna_type="OMNIDIRECTIONAL"):
        """REVISED: Action selection using new density categories"""
        self.state_visit_counts[state_indices] += 1
        
        density_category = get_neighbor_category(neighbor_count, antenna_type)
        adaptive_epsilon = self.epsilon
        
        # Antenna and density-aware exploration adjustment
        if antenna_type.upper() == "SECTORAL":
            if density_category in ["HIGH", "VERY_HIGH"]:
                adaptive_epsilon = min(1.0, self.epsilon * 1.2)  # More conservative
        
        if random.random() < adaptive_epsilon:
            # Random jump exploration
            if self.epsilon > 0.5 and random.random() < 0.2:
                if random.random() < 0.5:  # Beacon exploration
                    target_beacon = random.randint(1, 20)
                    current_beacon = state_indices[2] + 1
                    beacon_delta = target_beacon - current_beacon
                    best_action = 0
                    best_diff = float('inf')
                    for i, (b, m) in enumerate(MAC_ACTIONS):
                        if abs(b - beacon_delta) < best_diff:
                            best_diff = abs(b - beacon_delta)
                            best_action = i
                    return best_action
                else:  # MCS exploration
                    target_mcs = random.randint(0, 9)
                    current_mcs = state_indices[3]
                    mcs_delta = target_mcs - current_mcs
                    best_action = 0
                    best_diff = float('inf')
                    for i, (b, m) in enumerate(MAC_ACTIONS):
                        if abs(m - mcs_delta) < best_diff:
                            best_diff = abs(m - mcs_delta)
                            best_action = i
                    return best_action
            
            # Density-aware biased exploration
            if density_category in ["HIGH", "VERY_HIGH"]:
                # Conservative beacon actions in high density
                preferred_actions = [i for i, (b, m) in enumerate(MAC_ACTIONS) if b <= 1]
                if preferred_actions and random.random() < 0.8:
                    return random.choice(preferred_actions)
            elif density_category == "LOW":
                # Can use higher beacon rates in low density
                preferred_actions = [i for i, (b, m) in enumerate(MAC_ACTIONS) if b >= -1]
                if preferred_actions and random.random() < 0.7:
                    return random.choice(preferred_actions)
            
            return random.randint(0, MAC_ACTION_DIM - 1)
        else:
            return np.argmax(self.q_table[state_indices])
    
    def calculate_reward(self, cbr, sinr, beacon, mcs, neighbors, next_cbr, next_beacon, next_mcs, antenna_type="OMNIDIRECTIONAL"):
        """
        REPLACE THE calculate_reward METHOD IN MACAgent CLASS WITH THIS
        
        Improved mathematical formulations:
        - Logarithmic neighbor scaling for beacon optimization
        - Quadratic penalties for parameter deviations
        - Smoother reward landscape
        """
        
        # Primary CBR optimization (keep existing formula - it's already good)
        cbr_error = abs(cbr - 0.4)  # CBR_TARGET = 0.4
        cbr_reward = 10.0 * (1 - math.tanh(25 * cbr_error))
        
        # IMPROVED: Beacon optimization with logarithmic neighbor scaling
        # Optimal beacon rate decreases logarithmically with neighbor density
        optimal_beacon_factor = 1.0 - (0.3 * math.log(1 + neighbors) / math.log(1 + 40))
        optimal_beacon = 12.0 * optimal_beacon_factor  # Base beacon 12, scaled by density
        
        # Antenna awareness
        if antenna_type.upper() == "SECTORAL":
            optimal_beacon *= 1.1  # Sectoral can handle 10% higher beacon rates
        
        # IMPROVED: Quadratic beacon penalty (stronger than linear)
        beacon_error = abs(beacon - optimal_beacon)
        beacon_reward = -2.0 * (beacon_error / 5.0) ** 2
        
        # IMPROVED: MCS optimization with logarithmic neighbor consideration
        optimal_mcs_factor = 1.0 - (0.4 * math.log(1 + neighbors) / math.log(1 + 50))
        optimal_mcs = 8.0 * optimal_mcs_factor  # Base MCS 8, reduced by density
        
        if antenna_type.upper() == "SECTORAL":
            optimal_mcs *= 1.2  # Sectoral can handle 20% higher MCS
        
        # IMPROVED: Quadratic MCS penalty
        mcs_error = abs(mcs - optimal_mcs)
        mcs_reward = -1.5 * (mcs_error / 3.0) ** 2
        
        # IMPROVED: Logarithmic smoothness penalty (more forgiving than linear)
        beacon_change = abs(next_beacon - beacon)
        mcs_change = abs(next_mcs - mcs)
        smoothness_penalty = -1.0 * (math.log(1 + beacon_change) + math.log(1 + mcs_change))
        
        total_reward = cbr_reward + beacon_reward + mcs_reward + smoothness_penalty
        
        return np.clip(total_reward, -20, 20)
    
    def update_q_table(self, state_indices, action, reward, next_state_indices):
        current_q = self.q_table[state_indices][action]
        max_next_q = np.max(self.q_table[next_state_indices])
        
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q - current_q)
        self.q_table[state_indices][action] = new_q
        
        return new_q - current_q

class PHYAgent:
    """PHY Agent with revised density awareness"""
    
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon
        self.q_table = phy_q_table
        self.state_visit_counts = defaultdict(int)
        
    def get_state_indices(self, cbr, sinr, power, neighbors):
        cbr_idx = discretize(cbr, CBR_BINS)
        sinr_idx = discretize(sinr, SINR_BINS)
        power_idx = discretize(power, POWER_BINS)
        neighbors_idx = discretize(neighbors, NEIGHBORS_BINS)
        
        return (cbr_idx, sinr_idx, power_idx, neighbors_idx)
    
    def select_action(self, state_indices, neighbor_count, current_sinr, antenna_type="OMNIDIRECTIONAL"):
        """REVISED: PHY action selection with new density categories"""
        self.state_visit_counts[state_indices] += 1
        
        density_category = get_neighbor_category(neighbor_count, antenna_type)
        expected_min, expected_max = get_expected_sinr_range(neighbor_count, antenna_type)
        
        adaptive_epsilon = self.epsilon
        
        # Adjust exploration based on SINR relative to expected range
        if current_sinr < expected_min:
            adaptive_epsilon = min(1.0, self.epsilon * 1.4)  # More exploration if below expected
        elif current_sinr > expected_max:
            adaptive_epsilon = max(0.1, self.epsilon * 0.8)  # Less exploration if above expected
        
        if random.random() < adaptive_epsilon:
            # Random jump exploration
            if self.epsilon > 0.5 and random.random() < 0.2:
                target_power = random.randint(1, 30)
                current_power = state_indices[2] + 1
                power_delta = target_power - current_power
                best_action = 0
                best_diff = float('inf')
                for i, p in enumerate(PHY_ACTIONS):
                    if abs(p - power_delta) < best_diff:
                        best_diff = abs(p - power_delta)
                        best_action = i
                return best_action
            
            # Density and SINR-aware biased exploration
            if current_sinr < expected_min:
                # Need more power
                preferred_actions = [i for i, p in enumerate(PHY_ACTIONS) if p >= 1]
                if preferred_actions and random.random() < 0.8:
                    return random.choice(preferred_actions)
            elif current_sinr > expected_max and density_category in ["HIGH", "VERY_HIGH"]:
                # Can reduce power in high density
                preferred_actions = [i for i, p in enumerate(PHY_ACTIONS) if p <= -1]
                if preferred_actions and random.random() < 0.8:
                    return random.choice(preferred_actions)
            
            return random.randint(0, PHY_ACTION_DIM - 1)
        else:
            return np.argmax(self.q_table[state_indices])
    
    def calculate_reward(self, cbr, sinr, power, neighbors, next_sinr, next_power, antenna_type="OMNIDIRECTIONAL"):
        """
        REPLACE THE calculate_reward METHOD IN PHYAgent CLASS WITH THIS
        
        Uses improved SINR reward function and better power efficiency formulations
        """
        
        # Primary: Use improved SINR reward function
        power_norm = (power - 1) / (30 - 1)  # Assuming POWER_MIN=1, POWER_MAX=30
        sinr_reward = calculate_sinr_reward(sinr, power_norm, neighbors, antenna_type)
        
        # IMPROVED: Power efficiency with logarithmic neighbor consideration
        # Optimal power depends logarithmically on neighbor density
        base_power_need = 0.3 + (0.4 * math.log(1 + neighbors) / math.log(1 + 40))  # 0.3 to 0.7
        
        if antenna_type.upper() == "SECTORAL":
            base_power_need *= 0.8  # Sectoral needs 20% less power
        
        # IMPROVED: Quadratic power efficiency reward/penalty
        power_deviation = power_norm - base_power_need
        
        if abs(power_deviation) <= 0.1:  # Within optimal range
            power_efficiency_reward = 3.0
        else:
            # Quadratic penalty for deviation from optimal
            power_efficiency_reward = -4.0 * (power_deviation ** 2)
        
        # IMPROVED: Logarithmic neighbor impact penalty
        if power_norm > 0.7:
            neighbor_impact_penalty = -2.0 * math.log(1 + neighbors) / math.log(1 + 30) * (power_norm - 0.7) ** 2
            power_efficiency_reward += neighbor_impact_penalty
        
        # IMPROVED: Logarithmic smoothness penalty
        power_change = abs(next_power - power)
        smoothness_penalty = -0.5 * math.log(1 + power_change)
        
        total_reward = sinr_reward + power_efficiency_reward + smoothness_penalty
        
        return np.clip(total_reward, -20, 20)
    
    def update_q_table(self, state_indices, action, reward, next_state_indices):
        current_q = self.q_table[state_indices][action]
        max_next_q = np.max(self.q_table[next_state_indices])
        
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q - current_q)
        self.q_table[state_indices][action] = new_q
        
        return new_q - current_q

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
        batch = random.sample(self.experience_buffer, self.batch_size)
        
        mac_td_errors = []
        phy_td_errors = []
        
        for exp in batch:
            mac_td = self.mac_agent.update_q_table(
                exp['mac_state'], exp['mac_action'], 
                exp['mac_reward'], exp['next_mac_state']
            )
            mac_td_errors.append(abs(mac_td))
            
            phy_td = self.phy_agent.update_q_table(
                exp['phy_state'], exp['phy_action'],
                exp['phy_reward'], exp['next_phy_state']
            )
            phy_td_errors.append(abs(phy_td))
        
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
            'sinr_above_12_rate': sum(1 for sinr in self.sinr_values if sinr >= 12) / len(self.sinr_values),
            
            # Density analysis with revised categories
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
                
                # Revised density analysis
                analysis_data = {
                    'Metric': ['MAC Avg Reward', 'PHY Avg Reward', 'Joint Avg Reward',
                               'CBR Performance', 'SINR Performance', 'Low Density Rate',
                               'Medium Density Rate', 'High Density Rate', 'Very High Density Rate'],
                    'Value': [
                        np.mean([d['avg_mac_reward'] for d in self.episode_data[-10:]]),
                        np.mean([d['avg_phy_reward'] for d in self.episode_data[-10:]]),
                        np.mean([d['avg_joint_reward'] for d in self.episode_data[-10:]]),
                        np.mean([d['cbr_in_range_rate'] for d in self.episode_data[-10:]]),
                        np.mean([d['sinr_above_12_rate'] for d in self.episode_data[-10:]]),
                        np.mean([d['low_density_rate'] for d in self.episode_data[-10:]]),
                        np.mean([d['medium_density_rate'] for d in self.episode_data[-10:]]),
                        np.mean([d['high_density_rate'] for d in self.episode_data[-10:]]),
                        np.mean([d['very_high_density_rate'] for d in self.episode_data[-10:]])
                    ]
                }
                analysis_df = pd.DataFrame(analysis_data)
                analysis_df.to_excel(writer, sheet_name='Density_Analysis', index=False)
                
            logger.info(f"Performance data saved to {PERFORMANCE_LOG_PATH}")
            
        except Exception as e:
            logger.error(f"Error saving to Excel: {e}")

# ================== Dual-Agent Q-Learning Implementation ==================
class DualAgentQLearning:
    def __init__(self, training_mode=True):
        self.training_mode = training_mode
        self.mac_agent = MACAgent(epsilon=EPSILON if training_mode else 0.0)
        self.phy_agent = PHYAgent(epsilon=EPSILON if training_mode else 0.0)
        self.centralized_manager = CentralizedLearningManager(self.mac_agent, self.phy_agent)
        self.performance = DualAgentPerformanceMetrics()
        self.episode_count = 0
        
        self.load_models()
    
    def process_vehicle(self, veh_id, veh_info):
        """Process vehicle with revised density categorization"""
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
            
            # Get state indices
            mac_state_indices = self.mac_agent.get_state_indices(
                cbr, sinr, current_beacon, current_mcs, neighbors
            )
            
            phy_state_indices = self.phy_agent.get_state_indices(
                cbr, sinr, current_power, neighbors
            )
            
            # Select actions with revised density awareness
            mac_action_idx = self.mac_agent.select_action(mac_state_indices, neighbors, antenna_type)
            phy_action_idx = self.phy_agent.select_action(phy_state_indices, neighbors, sinr, antenna_type)
            
            # Apply actions
            beacon_delta, mcs_delta = MAC_ACTIONS[mac_action_idx]
            power_delta = PHY_ACTIONS[phy_action_idx]
            
            new_beacon = np.clip(current_beacon + beacon_delta, BEACON_MIN, BEACON_MAX)
            new_mcs = np.clip(current_mcs + mcs_delta, 0, 9)
            new_power = np.clip(current_power + power_delta, POWER_MIN, POWER_MAX)
            
            # Training updates
            if self.training_mode:
                next_cbr = cbr + random.uniform(-0.05, 0.05)
                next_cbr = np.clip(next_cbr, 0, 1)
                next_sinr = sinr + random.uniform(-2, 2)
                
                mac_reward = self.mac_agent.calculate_reward(
                    cbr, sinr, current_beacon, current_mcs, neighbors,
                    next_cbr, new_beacon, new_mcs, antenna_type
                )
                
                phy_reward = self.phy_agent.calculate_reward(
                    cbr, sinr, current_power, neighbors,
                    next_sinr, new_power, antenna_type
                )
                
                next_mac_state = self.mac_agent.get_state_indices(
                    next_cbr, next_sinr, new_beacon, new_mcs, neighbors
                )
                next_phy_state = self.phy_agent.get_state_indices(
                    next_cbr, next_sinr, new_power, neighbors
                )
                
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
                
                self.mac_agent.epsilon = max(MIN_EPSILON, self.mac_agent.epsilon * EPSILON_DECAY)
                self.phy_agent.epsilon = max(MIN_EPSILON, self.phy_agent.epsilon * EPSILON_DECAY)
            
            # Enhanced logging with revised density categories
            if veh_id.endswith('0'):
                density_cat = get_neighbor_category(neighbors, antenna_type)
                expected_sinr = get_expected_sinr_range(neighbors, antenna_type)
                logger.info(f"Vehicle {veh_id} [{antenna_type}][{density_cat}]: "
                           f"CBR={cbr:.3f}, SINR={sinr:.1f}dB, Neighbors={neighbors}")
                logger.info(f"  Expected SINR range: {expected_sinr[0]}-{expected_sinr[1]} dB")
                logger.info(f"  MAC: Beacon {current_beacon:.0f}->{new_beacon:.0f}Hz, "
                           f"MCS {current_mcs}->{new_mcs}")
                logger.info(f"  PHY: Power {current_power:.0f}->{new_power:.0f}dBm")
            
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

def main():
    if OPERATION_MODE.upper() not in ["TRAINING", "TESTING"]:
        print(f"ERROR: Invalid OPERATION_MODE '{OPERATION_MODE}'. Must be 'TRAINING' or 'TESTING'")
        sys.exit(1)
    
    if ANTENNA_TYPE.upper() not in ["SECTORAL", "OMNIDIRECTIONAL"]:
        print(f"ERROR: Invalid ANTENNA_TYPE '{ANTENNA_TYPE}'. Must be 'SECTORAL' or 'OMNIDIRECTIONAL'")
        sys.exit(1)
    
    training_mode = (OPERATION_MODE.upper() == "TRAINING")
    
    print("="*80)
    print(" DUAL-AGENT Q-LEARNING - REVISED DENSITY SYSTEM")
    print(f"Host: {HOST}:{PORT}")
    print(f"Mode: {OPERATION_MODE.upper()}")
    print(f"Antenna Type: {ANTENNA_TYPE.upper()}")
    print("="*40)
    print("REVISED DENSITY CATEGORIES (Evidence-based):")
    
    if ANTENNA_TYPE.upper() == "SECTORAL":
        print("  • LOW: ≤13 neighbors (Expected SINR: 18-35 dB)")
        print("  • MEDIUM: ≤26 neighbors (Expected SINR: 11-25 dB)")
        print("  • HIGH: ≤40 neighbors (Expected SINR: 5-17 dB)")
        print("  • VERY_HIGH: >40 neighbors (Expected SINR: -2 to 13 dB)")
    else:
        print("  • LOW: ≤10 neighbors (Expected SINR: 15-30 dB)")
        print("  • MEDIUM: ≤20 neighbors (Expected SINR: 8-20 dB)")
        print("  • HIGH: ≤30 neighbors (Expected SINR: 2-12 dB)")
        print("  • VERY_HIGH: >30 neighbors (Expected SINR: -5 to 8 dB)")
    
    print("="*40)
    print("SCIENTIFIC IMPROVEMENTS:")
    print("  ✓ Aligned with VANET research evidence")
    print("  ✓ Expected SINR ranges for each density")
    print("  ✓ Antenna-aware thresholds (sectoral +20-30%)")
    print("  ✓ Simplified 4-category system")
    print("  ✓ Performance-driven categorization")
    print("="*40)
    print(f"CBR Target: {CBR_TARGET} (optimized for latency/PDR)")
    print(f"SINR Target: {SINR_TARGET} dB (fixed)")
    print("="*40)
    print("REWARD STRUCTURE:")
    print("CBR: 10.0 * (1 - tanh(25 * |cbr - 0.4|))")
    print("SINR: Two-phase with diminishing returns after 12 dB")
    print("  - Phase 1 (SINR < 12): Linear growth")
    print("  - Phase 2 (SINR ≥ 12): Diminishing + power efficiency")
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
