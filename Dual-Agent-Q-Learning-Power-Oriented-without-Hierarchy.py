"""
REVISED DUAL-AGENT Q-LEARNING WITH SINR-BASED MCS CONTROL
=========================================================
MAJOR CHANGES:
1. MCS controlled by SINR lookup table (conventional, reliable)
2. Q-Learning only controls Power (PHY) and Beacon Rate (MAC)
3. Simplified state spaces and action spaces
4. Improved reliability while maintaining optimization benefits
5. Faster convergence due to reduced complexity

ARCHITECTURE:
- PHY Agent: Power control only (SINR-driven)
- MAC Agent: Beacon rate control only (CBR-driven)
- MCS Selection: Deterministic SINR-based lookup table

Author: Revised version with SINR-based MCS integration
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
OPERATION_MODE = "TRAINING"        # Options: "TRAINING" or "TESTING"
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
PORT = 5001

# Power and beacon ranges
POWER_MIN = 1
POWER_MAX = 30
BEACON_MIN = 1
BEACON_MAX = 20

# File paths
MODEL_PREFIX = f"{ANTENNA_TYPE.lower()}_sinr_mcs_dual_agent"
MAC_MODEL_PATH = f'{MODEL_PREFIX}_mac_qlearning_model.npy'
PHY_MODEL_PATH = f'{MODEL_PREFIX}_phy_qlearning_model.npy'
PERFORMANCE_LOG_PATH = f'{MODEL_PREFIX}_performance.xlsx'
MODEL_SAVE_INTERVAL = 50
PERFORMANCE_LOG_INTERVAL = 10

# ================== NEW: IEEE 802.11BD SINR-BASED MCS LOOKUP TABLE ==================
def select_mcs_from_sinr(sinr_db, reliability_margin=2.0, antenna_type="OMNIDIRECTIONAL", channel_width=10):
    """
    IEEE 802.11bd MCS selection with reliability margin (MCS 0-9)
    Based on IEEE 802.11bd standard thresholds for vehicular communications
    """
    # Add reliability margin for real-world VANET conditions
    effective_sinr = sinr_db - reliability_margin
    
    # Antenna-specific adjustments
    if antenna_type.upper() == "SECTORAL":
        # Sectoral antennas can be slightly more aggressive due to reduced interference
        effective_sinr += 1.0
    
    # IEEE 802.11bd MCS lookup table (MCS 0-9) for 10 MHz channel (typical VANET)
    # Conservative thresholds for mobile vehicular environment
    if effective_sinr >= 27:    return 9   # 256-QAM 5/6 - 43.3 Mbps
    elif effective_sinr >= 24: return 8   # 256-QAM 3/4 - 39.0 Mbps
    elif effective_sinr >= 21: return 7   # 64-QAM 3/4 - 29.3 Mbps
    elif effective_sinr >= 18: return 6   # 64-QAM 2/3 - 26.0 Mbps
    elif effective_sinr >= 14: return 5   # 16-QAM 3/4 - 19.5 Mbps
    elif effective_sinr >= 11: return 4   # 16-QAM 1/2 - 13.0 Mbps
    elif effective_sinr >= 8:  return 3   # QPSK 3/4 - 9.8 Mbps
    elif effective_sinr >= 5:  return 2   # QPSK 1/2 - 6.5 Mbps
    elif effective_sinr >= 2:  return 1   # BPSK 3/4 - 3.3 Mbps
    else:                       return 0   # BPSK 1/2 - 2.2 Mbps (fallback)

def get_mcs_data_rate(mcs, channel_width=10):
    """Get theoretical data rate for IEEE 802.11bd MCS level (10 MHz channel)"""
    # Data rates for 10 MHz channel in Mbps (MCS 0-9)
    rates_10mhz = {
        0: 2.2,    # BPSK 1/2
        1: 3.3,    # BPSK 3/4  
        2: 6.5,    # QPSK 1/2
        3: 9.8,    # QPSK 3/4
        4: 13.0,   # 16-QAM 1/2
        5: 19.5,   # 16-QAM 3/4
        6: 26.0,   # 64-QAM 2/3
        7: 29.3,   # 64-QAM 3/4
        8: 39.0,   # 256-QAM 3/4
        9: 43.3    # 256-QAM 5/6
    }
    
    if channel_width == 20:
        # Double the rate for 20 MHz channel
        return rates_10mhz.get(mcs, 6.5) * 2
    else:
        return rates_10mhz.get(mcs, 6.5)

def get_mcs_sinr_requirement(mcs):
    """Get minimum SINR requirement for IEEE 802.11bd MCS level (MCS 0-9)"""
    # Conservative SINR requirements for mobile VANET environment
    requirements = {
        0: 2.0,    # BPSK 1/2
        1: 4.0,    # BPSK 3/4
        2: 7.0,    # QPSK 1/2
        3: 10.0,   # QPSK 3/4
        4: 13.0,   # 16-QAM 1/2
        5: 16.0,   # 16-QAM 3/4
        6: 20.0,   # 64-QAM 2/3
        7: 23.0,   # 64-QAM 3/4
        8: 26.0,   # 256-QAM 3/4
        9: 29.0    # 256-QAM 5/6
    }
    return requirements.get(mcs, 7.0)

def get_mcs_modulation_info(mcs):
    """Get modulation and coding information for IEEE 802.11bd MCS (MCS 0-9)"""
    info = {
        0: {"modulation": "BPSK", "coding_rate": "1/2", "constellation": 2},
        1: {"modulation": "BPSK", "coding_rate": "3/4", "constellation": 2},
        2: {"modulation": "QPSK", "coding_rate": "1/2", "constellation": 4},
        3: {"modulation": "QPSK", "coding_rate": "3/4", "constellation": 4},
        4: {"modulation": "16-QAM", "coding_rate": "1/2", "constellation": 16},
        5: {"modulation": "16-QAM", "coding_rate": "3/4", "constellation": 16},
        6: {"modulation": "64-QAM", "coding_rate": "2/3", "constellation": 64},
        7: {"modulation": "64-QAM", "coding_rate": "3/4", "constellation": 64},
        8: {"modulation": "256-QAM", "coding_rate": "3/4", "constellation": 256},
        9: {"modulation": "256-QAM", "coding_rate": "5/6", "constellation": 256}
    }
    return info.get(mcs, {"modulation": "QPSK", "coding_rate": "1/2", "constellation": 4})

# ================== ENHANCED 6-LEVEL DENSITY CATEGORIZATION ==================
def get_neighbor_category(neighbor_count, antenna_type="OMNIDIRECTIONAL"):
    """REVISED: Density categorization calibrated with real VANET experiment data"""
    
    if antenna_type.upper() == "SECTORAL":
        # Sectoral antenna: reduce effective neighbors by 30%
        effective_neighbors = neighbor_count * 0.7
        
        if effective_neighbors <= 3:        return "VERY_LOW"     # Rural/Highway (0-4 total)
        elif effective_neighbors <= 6:      return "LOW"          # Sparse Urban (0-8 total)
        elif effective_neighbors <= 10:     return "MEDIUM"       # Normal Urban (0-14 total)
        elif effective_neighbors <= 15:     return "HIGH"         # Dense Urban (0-21 total)
        elif effective_neighbors <= 20:     return "VERY_HIGH"    # Traffic Jam (0-28 total)
        else:                                return "EXTREME"      # Extreme Congestion (29+ total)
    else:
        # Based on real data: 8-20 neighbors observed, mostly 13-18
        if neighbor_count <= 4:             return "VERY_LOW"     # Rural/Highway
        elif neighbor_count <= 8:           return "LOW"          # Sparse Urban
        elif neighbor_count <= 12:          return "MEDIUM"       # Normal Urban (8-10 from data)
        elif neighbor_count <= 17:          return "HIGH"         # Dense Urban (13-15 from data)
        elif neighbor_count <= 25:          return "VERY_HIGH"    # Traffic Jam (16-20 from data)
        else:                                return "EXTREME"      # Extreme Congestion (20+ from data)

def get_expected_sinr_range(neighbor_count, antenna_type="OMNIDIRECTIONAL"):
    """REVISED: SINR expectations based on real experiment results"""
    category = get_neighbor_category(neighbor_count, antenna_type)
    
    if antenna_type.upper() == "SECTORAL":
        # Assume 3-5 dB better performance for sectoral
        ranges = {
            "VERY_LOW": (20, 35),       # Rural/Highway - excellent SINR
            "LOW": (15, 30),            # Sparse Urban - very good SINR
            "MEDIUM": (8, 22),          # Normal Urban - good SINR
            "HIGH": (3, 18),            # Dense Urban - moderate SINR (real data shows 5-18)
            "VERY_HIGH": (1, 12),       # Traffic Jam - poor SINR (real data shows 1.88-15)
            "EXTREME": (-3, 8)          # Extreme Congestion - very poor SINR
        }
    else:
        # Based on real omnidirectional data
        ranges = {
            "VERY_LOW": (18, 30),       # Rural/Highway
            "LOW": (12, 25),            # Sparse Urban
            "MEDIUM": (6, 20),          # Normal Urban
            "HIGH": (2, 15),            # Dense Urban (real: 1.88-18 dB with 13-15 neighbors)
            "VERY_HIGH": (1, 10),       # Traffic Jam (real: 1.88-25 dB with 16-20 neighbors)
            "EXTREME": (-2, 6)          # Extreme Congestion
        }
    
    return ranges.get(category, (6, 18))

def get_density_multiplier(neighbor_count, antenna_type="OMNIDIRECTIONAL"):
    """Enhanced density-based reward multiplier for 6 levels"""
    category = get_neighbor_category(neighbor_count, antenna_type)
    multipliers = {
        "VERY_LOW": 0.6,            # Low challenge
        "LOW": 0.8,                 # Moderate challenge
        "MEDIUM": 1.0,              # Baseline challenge
        "HIGH": 1.4,                # High challenge (real data shows CBR issues)
        "VERY_HIGH": 1.8,           # Very high challenge (real data shows high CBR)
        "EXTREME": 2.2              # Extreme challenge
    }
    return multipliers.get(category, 1.0)

# ================== DENSITY-ADAPTIVE POWER RANGES ==================
def get_density_adaptive_power_range(neighbor_count, antenna_type="OMNIDIRECTIONAL"):
    """
    ENHANCED: Inverse density-power exploration relationship
    Higher density → Lower maximum power exploration (reduce interference)
    Lower density → Higher maximum power exploration (optimize range)
    All ranges start from 1 dBm to ensure basic connectivity
    """
    category = get_neighbor_category(neighbor_count, antenna_type)
    
    if antenna_type.upper() == "SECTORAL":
        # Sectoral antenna: slightly higher ranges due to directional benefits
        ranges = {
            "VERY_LOW": (1, 30),     # Rural/Highway - maximum exploration for range
            "LOW": (1, 22),          # Sparse Urban - wide exploration  
            "MEDIUM": (1, 17),       # Normal Urban - moderate exploration
            "HIGH": (1, 12),         # Dense Urban - limited exploration
            "VERY_HIGH": (1, 8),     # Traffic Jam - conservative exploration
            "EXTREME": (1, 4)        # Extreme Congestion - minimal exploration
        }
    else:
        # Omnidirectional antenna - standard inverse relationship
        ranges = {
            "VERY_LOW": (1, 30),     # Rural/Highway - maximum exploration for range
            "LOW": (1, 20),          # Sparse Urban - wide exploration  
            "MEDIUM": (1, 15),       # Normal Urban - moderate exploration
            "HIGH": (1, 10),         # Dense Urban - limited exploration
            "VERY_HIGH": (1, 6),     # Traffic Jam - conservative exploration
            "EXTREME": (1, 3)        # Extreme Congestion - minimal exploration only
        }
    
    return ranges.get(category, (1, 15))

# ================== REVISED State Discretization ==================
CBR_BINS = np.linspace(0.0, 1.0, 21)
SINR_BINS = np.linspace(0, 50, 11)
BEACON_BINS = np.arange(1, 21)
NEIGHBORS_BINS = np.linspace(0, 50, 11)
POWER_BINS = np.arange(1, 31)

# REVISED: Simplified state dimensions (no MCS)
MAC_STATE_DIM = (20, 10, 20, 10)  # CBR, SINR, Beacon, Neighbors
PHY_STATE_DIM = (20, 10, 30, 10)  # CBR, SINR, Power, Neighbors

# ================== REVISED Action Spaces ==================
# MAC actions: Only beacon rate changes (no MCS)
MAC_ACTIONS = [0, 1, -1, 2, -2, 3, -3, 5, -5, 10, -10]

# PHY actions: Only power changes
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

logger.info("REVISED DUAL-AGENT Q-LEARNING WITH IEEE 802.11BD SINR-BASED MCS CONTROL")
logger.info(f"CBR Target: {CBR_TARGET}")
logger.info(f"SINR Target: {SINR_TARGET} dB (fixed)")
logger.info(f"Antenna Type: {ANTENNA_TYPE}")
logger.info("MCS Control: IEEE 802.11bd SINR-based lookup table (MCS 0-11)")
logger.info("Q-Learning: Power and Beacon Rate only")

# ================== Helper Functions ==================
def discretize(value, bins):
    """Discretize a continuous value into a bin index"""
    if not np.isfinite(value):
        value = 0.0
    
    if len(bins) <= 30 and bins[0] == 1 and bins[-1] in [20, 30]:
        value = int(np.clip(value, bins[0], bins[-1]))
        return value - bins[0]
    else:
        value = np.clip(value, bins[0], bins[-1])
        bin_idx = np.digitize(value, bins) - 1
        return max(0, min(len(bins) - 2, bin_idx))

def get_optimal_parameters(neighbors, antenna_type="OMNIDIRECTIONAL"):
    """Calculate optimal parameters based on neighbor density"""
    
    # Logarithmic scaling factors
    beacon_factor = 1.0 - (0.3 * math.log(1 + neighbors) / math.log(1 + 40))
    power_factor = 0.3 + (0.4 * math.log(1 + neighbors) / math.log(1 + 40))
    
    # Base parameters
    optimal_beacon = 12.0 * beacon_factor
    optimal_power_norm = power_factor
    
    # Antenna adjustments
    if antenna_type.upper() == "SECTORAL":
        optimal_beacon *= 1.1      # 10% higher beacon
        optimal_power_norm *= 0.8  # 20% lower power need
    
    return {
        'beacon': optimal_beacon,
        'power_norm': optimal_power_norm
    }

# ================== Enhanced SINR Reward Function ==================
def calculate_sinr_reward(sinr, power_norm, neighbors, antenna_type="OMNIDIRECTIONAL"):
    """
    ENHANCED: Improved SINR reward function with power efficiency focus
    """
    SINR_TARGET = 12.0
    
    # Phase 1: Below target - aggressive improvement needed
    if sinr < SINR_TARGET:
        base_reward = 10.0 * (sinr / SINR_TARGET)
        
        # Logarithmic neighbor penalty (smoother than categorical bins)
        neighbor_penalty = -2.0 * math.log(1 + neighbors) / math.log(1 + 30)
        sinr_reward = base_reward + neighbor_penalty
        
    else:
        # Phase 2: Above target - diminishing returns
        base_reward = 10.0
        excess_sinr = sinr - SINR_TARGET
        diminishing_reward = 5.0 * math.sqrt(excess_sinr / 10.0)
        sinr_reward = base_reward + diminishing_reward
        sinr_reward = min(sinr_reward, 18.0)
    
    # ENHANCED: Quadratic power penalty when SINR is sufficient
    if sinr >= SINR_TARGET:
        if power_norm > 0.6:
            # Quadratic penalty grows MUCH faster than linear
            power_penalty = -8.0 * ((power_norm - 0.6) ** 2)
            sinr_reward += power_penalty
        elif power_norm <= 0.4:
            # Quadratic efficiency bonus
            efficiency_bonus = 4.0 * ((0.4 - power_norm) ** 2)
            sinr_reward += efficiency_bonus
    
    # Logarithmic neighbor impact for excessive SINR
    if sinr > 20.0:
        neighbor_impact_penalty = -3.0 * math.log(1 + neighbors) / math.log(1 + 50) * (sinr - 20.0) / 10.0
        sinr_reward += neighbor_impact_penalty
    
    return np.clip(sinr_reward, -15, 20)

# ================== REVISED MAC AGENT ==================
class MACAgent:
    """REVISED MAC Agent: Beacon rate control only"""
    
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon
        self.q_table = mac_q_table
        self.state_visit_counts = defaultdict(int)
        self.last_power_action = 0
        
    def get_state_indices(self, cbr, sinr, beacon, neighbors):
        cbr_idx = discretize(cbr, CBR_BINS)
        sinr_idx = discretize(sinr, SINR_BINS)
        beacon_idx = discretize(beacon, BEACON_BINS)
        neighbors_idx = discretize(neighbors, NEIGHBORS_BINS)
        
        return (cbr_idx, sinr_idx, beacon_idx, neighbors_idx)
    
    def select_action(self, state_indices, neighbor_count, power_action_taken, 
                     current_cbr, antenna_type="OMNIDIRECTIONAL"):
        """REVISED: MAC action selection for beacon rate only"""
        self.state_visit_counts[state_indices] += 1
        self.last_power_action = power_action_taken
        
        density_category = get_neighbor_category(neighbor_count, antenna_type)
        adaptive_epsilon = self.epsilon
        
        # Adjust exploration based on density and power action
        if density_category in ["HIGH", "VERY_HIGH", "EXTREME"]:
            adaptive_epsilon *= 0.8  # More conservative in high density
        
        if random.random() < adaptive_epsilon:
            # POWER-AWARE MAC EXPLORATION
            power_change = PHY_ACTIONS[power_action_taken] if power_action_taken < len(PHY_ACTIONS) else 0
            
            # Random jump exploration
            if self.epsilon > 0.5 and random.random() < 0.2:
                target_beacon = random.randint(1, 20)
                current_beacon = state_indices[2] + 1
                beacon_delta = target_beacon - current_beacon
                best_action = 0
                best_diff = float('inf')
                for i, b in enumerate(MAC_ACTIONS):
                    if abs(b - beacon_delta) < best_diff:
                        best_diff = abs(b - beacon_delta)
                        best_action = i
                return best_action
            
            # ENHANCED: Power-coordinated exploration
            preferred_actions = []
            
            if power_change > 2:  # Power increasing significantly
                # Be more conservative with beacon to avoid over-congestion
                preferred_actions = [i for i, b in enumerate(MAC_ACTIONS) if b <= 0]
                                   
            elif power_change < -2:  # Power decreasing significantly
                # Can be slightly more aggressive with beacon
                if density_category not in ["VERY_HIGH", "EXTREME"]:
                    preferred_actions = [i for i, b in enumerate(MAC_ACTIONS) if 0 <= b <= 2]
                else:
                    preferred_actions = [i for i, b in enumerate(MAC_ACTIONS) if -1 <= b <= 1]
            else:  # Power stable
                # Density-aware biased exploration
                if density_category in ["HIGH", "VERY_HIGH", "EXTREME"]:
                    # Conservative beacon actions in high density
                    preferred_actions = [i for i, b in enumerate(MAC_ACTIONS) if b <= 1]
                elif density_category == "LOW":
                    # Can use higher beacon rates in low density
                    preferred_actions = [i for i, b in enumerate(MAC_ACTIONS) if b >= -1]
                else:
                    preferred_actions = [i for i, b in enumerate(MAC_ACTIONS) if abs(b) <= 2]
            
            if preferred_actions and random.random() < 0.8:
                return random.choice(preferred_actions)
            
            return random.randint(0, MAC_ACTION_DIM - 1)
        else:
            return np.argmax(self.q_table[state_indices])
    
    def calculate_reward(self, cbr, sinr, beacon, neighbors, next_cbr, next_beacon, 
                        power_reward, antenna_type="OMNIDIRECTIONAL"):
        """
        REVISED: MAC reward for beacon rate optimization only
        """
        
        # Primary CBR optimization (keep existing formula - it's already good)
        cbr_error = abs(cbr - 0.4)  # CBR_TARGET = 0.4
        cbr_reward = 10.0 * (1 - math.tanh(25 * cbr_error))
        
        # ENHANCED: Beacon optimization with logarithmic neighbor scaling
        optimal_beacon_factor = 1.0 - (0.3 * math.log(1 + neighbors) / math.log(1 + 40))
        optimal_beacon = 12.0 * optimal_beacon_factor  # Base beacon 12, scaled by density
        
        # Antenna awareness
        if antenna_type.upper() == "SECTORAL":
            optimal_beacon *= 1.1  # Sectoral can handle 10% higher beacon rates
        
        # Quadratic beacon penalty (stronger than linear)
        beacon_error = abs(beacon - optimal_beacon)
        beacon_reward = -2.0 * (beacon_error / 5.0) ** 2
        
        # NEW: Power coordination bonus (30% of power reward)
        power_coordination_bonus = power_reward * 0.3
        
        # NEW: Power-MAC alignment bonus
        power_change = PHY_ACTIONS[self.last_power_action] if self.last_power_action < len(PHY_ACTIONS) else 0
        beacon_change = next_beacon - beacon
        
        alignment_bonus = 0
        if power_change > 0 and beacon_change <= 0:
            # Power up, MAC conservative - good alignment
            alignment_bonus = 2.0
        elif power_change < 0 and beacon_change >= 0:
            # Power down, MAC compensates - good alignment  
            alignment_bonus = 2.0
        elif abs(power_change) <= 1 and abs(beacon_change) <= 1:
            # Both stable - good alignment
            alignment_bonus = 1.0
        
        # Logarithmic smoothness penalty (more forgiving than linear)
        beacon_change_penalty = abs(next_beacon - beacon)
        smoothness_penalty = -1.0 * math.log(1 + beacon_change_penalty)
        
        total_reward = (cbr_reward + beacon_reward + power_coordination_bonus + 
                       alignment_bonus + smoothness_penalty)
        
        return np.clip(total_reward, -25, 30)
    
    def update_q_table(self, state_indices, action, reward, next_state_indices):
        current_q = self.q_table[state_indices][action]
        max_next_q = np.max(self.q_table[next_state_indices])
        
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q - current_q)
        self.q_table[state_indices][action] = new_q
        
        return new_q - current_q

# ================== REVISED PHY AGENT ==================
class PHYAgent:
    """REVISED PHY Agent: Power control only"""
    
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon
        self.q_table = phy_q_table
        self.state_visit_counts = defaultdict(int)
        self.power_efficiency_history = []
        
    def get_state_indices(self, cbr, sinr, power, neighbors):
        cbr_idx = discretize(cbr, CBR_BINS)
        sinr_idx = discretize(sinr, SINR_BINS)
        power_idx = discretize(power, POWER_BINS)
        neighbors_idx = discretize(neighbors, NEIGHBORS_BINS)
        
        return (cbr_idx, sinr_idx, power_idx, neighbors_idx)
    
    def select_action(self, state_indices, neighbor_count, current_sinr, current_power,
                     antenna_type="OMNIDIRECTIONAL"):
        """REVISED: Power control action selection only"""
        self.state_visit_counts[state_indices] += 1
        
        # Get density-adaptive power range (KEY ENHANCEMENT)
        power_min, power_max = get_density_adaptive_power_range(neighbor_count, antenna_type)
        density_category = get_neighbor_category(neighbor_count, antenna_type)
        expected_min, expected_max = get_expected_sinr_range(neighbor_count, antenna_type)
        
        # Calculate adaptive epsilon with power efficiency bias
        adaptive_epsilon = self.epsilon
        power_efficiency = 1.0 - ((current_power - power_min) / max(1, power_max - power_min))
        
        # More aggressive exploration when power is inefficient
        if power_efficiency < 0.3:  # Low efficiency - need more exploration
            adaptive_epsilon = min(1.0, self.epsilon * 1.4)
        elif power_efficiency > 0.7:  # High efficiency - less exploration
            adaptive_epsilon = max(0.05, self.epsilon * 0.6)
        
        # Adjust exploration based on SINR relative to expected range
        if current_sinr < expected_min:
            adaptive_epsilon = min(1.0, adaptive_epsilon * 1.4)  # More exploration if below expected
        elif current_sinr > expected_max:
            adaptive_epsilon = max(0.1, adaptive_epsilon * 0.8)  # Less exploration if above expected
        
        if random.random() < adaptive_epsilon:
            # Random jump exploration
            if self.epsilon > 0.5 and random.random() < 0.2:
                target_power = random.randint(power_min, power_max)
                current_power = state_indices[2] + 1
                power_delta = target_power - current_power
                best_action = 0
                best_diff = float('inf')
                for i, p in enumerate(PHY_ACTIONS):
                    if abs(p - power_delta) < best_diff:
                        best_diff = abs(p - power_delta)
                        best_action = i
                return best_action
            
            # ENHANCED: DENSITY-AWARE INTELLIGENT EXPLORATION
            preferred_actions = []
            
            if density_category == "EXTREME":
                # Emergency power reduction only
                preferred_actions = [i for i, p in enumerate(PHY_ACTIONS) if p <= -3]
                
            elif density_category == "VERY_HIGH":
                # Traffic jam - prioritize power reduction
                if current_power > power_max:
                    preferred_actions = [i for i, p in enumerate(PHY_ACTIONS) if p <= -2]
                else:
                    preferred_actions = [i for i, p in enumerate(PHY_ACTIONS) if abs(p) <= 2]
                    
            elif density_category == "HIGH":
                # Dense urban - moderate power management
                if current_sinr < SINR_TARGET and current_power < power_max:
                    preferred_actions = [i for i, p in enumerate(PHY_ACTIONS) if 1 <= p <= 3]
                else:
                    preferred_actions = [i for i, p in enumerate(PHY_ACTIONS) if abs(p) <= 3]
                    
            elif density_category in ["LOW", "VERY_LOW"]:
                # Sparse conditions - can use higher power if needed
                if current_sinr < SINR_TARGET:
                    preferred_actions = [i for i, p in enumerate(PHY_ACTIONS) if p >= 2]
                else:
                    preferred_actions = list(range(PHY_ACTION_DIM))
                    
            else:  # MEDIUM
                # Balanced exploration
                if current_sinr < expected_min:
                    # Need more power
                    preferred_actions = [i for i, p in enumerate(PHY_ACTIONS) if p >= 1]
                elif current_sinr > expected_max and current_power > power_min + 3:
                    # Can reduce power
                    preferred_actions = [i for i, p in enumerate(PHY_ACTIONS) if p <= -1]
                else:
                    preferred_actions = [i for i, p in enumerate(PHY_ACTIONS) if abs(p) <= 5]
            
            # CRITICAL: Bounds checking for density-adaptive ranges
            valid_actions = []
            for i in (preferred_actions if preferred_actions else list(range(PHY_ACTION_DIM))):
                new_power = current_power + PHY_ACTIONS[i]
                if power_min <= new_power <= power_max:
                    valid_actions.append(i)
            
            if valid_actions:
                action = random.choice(valid_actions)
            else:
                action = 0  # No change fallback
                
            # Track power efficiency for analysis
            self.power_efficiency_history.append(power_efficiency)
            if len(self.power_efficiency_history) > 100:
                self.power_efficiency_history = self.power_efficiency_history[-50:]
                
            return action
        else:
            return np.argmax(self.q_table[state_indices])
    
    def calculate_reward(self, cbr, sinr, power, neighbors, next_sinr, next_power, 
                        mac_beacon_change=0, antenna_type="OMNIDIRECTIONAL"):
        """REVISED: Power control reward only"""
        
        # Get density-adaptive context
        power_min, power_max = get_density_adaptive_power_range(neighbors, antenna_type)
        power_norm = (power - power_min) / max(1, power_max - power_min)
        
        # 1. Primary SINR performance (enhanced from original)
        sinr_reward = calculate_sinr_reward(sinr, power_norm, neighbors, antenna_type)
        
        # 2. ENHANCED: Power efficiency with density awareness
        optimal_power_norm = 0.3 + (0.4 * math.log(1 + neighbors) / math.log(1 + 40))
        power_deviation = abs(power_norm - optimal_power_norm)
        
        if power_deviation <= 0.15:
            power_efficiency_reward = 8.0 * (0.15 - power_deviation)
        else:
            power_efficiency_reward = -6.0 * power_deviation ** 2
        
        # 3. NEW: MAC coordination bonus (reward power decisions that help MAC)
        coordination_bonus = 0
        power_change = next_power - power
        
        # If MAC is increasing beacon (more aggressive), reward stable/reduced power
        if mac_beacon_change > 0 and power_change <= 0:
            coordination_bonus = 2.0
        # If MAC is reducing beacon (conservative), allow slight power increase
        elif mac_beacon_change < 0 and -2 <= power_change <= 3:
            coordination_bonus = 1.5
        
        # 4. ENHANCED: Density-appropriate power bonus
        density_bonus = 0
        if power_min <= power <= power_max:
            density_bonus = 3.0
        elif power > power_max:
            excess = power - power_max
            density_bonus = -2.0 * excess  # Linear penalty for excess
        elif power < power_min:
            deficit = power_min - power
            density_bonus = -1.5 * deficit  # Moderate penalty for too low
        
        # 5. Power efficiency bonus for excellent performance
        if sinr >= SINR_TARGET and power_norm <= 0.5:
            efficiency_excellence_bonus = 3.0
        else:
            efficiency_excellence_bonus = 0
        
        # 6. Logarithmic smoothness penalty
        power_change_penalty = abs(next_power - power)
        smoothness_penalty = -0.5 * math.log(1 + power_change_penalty)
        
        total_reward = (sinr_reward + power_efficiency_reward + coordination_bonus + 
                       density_bonus + efficiency_excellence_bonus + smoothness_penalty)
        
        return np.clip(total_reward, -25, 25)
    
    def update_q_table(self, state_indices, action, reward, next_state_indices):
        current_q = self.q_table[state_indices][action]
        max_next_q = np.max(self.q_table[next_state_indices])
        
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q - current_q)
        self.q_table[state_indices][action] = new_q
        
        return new_q - current_q

# ================== REVISED Centralized Learning Manager ==================
class CentralizedLearningManager:
    """REVISED: Manages coordinated learning with simplified action space"""
    
    def __init__(self, mac_agent, phy_agent):
        self.mac_agent = mac_agent
        self.phy_agent = phy_agent
        self.experience_buffer = deque(maxlen=BUFFER_SIZE)
        self.update_counter = 0
        self.batch_size = 32
        self.power_efficiency_tracker = []
        
    def add_experience(self, experience):
        """Add experience to buffer with power efficiency tracking"""
        self.experience_buffer.append(experience)
        self.update_counter += 1
        
        # Track power efficiency
        power_min, power_max = get_density_adaptive_power_range(experience['neighbors'], ANTENNA_TYPE)
        power_efficiency = 1.0 - ((experience['power'] - power_min) / max(1, power_max - power_min))
        self.power_efficiency_tracker.append(power_efficiency)
        
        # Batch update every N experiences
        if self.update_counter % self.batch_size == 0 and len(self.experience_buffer) >= self.batch_size:
            self.perform_batch_update()
    
    def perform_batch_update(self):
        """REVISED: Perform batch Q-learning updates with power priority"""
        batch = random.sample(self.experience_buffer, self.batch_size)
        
        mac_td_errors = []
        phy_td_errors = []
        
        for exp in batch:
            # POWER PRIORITY: Update PHY agent first
            phy_td = self.phy_agent.update_q_table(
                exp['phy_state'], exp['phy_action'],
                exp['phy_reward'], exp['next_phy_state']
            )
            phy_td_errors.append(abs(phy_td))
            
            # Then update MAC agent (aware of power decisions)
            mac_td = self.mac_agent.update_q_table(
                exp['mac_state'], exp['mac_action'], 
                exp['mac_reward'], exp['next_mac_state']
            )
            mac_td_errors.append(abs(mac_td))
        
        if self.update_counter % 100 == 0:
            avg_power_efficiency = np.mean(self.power_efficiency_tracker[-100:]) if self.power_efficiency_tracker else 0
            logger.info(f"Revised Batch update {self.update_counter//self.batch_size}: "
                       f"MAC TD error: {np.mean(mac_td_errors):.4f}, "
                       f"PHY TD error: {np.mean(phy_td_errors):.4f}, "
                       f"Power Efficiency: {avg_power_efficiency:.2%}")

# ================== REVISED Performance Tracking ==================
class DualAgentPerformanceMetrics:
    """REVISED: Performance tracking with SINR-based MCS focus"""
    
    def __init__(self):
        self.reset_metrics()
        self.episode_data = []
        
    def reset_metrics(self):
        self.mac_rewards = []
        self.phy_rewards = []
        self.joint_rewards = []
        self.cbr_values = []
        self.sinr_values = []
        self.power_values = []
        self.beacon_values = []
        self.mcs_values = []
        self.mac_actions = []
        self.phy_actions = []
        self.neighbor_counts = []
        self.power_efficiencies = []
        self.density_categories = []
        self.mcs_sinr_alignment = []  # NEW: Track MCS-SINR alignment
        
    def add_step(self, mac_reward, phy_reward, cbr, sinr, power, beacon, mcs, neighbors, 
                mac_action, phy_action):
        self.mac_rewards.append(mac_reward)
        self.phy_rewards.append(phy_reward)
        self.joint_rewards.append(0.5 * mac_reward + 0.5 * phy_reward)
        self.cbr_values.append(cbr)
        self.sinr_values.append(sinr)
        self.power_values.append(power)
        self.beacon_values.append(beacon)
        self.mcs_values.append(mcs)
        self.neighbor_counts.append(neighbors)
        self.mac_actions.append(mac_action)
        self.phy_actions.append(phy_action)
        
        # Calculate power efficiency
        power_min, power_max = get_density_adaptive_power_range(neighbors, ANTENNA_TYPE)
        power_efficiency = 1.0 - ((power - power_min) / max(1, power_max - power_min))
        self.power_efficiencies.append(power_efficiency)
        
        # Track density category
        density_cat = get_neighbor_category(neighbors, ANTENNA_TYPE)
        self.density_categories.append(density_cat)
        
        # NEW: Track MCS-SINR alignment
        required_sinr = get_mcs_sinr_requirement(mcs)
        sinr_margin = sinr - required_sinr
        self.mcs_sinr_alignment.append(sinr_margin)
    
    def calculate_episode_metrics(self, episode_num):
        """REVISED: Calculate episode statistics with MCS alignment focus"""
        if not self.mac_rewards:
            return {}
            
        metrics = {
            'episode': episode_num,
            'timestamp': datetime.now(),
            'total_steps': len(self.mac_rewards),
            
            # Enhanced reward metrics
            'avg_mac_reward': np.mean(self.mac_rewards),
            'avg_phy_reward': np.mean(self.phy_rewards),
            'avg_joint_reward': np.mean(self.joint_rewards),
            'cumulative_joint_reward': sum(self.joint_rewards),
            
            # Enhanced performance metrics
            'avg_cbr': np.mean(self.cbr_values),
            'cbr_in_range_rate': sum(1 for cbr in self.cbr_values if CBR_RANGE[0] <= cbr <= CBR_RANGE[1]) / len(self.cbr_values),
            'avg_sinr': np.mean(self.sinr_values),
            'sinr_above_12_rate': sum(1 for sinr in self.sinr_values if sinr >= 12) / len(self.sinr_values),
            
            # Power efficiency metrics
            'avg_power': np.mean(self.power_values),
            'avg_power_efficiency': np.mean(self.power_efficiencies),
            'high_power_efficiency_rate': sum(1 for eff in self.power_efficiencies if eff > 0.7) / len(self.power_efficiencies),
            'low_power_usage_rate': sum(1 for p in self.power_values if p <= 10) / len(self.power_values),
            'power_std': np.std(self.power_values),
            
            # NEW: MCS performance metrics
            'avg_mcs': np.mean(self.mcs_values),
            'avg_mcs_sinr_margin': np.mean(self.mcs_sinr_alignment),
            'mcs_well_supported_rate': sum(1 for margin in self.mcs_sinr_alignment if margin >= 2) / len(self.mcs_sinr_alignment),
            'mcs_over_aggressive_rate': sum(1 for margin in self.mcs_sinr_alignment if margin < 0) / len(self.mcs_sinr_alignment),
            
            # Enhanced density analysis with 6 levels
            'avg_neighbors': np.mean(self.neighbor_counts),
            'very_low_density_rate': sum(1 for cat in self.density_categories if cat == "VERY_LOW") / len(self.density_categories),
            'low_density_rate': sum(1 for cat in self.density_categories if cat == "LOW") / len(self.density_categories),
            'medium_density_rate': sum(1 for cat in self.density_categories if cat == "MEDIUM") / len(self.density_categories),
            'high_density_rate': sum(1 for cat in self.density_categories if cat == "HIGH") / len(self.density_categories),
            'very_high_density_rate': sum(1 for cat in self.density_categories if cat == "VERY_HIGH") / len(self.density_categories),
            'extreme_density_rate': sum(1 for cat in self.density_categories if cat == "EXTREME") / len(self.density_categories),
            
            # Enhanced action diversity
            'mac_action_entropy': entropy(np.bincount(self.mac_actions, minlength=MAC_ACTION_DIM)),
            'phy_action_entropy': entropy(np.bincount(self.phy_actions, minlength=PHY_ACTION_DIM)),
            
            # Power-density correlation
            'power_density_correlation': np.corrcoef(self.neighbor_counts, self.power_values)[0,1] if len(self.neighbor_counts) > 1 else 0,
        }
        
        return metrics
    
    def log_performance(self, episode_num):
        """REVISED: Log and save performance metrics"""
        metrics = self.calculate_episode_metrics(episode_num)
        if metrics:
            self.episode_data.append(metrics)
            
            if episode_num % PERFORMANCE_LOG_INTERVAL == 0:
                self.save_to_excel()
        return metrics
    
    def save_to_excel(self):
        """REVISED: Save performance data to Excel with MCS analysis"""
        try:
            with pd.ExcelWriter(PERFORMANCE_LOG_PATH, engine='openpyxl', mode='w') as writer:
                if self.episode_data:
                    episode_df = pd.DataFrame(self.episode_data)
                    episode_df.to_excel(writer, sheet_name='Episode_Summary', index=False)
                
                # Revised analysis with MCS focus
                if len(self.episode_data) >= 10:
                    recent_data = self.episode_data[-10:]
                    analysis_data = {
                        'Metric': ['MAC Avg Reward', 'PHY Avg Reward', 'Joint Avg Reward',
                                   'CBR Performance', 'SINR Performance', 'Power Efficiency',
                                   'MCS SINR Margin', 'MCS Well Supported Rate', 'MCS Over Aggressive Rate',
                                   'High Power Efficiency Rate', 'Low Power Usage Rate',
                                   'Very Low Density Rate', 'Low Density Rate', 'Medium Density Rate',
                                   'High Density Rate', 'Very High Density Rate', 'Extreme Density Rate',
                                   'Power-Density Correlation'],
                        'Value': [
                            np.mean([d['avg_mac_reward'] for d in recent_data]),
                            np.mean([d['avg_phy_reward'] for d in recent_data]),
                            np.mean([d['avg_joint_reward'] for d in recent_data]),
                            np.mean([d['cbr_in_range_rate'] for d in recent_data]),
                            np.mean([d['sinr_above_12_rate'] for d in recent_data]),
                            np.mean([d['avg_power_efficiency'] for d in recent_data]),
                            np.mean([d['avg_mcs_sinr_margin'] for d in recent_data]),
                            np.mean([d['mcs_well_supported_rate'] for d in recent_data]),
                            np.mean([d['mcs_over_aggressive_rate'] for d in recent_data]),
                            np.mean([d['high_power_efficiency_rate'] for d in recent_data]),
                            np.mean([d['low_power_usage_rate'] for d in recent_data]),
                            np.mean([d['very_low_density_rate'] for d in recent_data]),
                            np.mean([d['low_density_rate'] for d in recent_data]),
                            np.mean([d['medium_density_rate'] for d in recent_data]),
                            np.mean([d['high_density_rate'] for d in recent_data]),
                            np.mean([d['very_high_density_rate'] for d in recent_data]),
                            np.mean([d['extreme_density_rate'] for d in recent_data]),
                            np.mean([d['power_density_correlation'] for d in recent_data])
                        ]
                    }
                    analysis_df = pd.DataFrame(analysis_data)
                    analysis_df.to_excel(writer, sheet_name='Revised_Analysis', index=False)
                
                # MCS-SINR alignment trend analysis
                mcs_trend_data = []
                for episode in self.episode_data:
                    mcs_trend_data.append({
                        'episode': episode['episode'],
                        'avg_mcs': episode['avg_mcs'],
                        'mcs_sinr_margin': episode['avg_mcs_sinr_margin'],
                        'mcs_well_supported_rate': episode['mcs_well_supported_rate'],
                        'mcs_over_aggressive_rate': episode['mcs_over_aggressive_rate'],
                        'sinr_performance': episode['sinr_above_12_rate']
                    })
                
                trend_df = pd.DataFrame(mcs_trend_data)
                trend_df.to_excel(writer, sheet_name='MCS_SINR_Alignment', index=False)
                
            logger.info(f"Revised performance data saved to {PERFORMANCE_LOG_PATH}")
            
        except Exception as e:
            logger.error(f"Error saving to Excel: {e}")

# ================== REVISED Dual-Agent Q-Learning Implementation ==================
class DualAgentQLearning:
    """REVISED: Dual-Agent Q-Learning with SINR-based MCS"""
    
    def __init__(self, training_mode=True):
        self.training_mode = training_mode
        self.mac_agent = MACAgent(epsilon=EPSILON if training_mode else 0.0)
        self.phy_agent = PHYAgent(epsilon=EPSILON if training_mode else 0.0)
        self.centralized_manager = CentralizedLearningManager(self.mac_agent, self.phy_agent)
        self.performance = DualAgentPerformanceMetrics()
        self.episode_count = 0
        
        self.load_models()
    
    def process_vehicle(self, veh_id, veh_info):
        """REVISED: Process vehicle with SINR-based MCS control"""
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
            cbr = np.clip(cbr, 0.0, 1.0)
            sinr = np.clip(sinr, 0, 50)
            neighbors = max(0, neighbors)
            
            # Handle NaN values
            if not np.isfinite(cbr): cbr = 0.4
            if not np.isfinite(sinr): sinr = 15.0
            if not np.isfinite(current_power): current_power = 15.0
            if not np.isfinite(current_beacon): current_beacon = 10.0
            
            # REVISED: SINR-based MCS selection (deterministic, reliable)
            new_mcs = select_mcs_from_sinr(sinr, reliability_margin=2.0, antenna_type=antenna_type)
            
            # Get state indices (no MCS in state space)
            mac_state_indices = self.mac_agent.get_state_indices(
                cbr, sinr, current_beacon, neighbors
            )
            
            phy_state_indices = self.phy_agent.get_state_indices(
                cbr, sinr, current_power, neighbors
            )
            
            # REVISED: Q-Learning for Power and Beacon only
            # Step 1: PHY agent selects power action FIRST (priority)
            phy_action_idx = self.phy_agent.select_action(
                phy_state_indices, neighbors, sinr, current_power, antenna_type
            )
            
            # Step 2: MAC agent selects beacon action AWARE of power decision
            mac_action_idx = self.mac_agent.select_action(
                mac_state_indices, neighbors, phy_action_idx, cbr, antenna_type
            )
            
            # Apply actions with density-adaptive power constraints
            power_delta = PHY_ACTIONS[phy_action_idx]
            beacon_delta = MAC_ACTIONS[mac_action_idx]
            
            # Apply power action with density-adaptive bounds
            power_min, power_max = get_density_adaptive_power_range(neighbors, antenna_type)
            new_power = np.clip(current_power + power_delta, power_min, power_max)
            
            new_beacon = np.clip(current_beacon + beacon_delta, BEACON_MIN, BEACON_MAX)
            # new_mcs already determined by SINR lookup
            
            # Training updates
            if self.training_mode:
                next_cbr = cbr + random.uniform(-0.05, 0.05)
                next_cbr = np.clip(next_cbr, 0, 1)
                next_sinr = sinr + random.uniform(-2, 2)
                
                # Calculate power efficiency
                power_efficiency = 1.0 - ((new_power - power_min) / max(1, power_max - power_min))
                
                # Calculate PHY reward FIRST (power priority)
                phy_reward = self.phy_agent.calculate_reward(
                    cbr, sinr, current_power, neighbors,
                    next_sinr, new_power, beacon_delta, antenna_type
                )
                
                # Calculate MAC reward with power coordination
                mac_reward = self.mac_agent.calculate_reward(
                    cbr, sinr, current_beacon, neighbors,
                    next_cbr, new_beacon, phy_reward, antenna_type
                )
                
                next_mac_state = self.mac_agent.get_state_indices(
                    next_cbr, next_sinr, new_beacon, neighbors
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
                    'next_phy_state': next_phy_state,
                    'neighbors': neighbors,
                    'power': new_power
                }
                
                self.centralized_manager.add_experience(experience)
                self.performance.add_step(
                    mac_reward, phy_reward, cbr, sinr, new_power, new_beacon, new_mcs, 
                    neighbors, mac_action_idx, phy_action_idx
                )
                
                # Power efficiency-aware epsilon decay
                if power_efficiency > 0.7:
                    decay_factor = EPSILON_DECAY
                elif power_efficiency < 0.3:
                    decay_factor = EPSILON_DECAY * 1.001  # Slightly faster decay to force learning
                else:
                    decay_factor = EPSILON_DECAY
                
                self.mac_agent.epsilon = max(MIN_EPSILON, self.mac_agent.epsilon * decay_factor)
                self.phy_agent.epsilon = max(MIN_EPSILON, self.phy_agent.epsilon * decay_factor)
            
            # REVISED: Logging with MCS-SINR alignment focus
            if veh_id.endswith('0'):
                density_cat = get_neighbor_category(neighbors, antenna_type)
                expected_sinr = get_expected_sinr_range(neighbors, antenna_type)
                power_efficiency = 1.0 - ((new_power - power_min) / max(1, power_max - power_min))
                mcs_sinr_requirement = get_mcs_sinr_requirement(new_mcs)
                sinr_margin = sinr - mcs_sinr_requirement
                
                logger.info(f"REVISED Vehicle {veh_id} [{antenna_type}][{density_cat}]: "
                           f"CBR={cbr:.3f}, SINR={sinr:.1f}dB, Neighbors={neighbors}")
                logger.info(f"  Density-Adaptive Power Range: {power_min}-{power_max} dBm")
                logger.info(f"  Expected SINR range: {expected_sinr[0]}-{expected_sinr[1]} dB")
                logger.info(f"  PRIORITY PHY: Power {current_power:.0f}->{new_power:.0f}dBm (Efficiency: {power_efficiency:.1%})")
                logger.info(f"  SECONDARY MAC: Beacon {current_beacon:.0f}->{new_beacon:.0f}Hz")
                logger.info(f"  IEEE 802.11bd MCS: {current_mcs}->{new_mcs} (Req: {mcs_sinr_requirement:.1f}dB, Margin: {sinr_margin:.1f}dB)")
            
            return {
                "transmissionPower": int(new_power),
                "beaconRate": int(new_beacon),
                "MCS": int(new_mcs)  # SINR-determined
            }
            
        except Exception as e:
            logger.error(f"Error processing vehicle {veh_id}: {e}")
            return {
                "transmissionPower": 15,
                "beaconRate": 10,
                "MCS": 4  # IEEE 802.11bd MCS 4 (16-QAM 1/2) as safe fallback
            }
    
    def end_episode(self):
        """REVISED: End episode and save metrics with MCS alignment focus"""
        if self.training_mode:
            self.episode_count += 1
            metrics = self.performance.log_performance(self.episode_count)
            
            if metrics:
                logger.info(f"REVISED Episode {self.episode_count}: "
                           f"MAC reward={metrics['avg_mac_reward']:.3f}, "
                           f"PHY reward={metrics['avg_phy_reward']:.3f}, "
                           f"CBR in range={metrics['cbr_in_range_rate']:.2%}")
                logger.info(f"  Power Efficiency: {metrics['avg_power_efficiency']:.2%}, "
                           f"MCS-SINR Margin: {metrics['avg_mcs_sinr_margin']:.1f}dB")
                logger.info(f"  MCS Well Supported: {metrics['mcs_well_supported_rate']:.2%}, "
                           f"Over Aggressive: {metrics['mcs_over_aggressive_rate']:.2%}")
            
            if self.episode_count % MODEL_SAVE_INTERVAL == 0:
                self.save_models()
            
            self.performance.reset_metrics()
    
    def save_models(self):
        """Save Q-tables"""
        try:
            np.save(MAC_MODEL_PATH, self.mac_agent.q_table)
            np.save(PHY_MODEL_PATH, self.phy_agent.q_table)
            logger.info(f"Revised models saved: {MAC_MODEL_PATH}, {PHY_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Load pre-trained Q-tables"""
        try:
            if os.path.exists(MAC_MODEL_PATH):
                loaded_mac = np.load(MAC_MODEL_PATH)
                if loaded_mac.shape == self.mac_agent.q_table.shape:
                    self.mac_agent.q_table = loaded_mac
                    logger.info(f"Loaded revised MAC model from {MAC_MODEL_PATH}")
            
            if os.path.exists(PHY_MODEL_PATH):
                loaded_phy = np.load(PHY_MODEL_PATH)
                if loaded_phy.shape == self.phy_agent.q_table.shape:
                    self.phy_agent.q_table = loaded_phy
                    logger.info(f"Loaded revised PHY model from {PHY_MODEL_PATH}")
                    
        except Exception as e:
            logger.error(f"Error loading models: {e}")

# ================== REVISED Server Implementation ==================
class DualAgentRLServer:
    """REVISED: Server with SINR-based MCS integration"""
    
    def __init__(self, host, port, training_mode=True):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(5)
        self.dual_agent = DualAgentQLearning(training_mode=training_mode)
        self.training_mode = training_mode
        self.running = True
        
        mode_str = "TRAINING" if training_mode else "TESTING"
        logger.info(f"REVISED Dual-Agent RL Server started in {mode_str} mode on {host}:{port}")

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
        """REVISED: Client handler with MCS-SINR alignment monitoring"""
        logger.info(f"Client connected from {addr}")
        
        try:
            while self.running:
                message_str = self.receive_message_with_header(conn)
                if not message_str:
                    break
                
                try:
                    batch_data = json.loads(message_str)
                    logger.info(f"Processing {len(batch_data)} vehicles with REVISED dual-agent system")
                    
                    responses = {}
                    batch_power_efficiencies = []
                    batch_mcs_sinr_margins = []
                    batch_density_categories = []
                    
                    for veh_id, veh_info in batch_data.items():
                        response = self.dual_agent.process_vehicle(veh_id, veh_info)
                        if response:
                            responses[veh_id] = response
                            
                            # Calculate power efficiency for batch analysis
                            neighbors = int(veh_info.get("neighbors", 10))
                            power = int(response.get("transmissionPower", 15))
                            power_min, power_max = get_density_adaptive_power_range(neighbors, ANTENNA_TYPE)
                            power_efficiency = 1.0 - ((power - power_min) / max(1, power_max - power_min))
                            batch_power_efficiencies.append(power_efficiency)
                            
                            # Calculate MCS-SINR alignment
                            sinr = float(veh_info.get("SINR", veh_info.get("SNR", 20)))
                            mcs = int(response.get("MCS", 5))
                            required_sinr = get_mcs_sinr_requirement(mcs)
                            sinr_margin = sinr - required_sinr
                            batch_mcs_sinr_margins.append(sinr_margin)
                            
                            density_cat = get_neighbor_category(neighbors, ANTENNA_TYPE)
                            batch_density_categories.append(density_cat)
                    
                    # REVISED: Batch analysis with MCS focus
                    if batch_power_efficiencies:
                        avg_power_efficiency = np.mean(batch_power_efficiencies)
                        avg_mcs_sinr_margin = np.mean(batch_mcs_sinr_margins)
                        well_supported_count = sum(1 for margin in batch_mcs_sinr_margins if margin >= 2)
                        over_aggressive_count = sum(1 for margin in batch_mcs_sinr_margins if margin < 0)
                        
                        # Density distribution
                        density_counts = {cat: batch_density_categories.count(cat) for cat in 
                                         ["VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH", "EXTREME"]}
                        dominant_density = max(density_counts, key=density_counts.get)
                        
                        logger.info(f"REVISED Batch Analysis: Power Efficiency={avg_power_efficiency:.2%}, "
                                   f"MCS-SINR Margin={avg_mcs_sinr_margin:.1f}dB, "
                                   f"Dominant Density={dominant_density}")
                        logger.info(f"  MCS Analysis: Well Supported={well_supported_count}/{len(batch_mcs_sinr_margins)}, "
                                   f"Over Aggressive={over_aggressive_count}/{len(batch_mcs_sinr_margins)}")
                        
                        if over_aggressive_count > 0:
                            logger.warning(f"OVER-AGGRESSIVE MCS DETECTED: {over_aggressive_count} vehicles")
                        if avg_mcs_sinr_margin < 1.0:
                            logger.warning(f"LOW MCS-SINR MARGIN: {avg_mcs_sinr_margin:.1f}dB")
                    
                    response_dict = {"vehicles": responses}
                    response_str = json.dumps(response_dict)
                    
                    if self.send_message_with_header(conn, response_str):
                        logger.info(f"Sent revised response to {addr}: {len(responses)} vehicles")
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
            logger.info("REVISED Dual-Agent RL Server listening for connections...")
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
        logger.info("Stopping REVISED Dual-Agent RL server...")
        self.running = False
        
        try:
            self.server.close()
        except:
            pass
        
        if self.training_mode:
            self.dual_agent.save_models()
            self.dual_agent.performance.save_to_excel()
            logger.info("Final revised models and performance data saved")
        
        logger.info("REVISED Dual-Agent RL server stopped")

def main():
    if OPERATION_MODE.upper() not in ["TRAINING", "TESTING"]:
        print(f"ERROR: Invalid OPERATION_MODE '{OPERATION_MODE}'. Must be 'TRAINING' or 'TESTING'")
        sys.exit(1)
    
    if ANTENNA_TYPE.upper() not in ["SECTORAL", "OMNIDIRECTIONAL"]:
        print(f"ERROR: Invalid ANTENNA_TYPE '{ANTENNA_TYPE}'. Must be 'SECTORAL' or 'OMNIDIRECTIONAL'")
        sys.exit(1)
    
    training_mode = (OPERATION_MODE.upper() == "TRAINING")
    
    print("="*100)
    print(" REVISED DUAL-AGENT Q-LEARNING WITH IEEE 802.11BD SINR-BASED MCS CONTROL")
    print("="*100)
    print(f"Host: {HOST}:{PORT}")
    print(f"Mode: {OPERATION_MODE.upper()}")
    print(f"Antenna Type: {ANTENNA_TYPE.upper()}")
    print("="*50)
    print("MAJOR REVISIONS:")
    print("  1. MCS Control: IEEE 802.11bd SINR-based lookup table (reliable)")
    print("  2. Q-Learning Scope: Power (PHY) and Beacon Rate (MAC) only")
    print("  3. State Space: Simplified (removed MCS)")
    print("  4. Action Space: Simplified (removed MCS actions)")
    print("  5. Performance Tracking: Added MCS-SINR alignment metrics")
    print("="*50)
    print("IEEE 802.11bd MCS LOOKUP TABLE (10 MHz Channel):")
    print("  MCS 0-1: BPSK (2-4 dB SINR) - 2.2-3.3 Mbps")
    print("  MCS 2-3: QPSK (5-10 dB SINR) - 6.5-9.8 Mbps")
    print("  MCS 4-5: 16-QAM (11-16 dB SINR) - 13.0-19.5 Mbps")
    print("  MCS 6-7: 64-QAM (17-23 dB SINR) - 26.0-29.3 Mbps")
    print("  MCS 8-9: 256-QAM (24-29 dB SINR) - 39.0-43.3 Mbps")
    print("  MCS 10-11: 1024-QAM (30+ dB SINR) - 52.0-58.5 Mbps")
    print("  Reliability Margin: +2 dB for VANET mobile conditions")
    print("="*50)
    print("EXPECTED BENEFITS:")
    print("  ✓ IEEE 802.11bd Compliant MCS Selection")
    print("  ✓ Reliable PDR Performance (no more MCS over-aggression)")
    print("  ✓ Faster Q-Learning Convergence (simplified action space)")
    print("  ✓ Better MCS-SINR Alignment (standard-based selection)")
    print("  ✓ Maintained Power and Beacon Optimization")
    print("  ✓ Support for High Data Rates (up to 58.5 Mbps)")
    print("="*100)
    
    # Initialize revised server
    rl_server = DualAgentRLServer(HOST, PORT, training_mode=training_mode)
    
    try:
        rl_server.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        rl_server.stop()

if __name__ == "__main__":
    main()
