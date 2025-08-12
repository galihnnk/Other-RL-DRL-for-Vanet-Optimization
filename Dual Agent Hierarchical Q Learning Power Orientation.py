"""
HIERARCHICAL MULTI-AGENT Q-LEARNING VANET OPTIMIZATION SYSTEM
==============================================================
Enhanced version with Meta-Controller + MAC/PHY Agents coordination
Author: Based on Galih Nugraha Nurkahfi's work with hierarchical improvements

ARCHITECTURE:
1. Meta-Controller: Decides high-level strategy (CONSERVATIVE/BALANCED/AGGRESSIVE)
2. MAC Agent: Optimizes beacon rate & MCS under strategy guidance  
3. PHY Agent: Optimizes power transmission under strategy guidance
4. Shared state space and coordinated rewards for better cooperation

KEY IMPROVEMENTS:
- Strategy-guided agent coordination
- Joint reward system with cooperation bonuses
- Unified state representation across all agents
- Message passing between agents
- Better exploration coordination
"""

import socket
import threading
import numpy as np
import json
import random
import os
import time
import math
import signal
import sys
from datetime import datetime
from collections import defaultdict, deque
import pandas as pd
from scipy.stats import entropy
import logging

# ================== CONFIGURATION ==================
OPERATION_MODE = "TRAINING"        # Options: "TRAINING" or "TESTING"
ANTENNA_TYPE = "SECTORAL"          # Options: "SECTORAL" or "OMNIDIRECTIONAL"

# ================== POWER-CENTRIC HIERARCHICAL CONSTANTS ==================
# Strategy types for Meta-Controller
STRATEGIES = ["CONSERVATIVE", "BALANCED", "AGGRESSIVE"]
STRATEGY_COUNT = len(STRATEGIES)

# Performance targets
CBR_TARGET = 0.4                   # Optimal CBR for latency/PDR
CBR_RANGE = (0.35, 0.45)          # Acceptable CBR range
SINR_TARGET = 12.0                 # Target SINR
SINR_GOOD_THRESHOLD = 12.0         # Threshold for diminishing returns

# ================== DENSITY-ADAPTIVE POWER RANGES (UPDATED) ==================
# Power exploration ranges based on realistic VANET density scenarios
POWER_RANGES = {
    "VERY_LOW": (15, 30),       # Rural/Highway: can use higher power for range
    "LOW": (12, 25),            # Sparse Urban: good power range
    "MEDIUM": (8, 18),          # Normal Urban: moderate power
    "HIGH": (5, 12),            # Dense Urban: reduced power
    "VERY_HIGH": (3, 8),        # Traffic Jam: low power to reduce interference
    "EXTREME": (1, 5)           # Extreme Congestion: minimal power only
}

# System parameters
BUFFER_SIZE = 100000
LEARNING_RATE = 0.15
DISCOUNT_FACTOR = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.1
HOST = '127.0.0.1'
PORT = 5000

# Parameter ranges
POWER_MIN = 1
POWER_MAX = 30
BEACON_MIN = 1
BEACON_MAX = 20
MCS_MIN = 0
MCS_MAX = 9

# File paths for power-centric hierarchical system
MODEL_PREFIX = f"{ANTENNA_TYPE.lower()}_power_centric"
META_MODEL_PATH = f'{MODEL_PREFIX}_meta_controller.npy'
POWER_MODEL_PATH = f'{MODEL_PREFIX}_power_agent.npy'
MAC_MODEL_PATH = f'{MODEL_PREFIX}_mac_agent.npy'
PERFORMANCE_LOG_PATH = f'{MODEL_PREFIX}_performance.xlsx'
MODEL_SAVE_INTERVAL = 50
PERFORMANCE_LOG_INTERVAL = 10

# ================== UPDATED DENSITY CATEGORIZATION (6-LEVEL REALISTIC) ==================
def get_neighbor_category(neighbor_count, antenna_type="OMNIDIRECTIONAL"):
    """REALISTIC: VANET density categorization with 6 levels for better precision"""
    
    if antenna_type.upper() == "SECTORAL":
        # Sectoral antenna: reduce effective neighbors by 30% (0.7 factor)
        effective_neighbors = neighbor_count * 0.7
        
        if effective_neighbors <= 2:        # ≤3 total - Rural/Highway
            return "VERY_LOW"
        elif effective_neighbors <= 4:      # ≤6 total - Sparse Urban  
            return "LOW"
        elif effective_neighbors <= 8:      # ≤12 total - Normal Urban
            return "MEDIUM"
        elif effective_neighbors <= 12:     # ≤18 total - Dense Urban
            return "HIGH"
        elif effective_neighbors <= 18:     # ≤25 total - Traffic Jam
            return "VERY_HIGH"
        else:                                # >25 total - Extreme Congestion
            return "EXTREME"
    else:
        # Omnidirectional antenna
        if neighbor_count <= 2:             # Rural/Highway
            return "VERY_LOW"
        elif neighbor_count <= 5:           # Sparse Urban
            return "LOW"
        elif neighbor_count <= 10:          # Normal Urban
            return "MEDIUM"
        elif neighbor_count <= 15:          # Dense Urban
            return "HIGH"
        elif neighbor_count <= 22:          # Traffic Jam
            return "VERY_HIGH"
        else:                                # >22 - Extreme Congestion
            return "EXTREME"

def get_expected_sinr_range(neighbor_count, antenna_type="OMNIDIRECTIONAL"):
    """Updated SINR ranges for new 6-level density system"""
    category = get_neighbor_category(neighbor_count, antenna_type)
    
    if antenna_type.upper() == "SECTORAL":
        ranges = {
            "VERY_LOW": (25, 40),       # Rural/Highway - excellent SINR
            "LOW": (18, 35),            # Sparse Urban - very good SINR
            "MEDIUM": (11, 25),         # Normal Urban - good SINR
            "HIGH": (5, 17),            # Dense Urban - moderate SINR
            "VERY_HIGH": (-2, 13),      # Traffic Jam - poor SINR
            "EXTREME": (-8, 8)          # Extreme Congestion - very poor SINR
        }
    else:
        ranges = {
            "VERY_LOW": (20, 35),       # Rural/Highway
            "LOW": (15, 30),            # Sparse Urban
            "MEDIUM": (8, 20),          # Normal Urban
            "HIGH": (2, 12),            # Dense Urban
            "VERY_HIGH": (-5, 8),       # Traffic Jam
            "EXTREME": (-10, 5)         # Extreme Congestion
        }
    
    return ranges.get(category, (8, 20))

def get_density_multiplier(neighbor_count, antenna_type="OMNIDIRECTIONAL"):
    """Updated density-based reward multiplier for 6 levels"""
    category = get_neighbor_category(neighbor_count, antenna_type)
    multipliers = {
        "VERY_LOW": 0.6,            # Low challenge
        "LOW": 0.8,                 # Moderate challenge
        "MEDIUM": 1.0,              # Baseline challenge
        "HIGH": 1.4,                # High challenge
        "VERY_HIGH": 1.8,           # Very high challenge
        "EXTREME": 2.2              # Extreme challenge
    }
    return multipliers.get(category, 1.0)

# ================== UNIFIED STATE DISCRETIZATION ==================
CBR_BINS = np.linspace(0.0, 1.0, 21)           # 20 states
SINR_BINS = np.linspace(0, 50, 11)             # 10 states  
NEIGHBORS_BINS = np.linspace(0, 30, 9)         # 8 states to handle up to EXTREME density
POWER_BINS = np.arange(1, 31)                  # 30 states
BEACON_BINS = np.arange(1, 21)                 # 20 states
MCS_BINS = np.arange(0, 10)                    # 10 states
STRATEGY_BINS = np.arange(0, 3)                # 3 strategies

# ================== POWER-CENTRIC STATE DIMENSIONS (UPDATED) ==================
# Updated dimensions to handle 6-level density system
CBR_STATES = 10          # CBR discretization
SINR_STATES = 8          # SINR discretization  
NEIGHBORS_STATES = 8     # Updated from 6 to better handle EXTREME category
POWER_STATES = 15        # Power discretization
BEACON_STATES = 10       # Beacon discretization
MCS_STATES = 8           # MCS discretization

UNIFIED_STATE_DIM = (10, 8, 8)                 # (CBR, SINR, Neighbors) - updated neighbors
META_STATE_DIM = (10, 8, 8)                    # Same as unified for meta decisions
POWER_STATE_DIM = UNIFIED_STATE_DIM + (3,)     # + Strategy (PRIMARY)
MAC_STATE_DIM = (10, 8, 8, 3, 5)              # CBR, SINR, Neighbors, Strategy, Power_Change_Category

# ================== POWER-CENTRIC ACTION SPACES ==================
# Meta-Controller actions (strategy selection)
META_ACTIONS = list(range(STRATEGY_COUNT))  # [0=CONSERVATIVE, 1=BALANCED, 2=AGGRESSIVE]

# PRIMARY Power Agent actions - density-adaptive and strategy-aware
POWER_ACTIONS = [
    # Fine adjustments (Conservative strategy preferred)
    0, 1, -1, 2, -2,
    # Moderate adjustments (Balanced strategy preferred)  
    3, -3, 5, -5,
    # Aggressive adjustments (Aggressive strategy preferred)
    7, -7, 10, -10, 15, -15
]

# SECONDARY MAC Agent actions - simplified since power is primary
MAC_ACTIONS = [
    # Conservative MAC adjustments
    (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
    # Moderate MAC adjustments
    (2, 0), (-2, 0), (0, 2), (0, -2), (1, 1), (-1, -1),
    # Fine-tuning adjustments
    (3, 0), (-3, 0), (0, 3), (0, -3), (2, 1), (-2, -1)
]

META_ACTION_DIM = len(META_ACTIONS)
POWER_ACTION_DIM = len(POWER_ACTIONS) 
MAC_ACTION_DIM = len(MAC_ACTIONS)

# Initialize Q-tables for power-centric hierarchical system
meta_q_table = np.zeros(META_STATE_DIM + (META_ACTION_DIM,), dtype=np.float32)
power_q_table = np.zeros(POWER_STATE_DIM + (POWER_ACTION_DIM,), dtype=np.float32)
mac_q_table = np.zeros(MAC_STATE_DIM + (MAC_ACTION_DIM,), dtype=np.float32)

# ================== LOGGING SETUP ==================
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

logger.info("POWER-CENTRIC HIERARCHICAL MULTI-AGENT Q-LEARNING SYSTEM INITIALIZED")
logger.info(f"Meta-Controller Q-table shape: {meta_q_table.shape}")
logger.info(f"PRIMARY Power Agent Q-table shape: {power_q_table.shape}")
logger.info(f"SECONDARY MAC Agent Q-table shape: {mac_q_table.shape}")

# ================== HELPER FUNCTIONS ==================
def discretize(value, bins):
    """Enhanced discretization with proper bounds checking for reduced dimensions"""
    if not np.isfinite(value):
        value = 0.0
    
    # Handle specific ranges based on the bins
    if isinstance(bins, np.ndarray) and len(bins) > 10:
        # For continuous ranges like CBR, SINR
        value = np.clip(value, bins[0], bins[-1])
        bin_idx = np.digitize(value, bins) - 1
        return max(0, min(len(bins) - 2, bin_idx))
    else:
        # For smaller discrete ranges
        if hasattr(bins, '__len__') and len(bins) <= 10:
            # Small discrete ranges
            value = int(np.clip(value, 0, len(bins) - 1))
            return value
        else:
            # Default handling
            value = np.clip(value, bins[0], bins[-1])
            bin_idx = np.digitize(value, bins) - 1
            return max(0, min(len(bins) - 2, bin_idx))

def get_unified_state_indices(cbr, sinr, neighbors):
    """UPDATED unified state representation for 6-level density system"""
    cbr_idx = discretize(cbr, CBR_BINS)
    sinr_idx = discretize(sinr, SINR_BINS)  
    neighbors_idx = discretize(neighbors, NEIGHBORS_BINS)
    
    # Ensure indices are within bounds for updated dimensions
    cbr_idx = min(cbr_idx, CBR_STATES - 1)
    sinr_idx = min(sinr_idx, SINR_STATES - 1)
    neighbors_idx = min(neighbors_idx, NEIGHBORS_STATES - 1)  # Now supports up to 7 (0-7 for 8 states)
    
    return (cbr_idx, sinr_idx, neighbors_idx)

def categorize_power_change(power_change):
    """Categorize power changes into discrete categories for MAC agent"""
    if power_change <= -5:
        return 0  # Large decrease
    elif power_change <= -2:
        return 1  # Small decrease  
    elif power_change == 0:
        return 2  # No change
    elif power_change <= 3:
        return 3  # Small increase
    else:
        return 4  # Large increase

def get_density_adaptive_power_range(neighbor_count, antenna_type="OMNIDIRECTIONAL"):
    """
    REALISTIC: More appropriate power ranges for actual VANET scenarios
    Much lower ranges to encourage efficiency learning
    """
    
    if antenna_type.upper() == "SECTORAL":
        # Use 0.7 factor for sectoral antenna directional focus
        effective_neighbors = neighbor_count * 0.7
        
        if effective_neighbors <= 2:        # VERY_LOW density (Rural/Highway)
            return (15, 30)                  # Can use higher power for range
        elif effective_neighbors <= 4:      # LOW density (Sparse Urban)
            return (12, 25)                  # Good power range
        elif effective_neighbors <= 8:      # MEDIUM density (Normal Urban)
            return (8, 18)                   # Moderate power range
        elif effective_neighbors <= 12:     # HIGH density (Dense Urban)
            return (5, 12)                   # Reduced power range
        elif effective_neighbors <= 18:     # VERY_HIGH density (Traffic Jam)
            return (3, 8)                    # Low power range
        else:                                # EXTREME density (Extreme Congestion)
            return (1, 5)                    # Minimal power range
    else:
        # Omnidirectional antenna - more conservative ranges
        if neighbor_count <= 2:             # VERY_LOW (Rural/Highway)
            return (15, 30)
        elif neighbor_count <= 5:           # LOW (Sparse Urban)
            return (12, 25)
        elif neighbor_count <= 10:          # MEDIUM (Normal Urban)
            return (8, 18)
        elif neighbor_count <= 15:          # HIGH (Dense Urban)
            return (5, 12)
        elif neighbor_count <= 22:          # VERY_HIGH (Traffic Jam)
            return (3, 8)
        else:                                # EXTREME (Extreme Congestion)
            return (1, 5)

def get_strategy_parameters(strategy_idx):
    """Get strategy-specific parameters for power-centric system"""
    strategies = {
        0: {  # CONSERVATIVE
            'exploration_multiplier': 0.7,
            'power_preference': 'minimal',     # Prefer minimal power usage
            'power_change_limit': 3,           # Max power change
            'mac_change_limit': 1,             # Conservative MAC changes
            'cbr_priority': 0.8,               # High priority on CBR reduction
            'sinr_priority': 0.4               # Lower priority on SINR boost
        },
        1: {  # BALANCED  
            'exploration_multiplier': 1.0,
            'power_preference': 'adaptive',    # Adaptive power based on conditions
            'power_change_limit': 7,           # Moderate power changes
            'mac_change_limit': 2,             # Moderate MAC changes
            'cbr_priority': 0.6,               # Balanced CBR priority
            'sinr_priority': 0.6               # Balanced SINR priority
        },
        2: {  # AGGRESSIVE
            'exploration_multiplier': 1.3,
            'power_preference': 'performance', # Prioritize performance over efficiency
            'power_change_limit': 15,          # Large power changes allowed
            'mac_change_limit': 3,             # More aggressive MAC changes
            'cbr_priority': 0.4,               # Lower CBR priority (accept higher CBR)
            'sinr_priority': 0.8               # High priority on SINR achievement
        }
    }
    return strategies.get(strategy_idx, strategies[1])

# ================== REWARD FUNCTIONS ==================
def calculate_joint_reward(cbr, sinr, power, beacon, mcs, neighbors, 
                          next_cbr, next_sinr, next_power, next_beacon, next_mcs,
                          strategy_idx, antenna_type="OMNIDIRECTIONAL"):
    """
    ENHANCED: Joint reward function that encourages cooperation
    Combines CBR, SINR, efficiency, and coordination bonuses
    """
    
    # 1. CBR Performance (Primary objective)
    cbr_error = abs(cbr - CBR_TARGET)
    cbr_reward = 15.0 * (1 - math.tanh(25 * cbr_error))
    
    # 2. SINR Performance with diminishing returns
    if sinr < SINR_TARGET:
        sinr_reward = 10.0 * (sinr / SINR_TARGET)
    else:
        base_sinr_reward = 10.0
        excess_sinr = sinr - SINR_TARGET
        diminishing_reward = 5.0 * math.sqrt(excess_sinr / 10.0)
        sinr_reward = min(base_sinr_reward + diminishing_reward, 18.0)
    
    # 3. Strategy-aware efficiency bonus
    strategy_params = get_strategy_parameters(strategy_idx)
    power_norm = (power - POWER_MIN) / (POWER_MAX - POWER_MIN)
    
    if strategy_idx == 0:  # CONSERVATIVE - reward low power usage
        if power_norm <= 0.4:
            efficiency_bonus = 5.0 * (0.4 - power_norm)
        else:
            efficiency_bonus = -3.0 * (power_norm - 0.4) ** 2
    elif strategy_idx == 2:  # AGGRESSIVE - allow higher power if needed
        if sinr >= SINR_TARGET:
            efficiency_bonus = 3.0
        else:
            efficiency_bonus = -2.0 * (SINR_TARGET - sinr) / SINR_TARGET
    else:  # BALANCED
        optimal_power_norm = 0.3 + (0.4 * math.log(1 + neighbors) / math.log(1 + 40))
        power_deviation = abs(power_norm - optimal_power_norm)
        efficiency_bonus = 4.0 * max(0, 0.2 - power_deviation)
    
    # 4. Parameter coordination bonus (NEW)
    # Reward when parameter changes are aligned with strategy
    coordination_bonus = 0
    
    power_change = next_power - power
    beacon_change = next_beacon - beacon
    mcs_change = next_mcs - mcs
    
    if strategy_idx == 0:  # CONSERVATIVE - reward small changes
        if abs(power_change) <= 2 and abs(beacon_change) <= 1 and abs(mcs_change) <= 1:
            coordination_bonus = 3.0
    elif strategy_idx == 2:  # AGGRESSIVE - reward decisive changes when needed
        if cbr > 0.6 or sinr < 8:  # Poor performance
            if abs(power_change) >= 3 or abs(beacon_change) >= 2:
                coordination_bonus = 4.0
    else:  # BALANCED - reward moderate, well-reasoned changes
        total_change = abs(power_change) + abs(beacon_change) + abs(mcs_change)
        if 2 <= total_change <= 6:
            coordination_bonus = 2.0
    
    # 5. Density-aware performance bonus
    density_multiplier = get_density_multiplier(neighbors, antenna_type)
    density_bonus = 0
    
    if cbr <= CBR_RANGE[1] and sinr >= SINR_TARGET:
        density_bonus = 3.0 * density_multiplier
    
    # 6. Antenna-specific bonus
    antenna_bonus = 0
    if antenna_type.upper() == "SECTORAL":
        # Sectoral antennas get bonus for using higher MCS/beacon efficiently
        if mcs >= 6 and beacon >= 12 and sinr >= SINR_TARGET:
            antenna_bonus = 2.0
    
    # 7. Smoothness penalty (prevent oscillations)
    smoothness_penalty = -0.5 * (abs(power_change) + abs(beacon_change) + abs(mcs_change))
    
    total_reward = (cbr_reward + sinr_reward + efficiency_bonus + 
                   coordination_bonus + density_bonus + antenna_bonus + smoothness_penalty)
    
    return np.clip(total_reward, -30, 40)

# ================== META-CONTROLLER ==================
class MetaController:
    """High-level strategy controller that guides MAC and PHY agents"""
    
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon
        self.q_table = meta_q_table
        self.state_visit_counts = defaultdict(int)
        self.strategy_history = []
        self.performance_history = []
        
    def get_state_indices(self, cbr, sinr, neighbors):
        """Simplified state space for meta-decisions with reduced dimensions"""
        cbr_idx = discretize(cbr, CBR_BINS)
        sinr_idx = discretize(sinr, SINR_BINS)
        neighbors_idx = discretize(neighbors, NEIGHBORS_BINS)
        
        # Ensure indices are within bounds
        cbr_idx = min(cbr_idx, CBR_STATES - 1)
        sinr_idx = min(sinr_idx, SINR_STATES - 1)
        neighbors_idx = min(neighbors_idx, NEIGHBORS_STATES - 1)
        
        return (cbr_idx, sinr_idx, neighbors_idx)
    
    def select_strategy(self, state_indices, cbr, sinr, neighbors, antenna_type="OMNIDIRECTIONAL"):
        """Select high-level strategy based on network conditions"""
        self.state_visit_counts[state_indices] += 1
        
        # Adaptive epsilon based on network conditions
        adaptive_epsilon = self.epsilon
        
        # Increase exploration in extreme conditions
        if cbr > 0.7 or sinr < 5:
            adaptive_epsilon = min(1.0, self.epsilon * 1.5)
        elif CBR_RANGE[0] <= cbr <= CBR_RANGE[1] and sinr >= SINR_TARGET:
            adaptive_epsilon = max(0.05, self.epsilon * 0.5)  # Less exploration when doing well
        
        if random.random() < adaptive_epsilon:
            # Intelligent exploration based on current conditions
            if cbr > 0.6:  # High CBR - need CONSERVATIVE approach
                if random.random() < 0.7:
                    return 0  # CONSERVATIVE
                else:
                    return random.choice([0, 1])
            elif cbr < 0.2 or sinr < 8:  # Poor performance - try AGGRESSIVE
                if random.random() < 0.7:
                    return 2  # AGGRESSIVE
                else:
                    return random.choice([1, 2])
            else:  # Normal conditions - any strategy
                return random.randint(0, META_ACTION_DIM - 1)
        else:
            return np.argmax(self.q_table[state_indices])
    
    def calculate_meta_reward(self, cbr, sinr, neighbors, strategy_idx, 
                             joint_reward, antenna_type="OMNIDIRECTIONAL"):
        """Calculate reward for meta-controller based on strategy effectiveness"""
        
        # Base reward from joint performance
        meta_reward = joint_reward * 0.3  # Scale down since meta gets credit for coordination
        
        # Strategy appropriateness bonus
        strategy_bonus = 0
        density_category = get_neighbor_category(neighbors, antenna_type)
        
        if strategy_idx == 0:  # CONSERVATIVE
            if cbr > 0.5 or density_category in ["HIGH", "VERY_HIGH", "EXTREME"]:
                strategy_bonus = 5.0  # Good choice in crowded conditions
            elif cbr < 0.3:
                strategy_bonus = -3.0  # Poor choice when network underutilized
                
        elif strategy_idx == 2:  # AGGRESSIVE
            if cbr < 0.3 or sinr < 8:
                strategy_bonus = 5.0  # Good choice when performance is poor
            elif cbr > 0.6:
                strategy_bonus = -4.0  # Poor choice in congested conditions
                
        else:  # BALANCED
            if 0.3 <= cbr <= 0.5 and sinr >= 10:
                strategy_bonus = 3.0  # Good general-purpose choice
        
        # Consistency bonus (reward stable strategy when performing well)
        if len(self.strategy_history) >= 3:
            recent_strategies = self.strategy_history[-3:]
            if len(set(recent_strategies)) == 1 and joint_reward > 10:  # Consistent and good
                consistency_bonus = 2.0
            elif len(set(recent_strategies)) == 3:  # Too much switching
                consistency_bonus = -1.0
            else:
                consistency_bonus = 0
        else:
            consistency_bonus = 0
        
        total_meta_reward = meta_reward + strategy_bonus + consistency_bonus
        
        # Track strategy performance
        self.strategy_history.append(strategy_idx)
        self.performance_history.append(joint_reward)
        
        # Keep history manageable
        if len(self.strategy_history) > 100:
            self.strategy_history = self.strategy_history[-50:]
            self.performance_history = self.performance_history[-50:]
        
        return np.clip(total_meta_reward, -20, 25)
    
    def update_q_table(self, state_indices, action, reward, next_state_indices):
        """Update meta-controller Q-table"""
        current_q = self.q_table[state_indices][action]
        max_next_q = np.max(self.q_table[next_state_indices])
        
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q - current_q)
        self.q_table[state_indices][action] = new_q
        
        return new_q - current_q

# ================== PRIMARY POWER AGENT ==================
class PrimaryPowerAgent:
    """PRIMARY Power Agent - Main controller for power transmission optimization"""
    
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon
        self.q_table = power_q_table
        self.state_visit_counts = defaultdict(int)
        self.power_history = []
        self.performance_history = []
        self.last_strategy = 1  # Default to BALANCED
        
    def get_state_indices(self, unified_state, strategy_idx):
        """Get state indices with strategy guidance for reduced dimensions"""
        # Ensure strategy index is within bounds
        strategy_idx = min(strategy_idx, 2)
        return unified_state + (strategy_idx,)
    
    def select_action(self, state_indices, strategy_idx, current_power, current_cbr, current_sinr,
                     neighbors, antenna_type="OMNIDIRECTIONAL"):
        """PRIMARY power action selection with 6-level density-adaptive exploration"""
        self.state_visit_counts[state_indices] += 1
        self.last_strategy = strategy_idx
        
        # Get strategy parameters and density-adaptive power range
        strategy_params = get_strategy_parameters(strategy_idx)
        power_min, power_max = get_density_adaptive_power_range(neighbors, antenna_type)
        density_category = get_neighbor_category(neighbors, antenna_type)
        
        # Calculate adaptive epsilon with efficiency bias
        adaptive_epsilon = self.epsilon * strategy_params['exploration_multiplier']
        
        # Much more aggressive epsilon reduction for efficiency learning
        if current_cbr > 0.6:  # High CBR - need aggressive power reduction
            adaptive_epsilon = min(1.0, adaptive_epsilon * 1.5)
        elif current_sinr < 8:  # Poor SINR - need careful power adjustment
            adaptive_epsilon = min(1.0, adaptive_epsilon * 1.3)
        elif power_min <= current_power <= (power_min + power_max) / 2 and CBR_RANGE[0] <= current_cbr <= CBR_RANGE[1]:
            # Good performance with reasonable power - much less exploration
            adaptive_epsilon = max(0.01, adaptive_epsilon * 0.2)  # Very low exploration
        elif current_power > (power_min + power_max) * 0.8:  # High power usage
            # Force learning of lower power - bias toward power reduction
            adaptive_epsilon = max(0.05, adaptive_epsilon * 0.4)
        
        if random.random() < adaptive_epsilon:
            # INTELLIGENT 6-LEVEL DENSITY-AWARE EXPLORATION
            
            if density_category == "EXTREME":  # >25 vehicles (OMNIDIRECTIONAL) / >25 total (SECTORAL)
                # Critical congestion - emergency power reduction only
                preferred_actions = [i for i, p in enumerate(POWER_ACTIONS) if p <= -5]
                
            elif density_category == "VERY_HIGH":  # Traffic jam scenario (15-22 OMNIDIRECTIONAL / 18-25 SECTORAL)
                if current_cbr > 0.5:
                    preferred_actions = [i for i, p in enumerate(POWER_ACTIONS) if p <= -3]
                elif current_power > power_max:
                    preferred_actions = [i for i, p in enumerate(POWER_ACTIONS) if p <= -2]
                else:
                    preferred_actions = [i for i, p in enumerate(POWER_ACTIONS) if abs(p) <= 2]
                    
            elif density_category == "HIGH":  # Dense urban (10-15 OMNIDIRECTIONAL / 12-18 SECTORAL)
                if current_cbr > 0.55:
                    preferred_actions = [i for i, p in enumerate(POWER_ACTIONS) if p <= -2]
                elif current_sinr < SINR_TARGET and current_power < power_max:
                    preferred_actions = [i for i, p in enumerate(POWER_ACTIONS) if 1 <= p <= 3]
                else:
                    preferred_actions = [i for i, p in enumerate(POWER_ACTIONS) if abs(p) <= 3]
                    
            elif density_category == "MEDIUM":  # Normal urban (5-10 OMNIDIRECTIONAL / 8-12 SECTORAL)
                if current_cbr > 0.5:
                    preferred_actions = [i for i, p in enumerate(POWER_ACTIONS) if p <= 0]
                elif current_sinr < SINR_TARGET:
                    preferred_actions = [i for i, p in enumerate(POWER_ACTIONS) if 1 <= p <= 5]
                else:
                    preferred_actions = [i for i, p in enumerate(POWER_ACTIONS) if abs(p) <= 5]
                    
            elif density_category == "LOW":  # Sparse urban (3-5 OMNIDIRECTIONAL / 4-8 SECTORAL)
                if current_sinr < SINR_TARGET - 2:
                    preferred_actions = [i for i, p in enumerate(POWER_ACTIONS) if p >= 2]
                elif current_cbr > 0.6:  # Unusual in low density
                    preferred_actions = [i for i, p in enumerate(POWER_ACTIONS) if p <= -1]
                else:
                    preferred_actions = [i for i, p in enumerate(POWER_ACTIONS) if abs(p) <= 7]
                    
            else:  # VERY_LOW - Rural/Highway (≤2 OMNIDIRECTIONAL / ≤3 SECTORAL)
                if current_sinr < SINR_TARGET - 5:
                    preferred_actions = [i for i, p in enumerate(POWER_ACTIONS) if p >= 5]
                else:
                    preferred_actions = list(range(POWER_ACTION_DIM))  # Any action
            
            # Log realistic density scenario
            if antenna_type.upper() == "SECTORAL" and random.random() < 0.05:  # 5% logging
                scenario_map = {
                    "VERY_LOW": "Rural/Highway", "LOW": "Sparse Urban", "MEDIUM": "Normal Urban",
                    "HIGH": "Dense Urban", "VERY_HIGH": "Traffic Jam", "EXTREME": "Extreme Congestion"
                }
                logger.debug(f"REALISTIC VANET: {neighbors} neighbors → {density_category} ({scenario_map[density_category]}), "
                           f"Power range={power_min}-{power_max}dBm")
            
            # Strategy-specific bias with efficiency focus
            if strategy_idx == 0:  # CONSERVATIVE
                conservative_actions = [i for i, p in enumerate(POWER_ACTIONS) if abs(p) <= strategy_params['power_change_limit']]
                if current_power > (power_min + power_max) / 2:
                    conservative_actions = [i for i, p in enumerate(conservative_actions) if POWER_ACTIONS[i] <= -1]
                preferred_actions = list(set(preferred_actions) & set(conservative_actions)) if preferred_actions else conservative_actions
                
            elif strategy_idx == 2:  # AGGRESSIVE
                if current_cbr > 0.6 or current_sinr < 8:
                    aggressive_actions = [i for i, p in enumerate(POWER_ACTIONS) if abs(p) >= 3]
                elif current_sinr >= SINR_TARGET and current_power > power_min + 5:
                    aggressive_actions = [i for i, p in enumerate(POWER_ACTIONS) if p <= -2]
                else:
                    aggressive_actions = [i for i, p in enumerate(POWER_ACTIONS) if abs(p) >= 2]
                preferred_actions = aggressive_actions if aggressive_actions else preferred_actions
                
            else:  # BALANCED - add efficiency bias
                if current_sinr >= SINR_TARGET - 2 and current_cbr <= 0.5:
                    efficiency_actions = [i for i, p in enumerate(POWER_ACTIONS) if p <= 1]
                    preferred_actions = efficiency_actions if efficiency_actions else preferred_actions
            
            # Bounds checking for density-adaptive ranges
            valid_actions = []
            for i in (preferred_actions if preferred_actions else list(range(POWER_ACTION_DIM))):
                new_power = current_power + POWER_ACTIONS[i]
                if power_min <= new_power <= power_max:
                    valid_actions.append(i)
            
            if valid_actions:
                action = random.choice(valid_actions)
            else:
                action = 0  # No change fallback
                
            self.power_history.append(('exploration', current_power, POWER_ACTIONS[action], density_category))
            return action
        else:
            # EXPLOITATION
            action = np.argmax(self.q_table[state_indices])
            self.power_history.append(('exploitation', current_power, POWER_ACTIONS[action], density_category))
            return action
    
    def calculate_power_reward(self, cbr, sinr, neighbors, strategy_idx, current_power, 
                              next_power, next_cbr, next_sinr, antenna_type="OMNIDIRECTIONAL"):
        """PRIMARY power agent reward - heavily weighted since it's the main controller"""
        
        strategy_params = get_strategy_parameters(strategy_idx)
        power_min, power_max = get_density_adaptive_power_range(neighbors, antenna_type)
        density_category = get_neighbor_category(neighbors, antenna_type)
        
        # 1. CBR PERFORMANCE (Highest priority - 40% weight)
        cbr_error = abs(cbr - CBR_TARGET)
        cbr_reward = 20.0 * (1 - math.tanh(25 * cbr_error)) * strategy_params['cbr_priority']
        
        # Massive penalty for very high CBR (congestion crisis)
        if cbr > 0.7:
            cbr_reward -= 30.0 * (cbr - 0.7) ** 2
        elif cbr > 0.6:
            cbr_reward -= 15.0 * (cbr - 0.6) ** 2
        
        # 2. SINR PERFORMANCE (30% weight)
        if sinr < SINR_TARGET:
            sinr_reward = 15.0 * (sinr / SINR_TARGET) * strategy_params['sinr_priority']
        else:
            base_sinr_reward = 15.0
            excess_sinr = sinr - SINR_TARGET
            diminishing_reward = 5.0 * math.sqrt(excess_sinr / 10.0)
            sinr_reward = min(base_sinr_reward + diminishing_reward, 22.0) * strategy_params['sinr_priority']
        
        # 3. ENHANCED POWER EFFICIENCY - much more aggressive penalties
        power_efficiency = 0
        power_norm = (current_power - power_min) / max(1, power_max - power_min)
        
        # Calculate optimal power based on SINR performance
        sinr_sufficient = sinr >= SINR_TARGET
        
        if strategy_idx == 0:  # CONSERVATIVE - reward very low power usage
            if power_norm <= 0.3:
                power_efficiency = 15.0 * (0.3 - power_norm)  # Strong efficiency bonus
            elif power_norm > 0.6:
                power_efficiency = -20.0 * (power_norm - 0.6) ** 2  # Harsh penalty
                
        elif strategy_idx == 2:  # AGGRESSIVE - only allow high power if SINR is insufficient
            if sinr_sufficient and power_norm > 0.7:
                power_efficiency = -25.0 * (power_norm - 0.7) ** 2  # Very harsh penalty for excess
            elif not sinr_sufficient and sinr < SINR_TARGET - 5:
                power_efficiency = 8.0  # Allow high power for poor SINR
            else:
                power_efficiency = -5.0 * power_norm  # General efficiency pressure
                
        else:  # BALANCED - optimal power curve
            # Target lower power usage: optimal at 30% of range
            optimal_power_ratio = 0.3 + (0.3 * neighbors / 50)  # 0.3 to 0.6 range
            power_deviation = abs(power_norm - optimal_power_ratio)
            
            if power_deviation <= 0.15:
                power_efficiency = 12.0 * (0.15 - power_deviation)  # Strong efficiency reward
            else:
                power_efficiency = -10.0 * power_deviation ** 2  # Quadratic penalty for deviation
        
        # ADDITIONAL PENALTY: If SINR is sufficient but power is high
        if sinr >= SINR_TARGET + 2 and power_norm > 0.5:  # Good SINR but high power
            excess_power_penalty = -15.0 * (power_norm - 0.5) ** 2
            power_efficiency += excess_power_penalty
        
        # 4. DENSITY-APPROPRIATE POWER USAGE (10% weight)
        density_appropriateness = 0
        if power_min <= current_power <= power_max:
            density_appropriateness = 5.0
        elif current_power > power_max:
            excess = current_power - power_max
            density_appropriateness = -3.0 * excess  # Linear penalty for excess
        elif current_power < power_min:
            deficit = power_min - current_power
            density_appropriateness = -2.0 * deficit  # Moderate penalty for too low
        
        # 5. POWER CHANGE APPROPRIATENESS
        power_change = abs(next_power - current_power)
        change_penalty = 0
        
        if power_change > strategy_params['power_change_limit']:
            change_penalty = -2.0 * (power_change - strategy_params['power_change_limit'])
        
        # 6. ENHANCED CONVERGENCE BONUS - reward efficiency + performance
        convergence_bonus = 0
        if cbr <= CBR_RANGE[1] and sinr >= SINR_TARGET:
            # Good performance - now reward power efficiency
            if power_change <= 2 and current_power <= (power_min + power_max) * 0.6:
                convergence_bonus = 8.0  # Strong bonus for efficient stable performance
            elif power_change <= 2:
                convergence_bonus = 4.0  # Moderate bonus for stability
        
        # 7. CRITICAL FIX: Massive penalty for power waste when SINR is excellent
        sinr_excess_penalty = 0
        if sinr >= SINR_TARGET + 5:  # Excellent SINR
            power_waste_ratio = (current_power - power_min) / max(1, power_max - power_min)
            if power_waste_ratio > 0.5:  # Using more than 50% of power range with excellent SINR
                sinr_excess_penalty = -20.0 * (power_waste_ratio - 0.5) ** 2
        
        total_power_reward = (cbr_reward + sinr_reward + power_efficiency + 
                             density_appropriateness + change_penalty + convergence_bonus + sinr_excess_penalty)
        
        # Track performance for analysis
        self.performance_history.append({
            'cbr': cbr,
            'sinr': sinr,
            'power': current_power,
            'density': density_category,
            'strategy': strategy_idx,
            'reward': total_power_reward
        })
        
        # Keep history manageable
        if len(self.performance_history) > 500:
            self.performance_history = self.performance_history[-300:]
        
        return np.clip(total_power_reward, -50, 60)
    
    def update_q_table(self, state_indices, action, reward, next_state_indices):
        """Update PRIMARY power Q-table"""
        current_q = self.q_table[state_indices][action]
        max_next_q = np.max(self.q_table[next_state_indices])
        
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q - current_q)
        self.q_table[state_indices][action] = new_q
        
        return new_q - current_q

# ================== SECONDARY MAC AGENT ==================
class SecondaryMACAgent:
    """SECONDARY MAC Agent - Fine-tunes beacon/MCS under power guidance"""
    
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon
        self.q_table = mac_q_table
        self.state_visit_counts = defaultdict(int)
        self.last_strategy = 1
        self.last_power_action = 0
        
    def get_state_indices(self, cbr, sinr, neighbors, strategy_idx, power_action_idx):
        """SIMPLIFIED state indices with power change category for reduced dimensions"""
        # Get basic state
        cbr_idx = discretize(cbr, CBR_BINS)
        sinr_idx = discretize(sinr, SINR_BINS)
        neighbors_idx = discretize(neighbors, NEIGHBORS_BINS)
        
        # Ensure indices are within bounds
        cbr_idx = min(cbr_idx, CBR_STATES - 1)
        sinr_idx = min(sinr_idx, SINR_STATES - 1)
        neighbors_idx = min(neighbors_idx, NEIGHBORS_STATES - 1)
        strategy_idx = min(strategy_idx, 2)
        
        # Categorize power change instead of using full power action space
        power_change = POWER_ACTIONS[power_action_idx]
        power_change_category = categorize_power_change(power_change)
        
        return (cbr_idx, sinr_idx, neighbors_idx, strategy_idx, power_change_category)
    
    def select_action(self, state_indices, strategy_idx, power_action_idx, current_beacon, current_mcs,
                     current_power, neighbors, antenna_type="OMNIDIRECTIONAL"):
        """SECONDARY MAC action selection guided by power decision"""
        self.state_visit_counts[state_indices] += 1
        self.last_strategy = strategy_idx
        self.last_power_action = power_action_idx
        
        strategy_params = get_strategy_parameters(strategy_idx)
        adaptive_epsilon = self.epsilon * strategy_params['exploration_multiplier'] * 0.7  # Less exploration for secondary
        
        # Understand power action context
        power_change = POWER_ACTIONS[power_action_idx]
        power_direction = "increase" if power_change > 0 else "decrease" if power_change < 0 else "maintain"
        
        if random.random() < adaptive_epsilon:
            # POWER-COORDINATED MAC EXPLORATION
            
            # 1. Coordinate with power action
            if power_direction == "decrease":  # Power is being reduced
                # Can afford to increase beacon/MCS slightly to maintain communication quality
                if neighbors <= 20:  # Not too dense
                    preferred_actions = [i for i, (b, m) in enumerate(MAC_ACTIONS) 
                                       if 0 <= b <= 2 and 0 <= m <= 1]
                else:  # Dense network
                    preferred_actions = [i for i, (b, m) in enumerate(MAC_ACTIONS) 
                                       if -1 <= b <= 1 and -1 <= m <= 1]
                                       
            elif power_direction == "increase":  # Power is being increased
                # Should reduce beacon/MCS to avoid over-congestion
                preferred_actions = [i for i, (b, m) in enumerate(MAC_ACTIONS) 
                                   if b <= 0 and m <= 0]
                                   
            else:  # Power maintained
                # Can make small adaptive adjustments
                preferred_actions = [i for i, (b, m) in enumerate(MAC_ACTIONS) 
                                   if abs(b) <= 1 and abs(m) <= 1]
            
            # 2. Strategy-specific MAC behavior
            if strategy_idx == 0:  # CONSERVATIVE
                conservative_actions = [i for i, (b, m) in enumerate(MAC_ACTIONS) 
                                      if abs(b) <= strategy_params['mac_change_limit'] and abs(m) <= strategy_params['mac_change_limit']]
                preferred_actions = list(set(preferred_actions) & set(conservative_actions)) if preferred_actions else conservative_actions
                
            elif strategy_idx == 2:  # AGGRESSIVE
                # Allow more aggressive MAC adjustments if power is being managed
                if abs(power_change) >= 3:  # Significant power change
                    aggressive_actions = [i for i, (b, m) in enumerate(MAC_ACTIONS) 
                                        if abs(b) <= 2 and abs(m) <= 2]
                    preferred_actions = aggressive_actions
            
            # 3. Density-aware bounds
            density_category = get_neighbor_category(neighbors, antenna_type)
            if density_category in ["HIGH", "VERY_HIGH", "EXTREME"]:
                # Limit beacon increases in high density
                safe_actions = [i for i in (preferred_actions if preferred_actions else list(range(MAC_ACTION_DIM))) 
                              if MAC_ACTIONS[i][0] <= 1]  # Beacon change ≤ 1
                preferred_actions = safe_actions if safe_actions else preferred_actions
            
            if preferred_actions:
                return random.choice(preferred_actions)
            else:
                return 0  # No change fallback
        else:
            return np.argmax(self.q_table[state_indices])
    
    def calculate_mac_reward(self, cbr, sinr, neighbors, strategy_idx, power_action_idx, 
                           current_beacon, current_mcs, next_beacon, next_mcs, 
                           primary_power_reward, antenna_type="OMNIDIRECTIONAL"):
        """SECONDARY MAC reward - coordinated with power decisions"""
        
        strategy_params = get_strategy_parameters(strategy_idx)
        power_change = POWER_ACTIONS[power_action_idx]
        
        # 1. Base reward from coordination with power agent (30% of power reward)
        coordination_reward = primary_power_reward * 0.3
        
        # 2. MAC parameter optimization (considering power context)
        beacon_reward = 0
        mcs_reward = 0
        
        # Optimal beacon based on density and power context
        optimal_beacon_factor = 1.0 - (0.3 * math.log(1 + neighbors) / math.log(1 + 40))
        optimal_beacon = 12.0 * optimal_beacon_factor
        
        if antenna_type.upper() == "SECTORAL":
            optimal_beacon *= 1.1
        
        # Adjust optimal beacon based on power action
        if power_change > 0:  # Power increasing - can reduce beacon
            optimal_beacon *= 0.9
        elif power_change < 0:  # Power decreasing - may need higher beacon
            optimal_beacon *= 1.1
        
        beacon_error = abs(current_beacon - optimal_beacon)
        beacon_reward = -1.5 * (beacon_error / 4.0) ** 2
        
        # MCS optimization
        optimal_mcs_factor = 1.0 - (0.4 * math.log(1 + neighbors) / math.log(1 + 50))
        optimal_mcs = 8.0 * optimal_mcs_factor
        
        if antenna_type.upper() == "SECTORAL":
            optimal_mcs *= 1.2
            
        mcs_error = abs(current_mcs - optimal_mcs)
        mcs_reward = -1.0 * (mcs_error / 3.0) ** 2
        
        # 3. Power-MAC coordination bonus
        coordination_bonus = 0
        beacon_change = abs(next_beacon - current_beacon)
        mcs_change = abs(next_mcs - current_mcs)
        
        if abs(power_change) <= 2:  # Small power change
            if beacon_change <= 1 and mcs_change <= 1:  # Small MAC changes
                coordination_bonus = 3.0
        elif abs(power_change) >= 5:  # Large power change
            if beacon_change <= 2 and mcs_change <= 2:  # Moderate MAC changes
                coordination_bonus = 2.0
        
        # 4. Strategy alignment for MAC
        strategy_alignment = 0
        total_mac_change = beacon_change + mcs_change
        
        if strategy_idx == 0:  # CONSERVATIVE
            if total_mac_change <= strategy_params['mac_change_limit']:
                strategy_alignment = 2.0
            else:
                strategy_alignment = -1.5
        elif strategy_idx == 2:  # AGGRESSIVE
            if neighbors > 30 and total_mac_change >= 1:  # Decisive action in high density
                strategy_alignment = 3.0
        else:  # BALANCED
            if 1 <= total_mac_change <= 3:
                strategy_alignment = 2.0
        
        # 5. Smoothness penalty
        smoothness_penalty = -0.3 * (beacon_change + mcs_change)
        
        total_mac_reward = (coordination_reward + beacon_reward + mcs_reward + 
                           coordination_bonus + strategy_alignment + smoothness_penalty)
        
        return np.clip(total_mac_reward, -25, 30)
    
    def update_q_table(self, state_indices, action, reward, next_state_indices):
        """Update SECONDARY MAC Q-table"""
        current_q = self.q_table[state_indices][action]
        max_next_q = np.max(self.q_table[next_state_indices])
        
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q - current_q)
        self.q_table[state_indices][action] = new_q
        
        return new_q - current_q

# ================== POWER-CENTRIC COORDINATION MANAGER ==================
class PowerCentricCoordinationManager:
    """Manages coordination between Meta-Controller, Primary Power Agent, and Secondary MAC Agent"""
    
    def __init__(self, meta_controller, power_agent, mac_agent):
        self.meta_controller = meta_controller
        self.power_agent = power_agent
        self.mac_agent = mac_agent
        self.experience_buffer = deque(maxlen=BUFFER_SIZE)
        self.coordination_history = []
        self.power_performance_tracker = []
        
    def add_experience(self, experience):
        """Add power-centric coordinated experience to buffer"""
        self.experience_buffer.append(experience)
        
        # Track power-centric coordination metrics
        power_change = POWER_ACTIONS[experience['power_action']]
        mac_beacon_change, mac_mcs_change = MAC_ACTIONS[experience['mac_action']]
        
        coordination_score = self._calculate_coordination_score(
            experience['strategy_idx'], power_change, mac_beacon_change, mac_mcs_change,
            experience['cbr'], experience['sinr'], experience['neighbors']
        )
        
        self.coordination_history.append({
            'timestamp': datetime.now(),
            'strategy': experience['strategy_idx'],
            'power_change': power_change,
            'mac_changes': (mac_beacon_change, mac_mcs_change),
            'coordination_score': coordination_score,
            'cbr': experience['cbr'],
            'sinr': experience['sinr'],
            'neighbors': experience['neighbors']
        })
        
        # Batch update every 32 experiences
        if len(self.experience_buffer) % 32 == 0:
            self.perform_power_centric_update()
    
    def _calculate_coordination_score(self, strategy_idx, power_change, beacon_change, mcs_change, cbr, sinr, neighbors):
        """Calculate how well the agents are coordinating"""
        score = 1.0
        
        # 1. Strategy-Power coordination
        strategy_params = get_strategy_parameters(strategy_idx)
        if abs(power_change) <= strategy_params['power_change_limit']:
            score += 0.3
        else:
            score -= 0.2
        
        # 2. Power-MAC coordination
        if power_change > 0:  # Power increasing
            if beacon_change <= 0 and mcs_change <= 0:  # MAC compensating
                score += 0.4
        elif power_change < 0:  # Power decreasing
            if beacon_change >= 0 or mcs_change >= 0:  # MAC compensating
                score += 0.4
        else:  # Power stable
            if abs(beacon_change) <= 1 and abs(mcs_change) <= 1:  # MAC stable too
                score += 0.3
        
        # 3. Performance outcome coordination
        if CBR_RANGE[0] <= cbr <= CBR_RANGE[1] and sinr >= SINR_TARGET:
            score += 0.5  # Good performance
        elif cbr > 0.6:
            score -= 0.3  # Poor CBR coordination
        
        return np.clip(score, 0, 2.0)
    
    def perform_power_centric_update(self):
        """Perform coordinated updates with power-centric priorities"""
        if len(self.experience_buffer) < 32:
            return
            
        batch = random.sample(self.experience_buffer, 32)
        
        meta_td_errors = []
        power_td_errors = []
        mac_td_errors = []
        
        for exp in batch:
            # Update all agents with power-centric priorities
            meta_td = self.meta_controller.update_q_table(
                exp['meta_state'], exp['meta_action'],
                exp['meta_reward'], exp['next_meta_state']
            )
            meta_td_errors.append(abs(meta_td))
            
            # PRIMARY: Power agent gets highest learning priority
            power_td = self.power_agent.update_q_table(
                exp['power_state'], exp['power_action'],
                exp['power_reward'], exp['next_power_state']
            )
            power_td_errors.append(abs(power_td))
            
            # SECONDARY: MAC agent learns under power guidance
            mac_td = self.mac_agent.update_q_table(
                exp['mac_state'], exp['mac_action'],
                exp['mac_reward'], exp['next_mac_state']
            )
            mac_td_errors.append(abs(mac_td))
        
        # Calculate coordination metrics
        avg_meta_td = np.mean(meta_td_errors)
        avg_power_td = np.mean(power_td_errors)
        avg_mac_td = np.mean(mac_td_errors)
        
        # Power-centric performance tracking
        recent_power_performance = np.mean([h['coordination_score'] for h in self.coordination_history[-20:]])
        
        self.power_performance_tracker.append({
            'timestamp': datetime.now(),
            'meta_td_error': avg_meta_td,
            'power_td_error': avg_power_td,
            'mac_td_error': avg_mac_td,
            'power_centric_coordination': recent_power_performance,
            'power_dominance_ratio': avg_power_td / (avg_power_td + avg_mac_td) if (avg_power_td + avg_mac_td) > 0 else 0.5
        })
        
        if len(self.power_performance_tracker) % 20 == 0:
            logger.info(f"Power-Centric Coordination Update: Meta TD={avg_meta_td:.4f}, "
                       f"POWER TD={avg_power_td:.4f}, MAC TD={avg_mac_td:.4f}, "
                       f"Coordination Score={recent_power_performance:.4f}")
            
            # Log power dominance
            power_dominance = self.power_performance_tracker[-1]['power_dominance_ratio']
            logger.info(f"Power Agent Dominance: {power_dominance:.2%} (target: >60%)")

    def get_coordination_metrics(self):
        """Get latest power-centric coordination metrics"""
        if not self.power_performance_tracker:
            return {}
            
        latest = self.power_performance_tracker[-1]
        recent_coordination = np.mean([h['coordination_score'] for h in self.coordination_history[-50:]])
        
        return {
            'power_centric_coordination': recent_coordination,
            'power_dominance_ratio': latest['power_dominance_ratio'],
            'meta_stability': 1.0 / (1.0 + latest['meta_td_error']),
            'power_learning_rate': latest['power_td_error'],
            'mac_learning_rate': latest['mac_td_error']
        }

# ================== POWER-CENTRIC PERFORMANCE TRACKING ==================
class PowerCentricPerformanceMetrics:
    """Enhanced performance tracking for power-centric hierarchical system"""
    
    def __init__(self):
        self.reset_metrics()
        self.episode_data = []
        
    def reset_metrics(self):
        self.meta_rewards = []
        self.power_rewards = []
        self.mac_rewards = []
        self.joint_rewards = []
        self.cbr_values = []
        self.sinr_values = []
        self.power_values = []
        self.beacon_values = []
        self.mcs_values = []
        self.strategies = []
        self.power_actions = []
        self.mac_actions = []
        self.neighbor_counts = []
        self.coordination_scores = []
        self.power_efficiency_scores = []
        
    def add_step(self, meta_reward, power_reward, mac_reward, joint_reward,
                cbr, sinr, power, beacon, mcs, neighbors, strategy_idx, 
                power_action, mac_action, coordination_score=0):
        """Add step data for power-centric system"""
        self.meta_rewards.append(meta_reward)
        self.power_rewards.append(power_reward)
        self.mac_rewards.append(mac_reward)
        self.joint_rewards.append(joint_reward)
        self.cbr_values.append(cbr)
        self.sinr_values.append(sinr)
        self.power_values.append(power)
        self.beacon_values.append(beacon)
        self.mcs_values.append(mcs)
        self.neighbor_counts.append(neighbors)
        self.strategies.append(strategy_idx)
        self.power_actions.append(power_action)
        self.mac_actions.append(mac_action)
        self.coordination_scores.append(coordination_score)
        
        # Calculate power efficiency score
        density_category = get_neighbor_category(neighbors, ANTENNA_TYPE)
        power_min, power_max = get_density_adaptive_power_range(neighbors, ANTENNA_TYPE)
        power_efficiency = 1.0 - ((power - power_min) / max(1, power_max - power_min))
        self.power_efficiency_scores.append(power_efficiency)
    
    def calculate_episode_metrics(self, episode_num):
        """Calculate comprehensive power-centric hierarchical metrics"""
        if not self.meta_rewards:
            return {}
            
        metrics = {
            'episode': episode_num,
            'timestamp': datetime.now(),
            'total_steps': len(self.meta_rewards),
            
            # Power-centric reward metrics
            'avg_meta_reward': np.mean(self.meta_rewards),
            'avg_power_reward': np.mean(self.power_rewards),
            'avg_mac_reward': np.mean(self.mac_rewards),
            'avg_joint_reward': np.mean(self.joint_rewards),
            'cumulative_joint_reward': sum(self.joint_rewards),
            'power_reward_dominance': np.mean(self.power_rewards) / (np.mean(self.power_rewards) + np.mean(self.mac_rewards)) if (np.mean(self.power_rewards) + np.mean(self.mac_rewards)) > 0 else 0.5,
            
            # Core performance metrics
            'avg_cbr': np.mean(self.cbr_values),
            'cbr_in_range_rate': sum(1 for cbr in self.cbr_values if CBR_RANGE[0] <= cbr <= CBR_RANGE[1]) / len(self.cbr_values),
            'cbr_violation_rate': sum(1 for cbr in self.cbr_values if cbr > 0.6) / len(self.cbr_values),
            'avg_sinr': np.mean(self.sinr_values),
            'sinr_above_target_rate': sum(1 for sinr in self.sinr_values if sinr >= SINR_TARGET) / len(self.sinr_values),
            
            # Power-centric analysis
            'avg_power': np.mean(self.power_values),
            'power_std': np.std(self.power_values),
            'power_efficiency_score': np.mean(self.power_efficiency_scores),
            'low_power_usage_rate': sum(1 for p in self.power_values if p <= 10) / len(self.power_values),
            'high_power_usage_rate': sum(1 for p in self.power_values if p >= 20) / len(self.power_values),
            'power_range_utilization': (max(self.power_values) - min(self.power_values)) / (POWER_MAX - POWER_MIN),
            
            # MAC secondary analysis
            'avg_beacon': np.mean(self.beacon_values),
            'avg_mcs': np.mean(self.mcs_values),
            'beacon_std': np.std(self.beacon_values),
            'mcs_std': np.std(self.mcs_values),
            
            # Strategy analysis
            'conservative_rate': sum(1 for s in self.strategies if s == 0) / len(self.strategies),
            'balanced_rate': sum(1 for s in self.strategies if s == 1) / len(self.strategies),
            'aggressive_rate': sum(1 for s in self.strategies if s == 2) / len(self.strategies),
            'strategy_entropy': entropy(np.bincount(self.strategies, minlength=3)),
            
            # Power action analysis
            'power_increase_rate': sum(1 for a in self.power_actions if POWER_ACTIONS[a] > 0) / len(self.power_actions),
            'power_decrease_rate': sum(1 for a in self.power_actions if POWER_ACTIONS[a] < 0) / len(self.power_actions),
            'power_maintain_rate': sum(1 for a in self.power_actions if POWER_ACTIONS[a] == 0) / len(self.power_actions),
            'avg_power_change': np.mean([abs(POWER_ACTIONS[a]) for a in self.power_actions]),
            
            # Coordination metrics
            'avg_coordination_score': np.mean(self.coordination_scores) if self.coordination_scores else 0,
            'power_action_entropy': entropy(np.bincount(self.power_actions, minlength=POWER_ACTION_DIM)),
            'mac_action_entropy': entropy(np.bincount(self.mac_actions, minlength=MAC_ACTION_DIM)),
            
            # Density-power correlation analysis
            'avg_neighbors': np.mean(self.neighbor_counts),
            'density_power_correlation': np.corrcoef(self.neighbor_counts, self.power_values)[0,1] if len(self.neighbor_counts) > 1 else 0,
            
            # Performance by density category
            'low_density_cbr_performance': np.mean([self.cbr_values[i] for i, n in enumerate(self.neighbor_counts) if get_neighbor_category(n, ANTENNA_TYPE) == "LOW"]) if any(get_neighbor_category(n, ANTENNA_TYPE) == "LOW" for n in self.neighbor_counts) else 0,
            'high_density_cbr_performance': np.mean([self.cbr_values[i] for i, n in enumerate(self.neighbor_counts) if get_neighbor_category(n, ANTENNA_TYPE) == "VERY_HIGH"]) if any(get_neighbor_category(n, ANTENNA_TYPE) == "VERY_HIGH" for n in self.neighbor_counts) else 0,
        }
        
        return metrics
    
    def log_performance(self, episode_num):
        """Log and save power-centric hierarchical performance metrics"""
        metrics = self.calculate_episode_metrics(episode_num)
        if metrics:
            self.episode_data.append(metrics)
            
            if episode_num % PERFORMANCE_LOG_INTERVAL == 0:
                self.save_to_excel()
        return metrics
    
    def save_to_excel(self):
        """Save power-centric performance data to Excel"""
        try:
            with pd.ExcelWriter(PERFORMANCE_LOG_PATH, engine='openpyxl', mode='w') as writer:
                # Episode summary
                if self.episode_data:
                    episode_df = pd.DataFrame(self.episode_data)
                    episode_df.to_excel(writer, sheet_name='Episode_Summary', index=False)
                
                # Power-centric analysis
                if len(self.episode_data) >= 10:
                    recent_data = self.episode_data[-10:]
                    power_centric_analysis = {
                        'Metric': ['Meta Controller Avg Reward', 'PRIMARY Power Agent Avg Reward', 'SECONDARY MAC Agent Avg Reward',
                                  'Joint Avg Reward', 'Power Reward Dominance', 'CBR Performance', 'SINR Performance',
                                  'Power Efficiency Score', 'Low Power Usage Rate', 'High Power Usage Rate',
                                  'Conservative Strategy Usage', 'Balanced Strategy Usage', 'Aggressive Strategy Usage',
                                  'Coordination Score', 'Density-Power Correlation'],
                        'Value': [
                            np.mean([d['avg_meta_reward'] for d in recent_data]),
                            np.mean([d['avg_power_reward'] for d in recent_data]),
                            np.mean([d['avg_mac_reward'] for d in recent_data]),
                            np.mean([d['avg_joint_reward'] for d in recent_data]),
                            np.mean([d['power_reward_dominance'] for d in recent_data]),
                            np.mean([d['cbr_in_range_rate'] for d in recent_data]),
                            np.mean([d['sinr_above_target_rate'] for d in recent_data]),
                            np.mean([d['power_efficiency_score'] for d in recent_data]),
                            np.mean([d['low_power_usage_rate'] for d in recent_data]),
                            np.mean([d['high_power_usage_rate'] for d in recent_data]),
                            np.mean([d['conservative_rate'] for d in recent_data]),
                            np.mean([d['balanced_rate'] for d in recent_data]),
                            np.mean([d['aggressive_rate'] for d in recent_data]),
                            np.mean([d['avg_coordination_score'] for d in recent_data]),
                            np.mean([d['density_power_correlation'] for d in recent_data])
                        ]
                    }
                    analysis_df = pd.DataFrame(power_centric_analysis)
                    analysis_df.to_excel(writer, sheet_name='Power_Centric_Analysis', index=False)
                
                # Power action effectiveness analysis
                power_action_analysis = []
                for episode in self.episode_data:
                    power_action_analysis.append({
                        'episode': episode['episode'],
                        'power_increase_rate': episode['power_increase_rate'],
                        'power_decrease_rate': episode['power_decrease_rate'],
                        'power_maintain_rate': episode['power_maintain_rate'],
                        'avg_power_change': episode['avg_power_change'],
                        'power_efficiency': episode['power_efficiency_score'],
                        'cbr_performance': episode['cbr_in_range_rate'],
                        'sinr_performance': episode['sinr_above_target_rate'],
                        'power_dominance': episode['power_reward_dominance']
                    })
                
                power_df = pd.DataFrame(power_action_analysis)
                power_df.to_excel(writer, sheet_name='Power_Action_Effectiveness', index=False)
                
                # Density-adaptive performance analysis
                density_analysis = []
                for episode in self.episode_data:
                    density_analysis.append({
                        'episode': episode['episode'],
                        'avg_neighbors': episode['avg_neighbors'],
                        'avg_power': episode['avg_power'],
                        'density_power_correlation': episode['density_power_correlation'],
                        'low_density_cbr': episode['low_density_cbr_performance'],
                        'high_density_cbr': episode['high_density_cbr_performance'],
                        'power_range_utilization': episode['power_range_utilization']
                    })
                
                density_df = pd.DataFrame(density_analysis)
                density_df.to_excel(writer, sheet_name='Density_Adaptive_Analysis', index=False)
                
            logger.info(f"Power-centric performance data saved to {PERFORMANCE_LOG_PATH}")
            
        except Exception as e:
            logger.error(f"Error saving to Excel: {e}")

# ================== MAIN POWER-CENTRIC HIERARCHICAL SYSTEM ==================
class PowerCentricHierarchicalSystem:
    """Main power-centric hierarchical Q-learning system"""
    
    def __init__(self, training_mode=True):
        self.training_mode = training_mode
        
        # Initialize all agents with power-centric priorities
        initial_epsilon = EPSILON if training_mode else 0.0
        self.meta_controller = MetaController(epsilon=initial_epsilon)
        self.power_agent = PrimaryPowerAgent(epsilon=initial_epsilon)  # PRIMARY
        self.mac_agent = SecondaryMACAgent(epsilon=initial_epsilon)     # SECONDARY
        
        # Initialize power-centric coordination manager
        self.coordination_manager = PowerCentricCoordinationManager(
            self.meta_controller, self.power_agent, self.mac_agent
        )
        
        # Initialize power-centric performance tracking
        self.performance = PowerCentricPerformanceMetrics()
        self.episode_count = 0
        self.last_save_time = time.time()  # Track time for periodic saves
        
        # Load pre-trained models if available
        self.load_models()
        
        logger.info(f"Power-Centric Hierarchical Q-Learning System initialized in {'TRAINING' if training_mode else 'TESTING'} mode")
        logger.info("POWER-FIRST HIERARCHY: Meta-Controller → PRIMARY Power Agent → SECONDARY MAC Agent")
    
    def process_vehicle(self, veh_id, veh_info):
        """Process vehicle with power-centric hierarchical decision making"""
        try:
            # Check for periodic saves (every 5 minutes)
            self.check_and_save_periodically()
            
            # Extract and validate current state
            cbr = float(veh_info.get("CBR", 0.4))
            sinr = float(veh_info.get("SINR", veh_info.get("SNR", 20)))
            neighbors = int(veh_info.get("neighbors", 10))
            current_power = float(veh_info.get("transmissionPower", 15))
            current_beacon = float(veh_info.get("beaconRate", 10))
            current_mcs = int(veh_info.get("MCS", 5))
            
            # Validate and clamp inputs
            current_power = np.clip(current_power, POWER_MIN, POWER_MAX)
            current_beacon = np.clip(current_beacon, BEACON_MIN, BEACON_MAX)
            current_mcs = np.clip(current_mcs, MCS_MIN, MCS_MAX)
            cbr = np.clip(cbr, 0.0, 1.0)
            sinr = np.clip(sinr, 0, 50)
            neighbors = max(0, neighbors)
            
            # Handle NaN values
            if not np.isfinite(cbr): cbr = 0.4
            if not np.isfinite(sinr): sinr = 15.0
            if not np.isfinite(current_power): current_power = 15.0
            if not np.isfinite(current_beacon): current_beacon = 10.0
            
            # STEP 1: Meta-Controller decides strategy
            meta_state_indices = self.meta_controller.get_state_indices(cbr, sinr, neighbors)
            strategy_idx = self.meta_controller.select_strategy(
                meta_state_indices, cbr, sinr, neighbors, ANTENNA_TYPE
            )
            
            # STEP 2: Get simplified unified state representation
            unified_state = get_unified_state_indices(cbr, sinr, neighbors)
            
            # STEP 3: PRIMARY Power Agent makes main control decision
            power_state_indices = self.power_agent.get_state_indices(unified_state, strategy_idx)
            power_action_idx = self.power_agent.select_action(
                power_state_indices, strategy_idx, current_power, cbr, sinr, neighbors, ANTENNA_TYPE
            )
            
            # STEP 4: SECONDARY MAC Agent fine-tunes under power guidance
            mac_state_indices = self.mac_agent.get_state_indices(cbr, sinr, neighbors, strategy_idx, power_action_idx)
            mac_action_idx = self.mac_agent.select_action(
                mac_state_indices, strategy_idx, power_action_idx, current_beacon, current_mcs,
                current_power, neighbors, ANTENNA_TYPE
            )
            
            # STEP 5: Apply power-centric actions
            power_delta = POWER_ACTIONS[power_action_idx]
            beacon_delta, mcs_delta = MAC_ACTIONS[mac_action_idx]
            
            # Power action has PRIORITY - apply with density-adaptive constraints
            power_min, power_max = get_density_adaptive_power_range(neighbors, ANTENNA_TYPE)
            new_power = np.clip(current_power + power_delta, power_min, power_max)
            
            # MAC actions are SECONDARY - constrained by power decision
            new_beacon = np.clip(current_beacon + beacon_delta, BEACON_MIN, BEACON_MAX)
            new_mcs = np.clip(current_mcs + mcs_delta, MCS_MIN, MCS_MAX)
            
            # STEP 6: Training updates (if in training mode)
            if self.training_mode:
                # Simulate next state (in real system this comes from environment)
                next_cbr = cbr + random.uniform(-0.05, 0.05)
                next_cbr = np.clip(next_cbr, 0, 1)
                next_sinr = sinr + random.uniform(-2, 2)
                next_sinr = np.clip(next_sinr, 0, 50)
                
                # Calculate PRIMARY power reward (most important)
                power_reward = self.power_agent.calculate_power_reward(
                    cbr, sinr, neighbors, strategy_idx, current_power, 
                    new_power, next_cbr, next_sinr, ANTENNA_TYPE
                )
                
                # Calculate SECONDARY MAC reward (coordinated with power)
                mac_reward = self.mac_agent.calculate_mac_reward(
                    cbr, sinr, neighbors, strategy_idx, power_action_idx, current_beacon, current_mcs,
                    new_beacon, new_mcs, power_reward, ANTENNA_TYPE
                )
                
                # Calculate joint reward (power-weighted)
                joint_reward = 0.6 * power_reward + 0.3 * mac_reward + 0.1 * (power_reward + mac_reward) / 2
                
                # Calculate meta reward based on coordination success
                meta_reward = self.meta_controller.calculate_meta_reward(
                    cbr, sinr, neighbors, strategy_idx, joint_reward, ANTENNA_TYPE
                )
                
                # Prepare next states
                next_unified_state = get_unified_state_indices(next_cbr, next_sinr, neighbors)
                next_meta_state = self.meta_controller.get_state_indices(next_cbr, next_sinr, neighbors)
                next_power_state = self.power_agent.get_state_indices(next_unified_state, strategy_idx)
                next_mac_state = self.mac_agent.get_state_indices(next_cbr, next_sinr, neighbors, strategy_idx, power_action_idx)
                
                # Create power-centric coordinated experience
                experience = {
                    'meta_state': meta_state_indices,
                    'meta_action': strategy_idx,
                    'meta_reward': meta_reward,
                    'next_meta_state': next_meta_state,
                    'power_state': power_state_indices,
                    'power_action': power_action_idx,
                    'power_reward': power_reward,
                    'next_power_state': next_power_state,
                    'mac_state': mac_state_indices,
                    'mac_action': mac_action_idx,
                    'mac_reward': mac_reward,
                    'next_mac_state': next_mac_state,
                    'joint_reward': joint_reward,
                    'strategy_idx': strategy_idx,
                    'cbr': cbr,
                    'sinr': sinr,
                    'neighbors': neighbors
                }
                
                # Add to power-centric coordination manager
                self.coordination_manager.add_experience(experience)
                
                # Update performance metrics with power-centric focus
                coordination_metrics = self.coordination_manager.get_coordination_metrics()
                coordination_score = coordination_metrics.get('power_centric_coordination', 0)
                
                self.performance.add_step(
                    meta_reward, power_reward, mac_reward, joint_reward,
                    cbr, sinr, new_power, new_beacon, new_mcs, neighbors, 
                    strategy_idx, power_action_idx, mac_action_idx, coordination_score
                )
                
                # ENHANCED: Faster epsilon decay based on power efficiency
                power_efficiency_current = 1.0 - ((new_power - power_min) / max(1, power_max - power_min))
                
                if power_efficiency_current > 0.7:  # High efficiency - normal decay
                    power_decay_factor = EPSILON_DECAY * 0.999
                elif power_efficiency_current < 0.3:  # Low efficiency - faster decay to force learning
                    power_decay_factor = EPSILON_DECAY * 1.002
                else:  # Medium efficiency - slightly faster decay
                    power_decay_factor = EPSILON_DECAY * 1.001
                
                # Decay exploration rates with efficiency awareness
                self.meta_controller.epsilon = max(MIN_EPSILON, self.meta_controller.epsilon * EPSILON_DECAY)
                self.power_agent.epsilon = max(MIN_EPSILON, self.power_agent.epsilon * power_decay_factor)
                self.mac_agent.epsilon = max(MIN_EPSILON, self.mac_agent.epsilon * (EPSILON_DECAY * 1.001))
            
            # STEP 7: Enhanced logging for power-centric decisions
            if veh_id.endswith('0') or random.random() < 0.01:  # Log sample vehicles
                strategy_name = STRATEGIES[strategy_idx]
                density_cat = get_neighbor_category(neighbors, ANTENNA_TYPE)
                
                logger.info(f"Vehicle {veh_id} [{ANTENNA_TYPE}][{density_cat}][{strategy_name}]: "
                           f"CBR={cbr:.3f}, SINR={sinr:.1f}dB, Neighbors={neighbors}")
                logger.info(f"  Power Range: {power_min}-{power_max} dBm (density-adaptive)")
                logger.info(f"  PRIMARY Power: {current_power:.0f}->{new_power:.0f}dBm (Δ{power_delta:+d})")
                logger.info(f"  SECONDARY MAC: Beacon {current_beacon:.0f}->{new_beacon:.0f}Hz, MCS {current_mcs}->{new_mcs}")
                
                if self.training_mode and 'joint_reward' in locals():
                    logger.info(f"  Power-Centric Rewards: Power={power_reward:.2f} (PRIMARY), "
                               f"MAC={mac_reward:.2f} (SECONDARY), Joint={joint_reward:.2f}")
                    
                    # Log power efficiency
                    power_efficiency = 1.0 - ((new_power - power_min) / max(1, power_max - power_min))
                    logger.info(f"  Power Efficiency: {power_efficiency:.2%}")
            
            # Calculate power efficiency for all vehicles
            power_efficiency = 1.0 - ((new_power - power_min) / max(1, power_max - power_min))
            
            # Return power-centric response
            return {
                "transmissionPower": int(new_power),
                "beaconRate": int(new_beacon),
                "MCS": int(new_mcs),
                # Additional metadata for analysis
                "strategy": STRATEGIES[strategy_idx],
                "density_category": get_neighbor_category(neighbors, ANTENNA_TYPE),
                "power_range": f"{power_min}-{power_max}",
                "power_change": power_delta,
                "power_efficiency": power_efficiency
            }
            
        except Exception as e:
            logger.error(f"Error processing vehicle {veh_id}: {e}")
            # Return safe defaults with minimal power
            return {
                "transmissionPower": 10,  # Low power default
                "beaconRate": 8,          # Conservative beacon rate
                "MCS": 3,                 # Conservative MCS
                "strategy": "BALANCED",
                "density_category": "MEDIUM"
            }
    
    def end_episode(self):
        """End episode and save metrics for power-centric hierarchical system"""
        if self.training_mode:
            self.episode_count += 1
            metrics = self.performance.log_performance(self.episode_count)
            
            if metrics:
                logger.info(f"Episode {self.episode_count}: "
                           f"Joint reward={metrics['avg_joint_reward']:.3f}, "
                           f"Power dominance={metrics['power_reward_dominance']:.2%}, "
                           f"CBR in range={metrics['cbr_in_range_rate']:.2%}")
                logger.info(f"  Power Performance: Efficiency={metrics['power_efficiency_score']:.2%}, "
                           f"Low power usage={metrics['low_power_usage_rate']:.2%}")
                logger.info(f"  Strategy usage: Conservative={metrics['conservative_rate']:.2%}, "
                           f"Balanced={metrics['balanced_rate']:.2%}, "
                           f"Aggressive={metrics['aggressive_rate']:.2%}")
                
                # Power-specific insights
                coordination_metrics = self.coordination_manager.get_coordination_metrics()
                if coordination_metrics:
                    logger.info(f"  Power-Centric Coordination: {coordination_metrics['power_centric_coordination']:.3f}, "
                               f"Power Dominance: {coordination_metrics['power_dominance_ratio']:.2%}")
            
            if self.episode_count % MODEL_SAVE_INTERVAL == 0:
                self.save_models()
                logger.info(f"PERIODIC MODEL SAVE: Episode {self.episode_count}")
            
            self.performance.reset_metrics()
            return metrics
        return {}
    
    def save_models(self):
        """Save all Q-tables for power-centric system"""
        try:
            np.save(META_MODEL_PATH, self.meta_controller.q_table)
            np.save(POWER_MODEL_PATH, self.power_agent.q_table)
            np.save(MAC_MODEL_PATH, self.mac_agent.q_table)
            logger.info(f"MODELS SAVED: {META_MODEL_PATH}, {POWER_MODEL_PATH}, {MAC_MODEL_PATH}")
            self.last_save_time = time.time()
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def check_and_save_periodically(self):
        """Check if it's time to save models (every 5 minutes during training)"""
        if self.training_mode:
            current_time = time.time()
            if current_time - self.last_save_time > 300:  # 5 minutes
                self.save_models()
                logger.info(f"TIME-BASED MODEL SAVE: {datetime.now().strftime('%H:%M:%S')}")
    
    def load_models(self):
        """Load pre-trained Q-tables for power-centric system"""
        try:
            if os.path.exists(META_MODEL_PATH):
                loaded_meta = np.load(META_MODEL_PATH)
                if loaded_meta.shape == self.meta_controller.q_table.shape:
                    self.meta_controller.q_table = loaded_meta
                    logger.info(f"Loaded Meta Controller model from {META_MODEL_PATH}")
            
            if os.path.exists(POWER_MODEL_PATH):
                loaded_power = np.load(POWER_MODEL_PATH)
                if loaded_power.shape == self.power_agent.q_table.shape:
                    self.power_agent.q_table = loaded_power
                    logger.info(f"Loaded PRIMARY Power Agent model from {POWER_MODEL_PATH}")
            
            if os.path.exists(MAC_MODEL_PATH):
                loaded_mac = np.load(MAC_MODEL_PATH)
                if loaded_mac.shape == self.mac_agent.q_table.shape:
                    self.mac_agent.q_table = loaded_mac
                    logger.info(f"Loaded SECONDARY MAC Agent model from {MAC_MODEL_PATH}")
                    
        except Exception as e:
            logger.error(f"Error loading models: {e}")

# ================== POWER-CENTRIC HIERARCHICAL SERVER ==================
class PowerCentricHierarchicalServer:
    """Enhanced server for power-centric hierarchical multi-agent system"""
    
    def __init__(self, host, port, training_mode=True):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(5)
        self.power_centric_system = PowerCentricHierarchicalSystem(training_mode=training_mode)
        self.training_mode = training_mode
        self.running = True
        
        mode_str = "TRAINING" if training_mode else "TESTING"
        logger.info(f"Power-Centric Hierarchical Multi-Agent RL Server started in {mode_str} mode on {host}:{port}")

    def receive_message_with_header(self, conn):
        """Receive message with header (kept from original)"""
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
        """Send message with header (kept from original)"""
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
        """Enhanced client handler for power-centric hierarchical system"""
        logger.info(f"Client connected from {addr}")
        
        try:
            while self.running:
                message_str = self.receive_message_with_header(conn)
                if not message_str:
                    break
                
                try:
                    batch_data = json.loads(message_str)
                    logger.info(f"Processing {len(batch_data)} vehicles with POWER-CENTRIC hierarchical system")
                    
                    responses = {}
                    batch_cbr_values = []
                    batch_power_values = []
                    batch_strategies = []
                    batch_density_categories = []
                    batch_power_efficiencies = []
                    
                    for veh_id, veh_info in batch_data.items():
                        response = self.power_centric_system.process_vehicle(veh_id, veh_info)
                        if response:
                            responses[veh_id] = response
                            
                            # Collect power-centric batch statistics
                            if "CBR" in veh_info:
                                batch_cbr_values.append(float(veh_info["CBR"]))
                            if "transmissionPower" in response:
                                batch_power_values.append(int(response["transmissionPower"]))
                            if "strategy" in response:
                                batch_strategies.append(response["strategy"])
                            if "density_category" in response:
                                batch_density_categories.append(response["density_category"])
                            if "power_efficiency" in response:
                                batch_power_efficiencies.append(float(response["power_efficiency"]))
                    
                    # Log power-centric batch statistics
                    if batch_cbr_values and batch_power_values:
                        avg_cbr = np.mean(batch_cbr_values)
                        avg_power = np.mean(batch_power_values)
                        max_cbr = max(batch_cbr_values)
                        min_power = min(batch_power_values)
                        max_power = max(batch_power_values)
                        
                        # Power efficiency analysis
                        avg_power_efficiency = np.mean(batch_power_efficiencies) if batch_power_efficiencies else 0
                        
                        # Strategy distribution
                        strategy_counts = {s: batch_strategies.count(s) for s in STRATEGIES}
                        dominant_strategy = max(strategy_counts, key=strategy_counts.get) if strategy_counts else "UNKNOWN"
                        
                        # Density distribution
                        density_counts = {}
                        for density in ["VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH", "EXTREME"]:
                            density_counts[density] = batch_density_categories.count(density)
                        dominant_density = max(density_counts, key=density_counts.get) if density_counts else "UNKNOWN"
                        
                        logger.info(f"POWER-CENTRIC BATCH ANALYSIS:")
                        logger.info(f"   CBR: Avg={avg_cbr:.3f}, Max={max_cbr:.3f} | Power: Avg={avg_power:.1f}dBm, Range=[{min_power}, {max_power}]")
                        logger.info(f"   Power Efficiency: {avg_power_efficiency:.2%} | Dominant Strategy: {dominant_strategy} | Dominant Density: {dominant_density}")
                        
                        # Critical alerts for power-centric system
                        if avg_cbr > 0.6:
                            high_cbr_vehicles = sum(1 for cbr in batch_cbr_values if cbr > 0.6)
                            logger.warning(f"HIGH CBR CRISIS: {high_cbr_vehicles}/{len(batch_cbr_values)} vehicles above 0.6 CBR")
                            logger.warning(f"   Recommended: Increase CONSERVATIVE strategy usage, reduce power globally")
                        
                        if avg_power > 20 and dominant_density in ["HIGH", "VERY_HIGH", "EXTREME"]:
                            logger.warning(f"POWER TOO HIGH for {dominant_density} density: Avg={avg_power:.1f}dBm")
                        
                        if avg_power_efficiency < 0.3:
                            logger.warning(f"LOW POWER EFFICIENCY: {avg_power_efficiency:.2%} - Power usage not optimal for density")
                        
                        # Positive feedback
                        if avg_cbr <= CBR_RANGE[1] and avg_power_efficiency > 0.6:
                            logger.info(f"EXCELLENT POWER-CENTRIC PERFORMANCE: CBR in range + high efficiency")
                    
                    # Send response
                    response_dict = {"vehicles": responses}
                    response_str = json.dumps(response_dict)
                    
                    if self.send_message_with_header(conn, response_str):
                        logger.debug(f"Sent power-centric response to {addr}: {len(responses)} vehicles")
                    else:
                        break
                    
                    # Episode management for training with power-centric focus
                    if self.training_mode and len(self.power_centric_system.performance.power_rewards) >= 100:
                        metrics = self.power_centric_system.end_episode()
                        if metrics:
                            logger.info(f"POWER-CENTRIC Training Episode {self.power_centric_system.episode_count} completed")
                            logger.info(f"   Power Dominance: {metrics.get('power_reward_dominance', 0):.2%} | "
                                       f"Power Efficiency: {metrics.get('power_efficiency_score', 0):.2%}")
                
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
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
        """Start power-centric hierarchical server with proper signal handling"""
        try:
            logger.info("Power-Centric Hierarchical Multi-Agent RL Server listening for connections...")
            # Set socket timeout to make it responsive to interrupts
            self.server.settimeout(1.0)
            
            while self.running:
                try:
                    conn, addr = self.server.accept()
                    client_thread = threading.Thread(
                        target=self.handle_client, 
                        args=(conn, addr),
                        daemon=True
                    )
                    client_thread.start()
                except socket.timeout:
                    # Timeout allows checking self.running periodically
                    continue
                except Exception as e:
                    if self.running:
                        logger.error(f"Error accepting connection: {e}")
        except Exception as e:
            if self.running:
                logger.error(f"Server error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop power-centric hierarchical server and save final results"""
        logger.info("Stopping Power-Centric Hierarchical Multi-Agent RL server...")
        self.running = False
        
        try:
            self.server.close()
        except:
            pass
        
        if self.training_mode:
            self.power_centric_system.save_models()
            self.power_centric_system.performance.save_to_excel()
            logger.info("Final power-centric models and performance data saved")
            
            # Final power-centric analysis
            final_metrics = self.power_centric_system.coordination_manager.get_coordination_metrics()
            if final_metrics:
                logger.info(f"FINAL POWER-CENTRIC ANALYSIS:")
                logger.info(f"   Power Dominance Ratio: {final_metrics.get('power_dominance_ratio', 0):.2%}")
                logger.info(f"   Power-Centric Coordination: {final_metrics.get('power_centric_coordination', 0):.3f}")
                logger.info(f"   Meta Stability: {final_metrics.get('meta_stability', 0):.3f}")
        
        logger.info("Power-Centric Hierarchical Multi-Agent RL server stopped")

# ================== SIGNAL HANDLERS FOR GRACEFUL SHUTDOWN ==================
def signal_handler(signum, frame, rl_server=None):
    """Handle interrupt signals gracefully"""
    logger.info(f"\nReceived signal {signum}. Gracefully shutting down...")
    if rl_server:
        rl_server.stop()
    else:
        sys.exit(0)

# ================== MAIN EXECUTION ==================
def main():
    """Main execution function for power-centric hierarchical system"""
    if OPERATION_MODE.upper() not in ["TRAINING", "TESTING"]:
        print(f"ERROR: Invalid OPERATION_MODE '{OPERATION_MODE}'. Must be 'TRAINING' or 'TESTING'")
        sys.exit(1)
    
    if ANTENNA_TYPE.upper() not in ["SECTORAL", "OMNIDIRECTIONAL"]:
        print(f"ERROR: Invalid ANTENNA_TYPE '{ANTENNA_TYPE}'. Must be 'SECTORAL' or 'OMNIDIRECTIONAL'")
        sys.exit(1)
    
    training_mode = (OPERATION_MODE.upper() == "TRAINING")
    
    print("="*120)
    print(" POWER-CENTRIC HIERARCHICAL MULTI-AGENT Q-LEARNING VANET OPTIMIZATION SYSTEM")
    print("="*120)
    print(f"Host: {HOST}:{PORT}")
    print(f"Mode: {OPERATION_MODE.upper()}")
    print(f"Antenna Type: {ANTENNA_TYPE.upper()}")
    print("="*60)
    print("POWER-FIRST HIERARCHICAL ARCHITECTURE:")
    print("  Meta-Controller: High-level strategy decisions")
    print("     Conservative: Minimal power, small changes, CBR-focused")
    print("     Balanced: Adaptive power, moderate changes, balanced objectives")
    print("     Aggressive: Performance-first power, large changes, SINR-focused")
    print("  PRIMARY Power Agent: Main power transmission control (PRIORITY #1)")
    print("     Density-adaptive power ranges (1-5 dBm in extreme density)")
    print("     Strategy-guided power exploration")
    print("     CBR-first optimization with SINR balance")
    print("  SECONDARY MAC Agent: Beacon/MCS fine-tuning under power guidance")
    print("     Power-coordinated MAC adjustments")
    print("     Strategy and power-action aware decisions")
    print("     Secondary optimization layer")
    print("="*60)
    print("POWER-CENTRIC IMPROVEMENTS:")
    print("  POWER CONTROLS NETWORK CONGESTION (CBR) - Primary objective")
    print("  SECTORAL ANTENNA OPTIMIZATION: Neighbor density split across front/rear")
    print("  Density-adaptive power exploration ranges")
    print("  REALISTIC VANET density categorization and power ranges:")
    
    # Display 6-level density categories without icons
    for density in ["VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH", "EXTREME"]:
        if ANTENNA_TYPE.upper() == "SECTORAL":
            if density == "VERY_LOW":
                neighbors = "≤3 total (≤2 effective)"
                power_range = "15-30"
                scenario = "Rural/Highway"
            elif density == "LOW":
                neighbors = "≤6 total (≤4 effective)" 
                power_range = "12-25"
                scenario = "Sparse Urban"
            elif density == "MEDIUM":
                neighbors = "≤12 total (≤8 effective)"
                power_range = "8-18"
                scenario = "Normal Urban"
            elif density == "HIGH":
                neighbors = "≤18 total (≤12 effective)"
                power_range = "5-12"
                scenario = "Dense Urban"
            elif density == "VERY_HIGH":
                neighbors = "≤25 total (≤18 effective)"
                power_range = "3-8"
                scenario = "Traffic Jam"
            else:
                neighbors = ">25 total (>18 effective)"
                power_range = "1-5"
                scenario = "Extreme Congestion"
        else:
            if density == "VERY_LOW":
                neighbors = "≤2 total"
                power_range = "15-30"
                scenario = "Rural/Highway"
            elif density == "LOW":
                neighbors = "≤5 total"
                power_range = "12-25" 
                scenario = "Sparse Urban"
            elif density == "MEDIUM":
                neighbors = "≤10 total"
                power_range = "8-18"
                scenario = "Normal Urban"
            elif density == "HIGH":
                neighbors = "≤15 total"
                power_range = "5-12"
                scenario = "Dense Urban"
            elif density == "VERY_HIGH":
                neighbors = "≤22 total"
                power_range = "3-8"
                scenario = "Traffic Jam"
            else:
                neighbors = ">22 total"
                power_range = "1-5"
                scenario = "Extreme Congestion"
        print(f"     {density:>9}: {neighbors:<22} → {power_range:>5} dBm ({scenario})")
    
    print("  Power-coordinated MAC fine-tuning (no conflicts)")
    print("  Strategy-guided power behavior (Conservative→Low, Aggressive→High)")
    print("  Power efficiency tracking and optimization")
    print("  Power dominance ratio monitoring (target: >60%)")
    print("="*60)
    print("PERFORMANCE TARGETS:")
    print(f"  CBR Target: {CBR_TARGET} (Primary - controlled by power)")
    print(f"  SINR Target: {SINR_TARGET} dB (Secondary - fine-tuned by MAC)")
    print(f"  CBR Acceptable Range: {CBR_RANGE[0]} - {CBR_RANGE[1]}")
    print(f"  Power Efficiency Target: >60% (within density-appropriate ranges)")
    print("="*60)
    print("Q-TABLE DIMENSIONS (Power-Centric):")
    print(f"  Meta-Controller: {meta_q_table.shape}")
    print(f"  PRIMARY Power Agent: {power_q_table.shape}")
    print(f"  SECONDARY MAC Agent: {mac_q_table.shape}")
    print(f"  Total Parameters: {meta_q_table.size + power_q_table.size + mac_q_table.size:,}")
    print(f"  Power Agent Parameters: {power_q_table.size:,} ({power_q_table.size/(meta_q_table.size + power_q_table.size + mac_q_table.size)*100:.1f}%)")
    print("="*60)
    print("POWER-CENTRIC COORDINATION FEATURES:")
    print("  • Power agent has PRIORITY in decision making")
    print("  • MAC agent coordinates with power decisions")
    print("  • Density-adaptive power exploration (intelligent bounds)")
    print("  • Power efficiency scoring and optimization")
    print("  • CBR-power correlation monitoring")
    print("  • Strategy-power alignment rewards")
    print("  • Power dominance ratio tracking")
    
    if training_mode:
        print("="*60)
        print("POWER-CENTRIC TRAINING PARAMETERS:")
        print(f"  Learning Rate: {LEARNING_RATE}")
        print(f"  Discount Factor: {DISCOUNT_FACTOR}")
        print(f"  Initial Epsilon: {EPSILON}")
        print(f"  Epsilon Decay: {EPSILON_DECAY}")
        print(f"  Min Epsilon: {MIN_EPSILON}")
        print("  Power Agent: Slower epsilon decay (more exploration)")
        print("  MAC Agent: Faster epsilon decay (follows power decisions)")
        print(f"  Model Save Interval: {MODEL_SAVE_INTERVAL} episodes")
        print(f"  Performance Log Interval: {PERFORMANCE_LOG_INTERVAL} episodes")
        print("  Press Ctrl+C to stop and save models")
    else:
        print("="*60)
        print("POWER-CENTRIC TESTING MODE:")
        print(f"  Using pre-trained power-centric models:")
        print(f"    Meta: {META_MODEL_PATH}")
        print(f"    PRIMARY Power: {POWER_MODEL_PATH}")
        print(f"    SECONDARY MAC: {MAC_MODEL_PATH}")
    
    print("="*60)
    print("EXPECTED POWER-CENTRIC BENEFITS:")
    print("  1. DIRECT CBR CONTROL: Power directly controls network congestion")
    print("  2. INTELLIGENT POWER RANGES: Density-adaptive exploration prevents connectivity issues")
    print("  3. COORDINATED DECISIONS: MAC fine-tunes under power guidance (no conflicts)")
    print("  4. EFFICIENCY OPTIMIZATION: Power usage optimized for density conditions")
    print("  5. FASTER CONVERGENCE: Clear priority hierarchy reduces training complexity")
    print("  6. STRATEGY ALIGNMENT: Power behavior adapts to network strategy needs")
    print("="*120)
    
    # Initialize and start power-centric hierarchical server
    rl_server = PowerCentricHierarchicalServer(HOST, PORT, training_mode=training_mode)
    
    # Set up signal handlers for graceful shutdown
    def signal_handler_with_server(signum, frame):
        signal_handler(signum, frame, rl_server)
    
    signal.signal(signal.SIGINT, signal_handler_with_server)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler_with_server)  # Termination
    
    try:
        rl_server.start()
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt")
        signal_handler_with_server(signal.SIGINT, None)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        rl_server.stop()
    finally:
        logger.info("Power-Centric System shutdown complete")

if __name__ == "__main__":
    main()
