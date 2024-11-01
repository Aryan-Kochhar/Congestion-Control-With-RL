# Traffic Congestion Control using Vanet and SUMO

A SUMO-based traffic simulation that combines reinforcement learning and real-time traffic detection to optimize traffic flow, with special handling for emergency vehicles. The system adapts signal timings based on current traffic conditions while ensuring emergency vehicles get priority access through intersections.

## Why This Project?

Traditional fixed-time traffic signals often struggle with varying traffic patterns and emergency response times. This project tackles both issues by:
- Using real-time traffic data to adjust signal timings
- Automatically detecting and prioritizing emergency vehicles
- Learning from past traffic patterns to improve future decisions
- Preventing congestion before it builds up

## Getting Started

### Quick Setup

- Installing:-
- SUMO
- Python 3.7 or newer
- TraCI for Python

📦 Structure :)
├── 📜 complete.py            # Main simulation logic and RL implementation
├── 📜 routes.rou.xml         # Vehicle and emergency routes definition
├── 📜 simulation.sumocfg     # SUMO configuration and parameters
├── 📜 grid_with_tls.net.xml  # Road network and traffic light layout
└── 📜 gui-settings.xml       # Visualization and GUI settings

Key Features are: 
- Smart Traffic Control: Signals adjust based on actual traffic, not fixed timings
- Emergency Response: Priority handling for emergency vehicles
- Self-Learning: System improves over time using reinforcement learning
- Congestion Prevention: Proactive traffic management
- Performance Tracking: Real-time statistics and monitoring

**How It Works**
The system uses three main components:
1. Traffic Detection
  Monitors real-time metrics like:
      Queue lengths at intersections
      Waiting times for vehicles
      Emergency vehicle presence
      Traffic density in each direction

2. Reinforcement Learning
  The RL agent learns optimal signal timing by:
      Observing traffic states
      Trying different signal timing patterns
      Getting rewards for good traffic flow
      Adapting to recurring patterns

3. Emergency Vehicle Priority
  When an emergency vehicle is detected:
      Traffic signals automatically adjust to give right of way
      Other vehicles are guided to clear the path
      System maintains safety while prioritizing emergency response

Issues that i have came across: 
Signal Flickering: If you notice rapid signal changes, adjust the minimum phase duration by increasing it a bit.
Emergency Vehicle Detection: Sometimes emergency vehicles might get stuck. 


# Reinforcement Learning Implementation

## Overview
The traffic management system uses Q-learning, a model-free reinforcement learning algorithm, to optimize traffic signal timings. The system learns from experience to minimize waiting times and maximize traffic flow while ensuring emergency vehicle priority.

## State Space
The state representation captures the traffic situation at each intersection:

```python
state = (
    ns_queue,        # Number of vehicles waiting in NS direction (0-10)
    ns_wait_time,    # Normalized waiting time for NS direction (0-10)
    ns_vehicles,     # Total vehicles in NS direction (0-10)
    ns_emergency,    # Emergency vehicle present in NS (0/1)
    ew_queue,        # Number of vehicles waiting in EW direction (0-10)
    ew_wait_time,    # Normalized waiting time for EW direction (0-10)
    ew_vehicles,     # Total vehicles in EW direction (0-10)
    ew_emergency,    # Emergency vehicle present in EW (0/1)
    current_phase    # Current traffic light phase
)

** ## Reward Function**
Reward Components:
Waiting Time Penalty: -sum(vehicle_wait_times) / 100
Queue Length Penalty: -number_of_stopped_vehicles
Throughput Reward: +vehicles_passed_through
Emergency Bonus: +100 for successfully handling emergency vehicles

** ## Q-Learning parameters**
alpha = 0.1   # Learning rate: How much to update Q-values
gamma = 0.95  # Discount factor: Importance of future rewards
epsilon = 0.1 # Exploration rate: Chance of trying new actions

** ## Action Selection**
Uses ε-greedy policy:
With probability ε: Explore (random action)
With probability 1-ε: Exploit (best known action)
