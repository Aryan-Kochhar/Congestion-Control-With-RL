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

## Structure 
├── complete.py           # Main simulation logic
├── routes.rou.xml        # Vehicle routes
├── simulation.sumocfg    # SUMO config
├── grid_with_tls.net.xml # Road network
└── gui-settings.xml      # Visualization settings

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
