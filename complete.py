import os
import sys
import traci
import random
import math
from collections import defaultdict

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# Constants
MAX_SPEED = 13.89  # 50 km/h
EMERGENCY_SPEED = 16.67  # 60 km/h
SIMULATION_STEPS = 36000  # 2 hours
HEAVY_TRAFFIC_THRESHOLD = 8

class ReinforcementLearning:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.last_state = {}
        self.last_action = {}
        self.last_action_time = defaultdict(int)
        self.num_actions = 4  # Number of possible actions
        
    def get_state(self, tl_id):
        """Enhanced state space with more traffic information"""
        try:
            lanes = traci.trafficlight.getControlledLanes(tl_id)
            
            # Separate lanes by direction
            ns_lanes = [lane for lane in lanes if 'A' in lane or 'B' in lane]
            ew_lanes = [lane for lane in lanes if not ('A' in lane or 'B' in lane)]
            
            # Get directional traffic data
            ns_state = {
                'queue': sum(traci.lane.getLastStepHaltingNumber(lane) for lane in ns_lanes),
                'waiting_time': max(traci.lane.getWaitingTime(lane) for lane in ns_lanes),
                'vehicles': sum(traci.lane.getLastStepVehicleNumber(lane) for lane in ns_lanes),
                'emergency': any(any(veh.startswith('emergency') for veh in traci.lane.getLastStepVehicleIDs(lane)) 
                            for lane in ns_lanes)
            }
            
            ew_state = {
                'queue': sum(traci.lane.getLastStepHaltingNumber(lane) for lane in ew_lanes),
                'waiting_time': max(traci.lane.getWaitingTime(lane) for lane in ew_lanes),
                'vehicles': sum(traci.lane.getLastStepVehicleNumber(lane) for lane in ew_lanes),
                'emergency': any(any(veh.startswith('emergency') for veh in traci.lane.getLastStepVehicleIDs(lane)) 
                            for lane in ew_lanes)
            }
            
            # Create state tuple (normalized values)
            state = (
                min(10, ns_state['queue']),
                min(10, int(ns_state['waiting_time']/10)),
                min(10, ns_state['vehicles']),
                1 if ns_state['emergency'] else 0,
                min(10, ew_state['queue']),
                min(10, int(ew_state['waiting_time']/10)),
                min(10, ew_state['vehicles']),
                1 if ew_state['emergency'] else 0,
                traci.trafficlight.getPhase(tl_id)
            )
            
            return state
        except:
            return None
            
    def calculate_reward(self, tl_id):
        """Enhanced reward calculation with multiple factors"""
        try:
            lanes = traci.trafficlight.getControlledLanes(tl_id)
            ns_lanes = [lane for lane in lanes if 'A' in lane or 'B' in lane]
            ew_lanes = [lane for lane in lanes if not ('A' in lane or 'B' in lane)]
            
            # Calculate directional rewards
            ns_reward = {
                'waiting': -sum(traci.lane.getWaitingTime(lane) for lane in ns_lanes) / 100,
                'queue': -sum(traci.lane.getLastStepHaltingNumber(lane) for lane in ns_lanes),
                'throughput': sum(traci.lane.getLastStepVehicleNumber(lane) for lane in ns_lanes),
                'emergency': -50 if any(any(veh.startswith('emergency') for veh in traci.lane.getLastStepVehicleIDs(lane)) 
                                    for lane in ns_lanes) else 0
            }
            
            ew_reward = {
                'waiting': -sum(traci.lane.getWaitingTime(lane) for lane in ew_lanes) / 100,
                'queue': -sum(traci.lane.getLastStepHaltingNumber(lane) for lane in ew_lanes),
                'throughput': sum(traci.lane.getLastStepVehicleNumber(lane) for lane in ew_lanes),
                'emergency': -50 if any(any(veh.startswith('emergency') for veh in traci.lane.getLastStepVehicleIDs(lane)) 
                                    for lane in ew_lanes) else 0
            }
            
            # Weight factors
            weights = {
                'waiting': 0.4,    # Highest priority to reduce waiting time
                'queue': 0.3,      # High priority to reduce queues
                'throughput': 0.2, # Medium priority for throughput
                'emergency': 0.1   # Base priority for emergency vehicles
            }
            
            # Calculate total reward
            ns_total = sum(weights[k] * v for k, v in ns_reward.items())
            ew_total = sum(weights[k] * v for k, v in ew_reward.items())
            
            # Current phase affects which direction's reward we prioritize
            current_phase = traci.trafficlight.getPhase(tl_id)
            if current_phase in [0, 1]:  # NS phase
                total_reward = ns_total
            else:  # EW phase
                total_reward = ew_total
            
            # Add emergency vehicle bonus
            if ns_reward['emergency'] or ew_reward['emergency']:
                total_reward += 100  # High bonus for handling emergency vehicles
            
            return max(-10, min(10, total_reward))  # Clip reward
            
        except:
            return 0.0
    def choose_action(self, state, tl_id):
        """Choose action using epsilon-greedy policy with minimum phase duration"""
        if state is None:
            return 0
            
        # Get current time
        current_time = traci.simulation.getTime()
        last_change = self.last_action_time.get(tl_id, 0)
        min_phase_duration = 30  # Minimum phase duration in seconds
        
        # If minimum phase duration hasn't elapsed, keep current phase
        if current_time - last_change < min_phase_duration:
            return traci.trafficlight.getPhase(tl_id)
        
        if random.random() < self.epsilon:
            # Exploration: choose random action
            action = random.randint(0, self.num_actions - 1)
        else:
            # Exploitation: choose best action
            state_key = str(state)
            if not self.q_table[state_key]:
                # If state is new, initialize Q-values for all actions to 0
                for i in range(self.num_actions):
                    self.q_table[state_key][i] = 0.0
            
            # Choose action with maximum Q-value
            max_q = max(self.q_table[state_key].values())
            actions_with_max_q = [a for a, q in self.q_table[state_key].items() 
                                if q == max_q]
            action = random.choice(actions_with_max_q)
        
        self.last_state[tl_id] = state
        self.last_action[tl_id] = action
        self.last_action_time[tl_id] = current_time
        return action

    def learn(self, state, action, reward, next_state):
        """Update Q-value using Q-learning"""
        if state is None or next_state is None:
            return
            
        # Convert states to strings for dictionary keys
        state_key = str(state)
        next_state_key = str(next_state)
        
        # Initialize Q-values for new states
        if not self.q_table[state_key]:
            for i in range(self.num_actions):
                self.q_table[state_key][i] = 0.0
        if not self.q_table[next_state_key]:
            for i in range(self.num_actions):
                self.q_table[next_state_key][i] = 0.0
        
        # Get max Q-value for next state
        next_max = max(self.q_table[next_state_key].values())
        
        # Update Q-value
        current_q = self.q_table[state_key][action]
        new_q = current_q + self.alpha * (
            reward + self.gamma * next_max - current_q
        )
        self.q_table[state_key][action] = new_q

class TrafficManager:
    def __init__(self):
        self.rl_agent = ReinforcementLearning()
        self.emergency_vehicles = set()
        self.traffic_history = defaultdict(lambda: {'ns': [], 'ew': []})  # Store traffic patterns
        
    def handle_emergency_vehicle(self, vehicle_id):
        """Optimized emergency vehicle handling"""
        try:
            # Set emergency vehicle parameters
            traci.vehicle.setSpeedMode(vehicle_id, 0)  # Disable all speed checks
            traci.vehicle.setSpeed(vehicle_id, EMERGENCY_SPEED)
            
            # Get next traffic light
            try:
                next_tls = traci.vehicle.getNextTLS(vehicle_id)
                if not next_tls:  # If no traffic light ahead, return
                    return
                    
                next_tl = next_tls[0][0]  # Get next traffic light ID
                distance = next_tls[0][2]  # Distance to traffic light
                
                if distance < 100:  # If within 100 meters
                    current_lane = traci.vehicle.getLaneID(vehicle_id)
                    controlled_lanes = traci.trafficlight.getControlledLanes(next_tl)
                    
                    # Set appropriate phase based on emergency vehicle direction
                    if current_lane and any(current_lane in lane for lane in controlled_lanes):
                        if 'A' in current_lane or 'B' in current_lane:
                            traci.trafficlight.setPhase(next_tl, 0)  # NS green
                        else:
                            traci.trafficlight.setPhase(next_tl, 2)  # EW green
                
                # Clear path for emergency vehicle
                current_edge = traci.vehicle.getRoadID(vehicle_id)
                if not current_edge.startswith(':'):  # Not in intersection
                    self.clear_path(vehicle_id)
                    
            except traci.exceptions.TraCIException as e:
                print(f"Warning: Could not get next traffic light for {vehicle_id}: {e}")
                # Continue execution even if there's an error
                pass
                
        except Exception as e:
            print(f"Warning in emergency vehicle handling for {vehicle_id}: {e}")
            # Continue execution even if there's an error
            pass

    def clear_path(self, emergency_id):
        """Clear path for emergency vehicle"""
        try:
            # Get emergency vehicle data
            ev_lane = traci.vehicle.getLaneID(emergency_id)
            if not ev_lane:  # If vehicle not found
                return
                
            ev_edge = traci.vehicle.getRoadID(emergency_id)
            if ev_edge.startswith(':'):  # If in intersection
                return
                
            # Get nearby vehicles
            for vehicle_id in traci.vehicle.getIDList():
                if vehicle_id != emergency_id:
                    try:
                        veh_lane = traci.vehicle.getLaneID(vehicle_id)
                        veh_edge = traci.lane.getEdgeID(veh_lane)
                        
                        # If vehicle is on same edge
                        if veh_edge == ev_edge:
                            # Try to change lane or reduce speed
                            try:
                                num_lanes = len(traci.edge.getLanes(veh_edge))
                                if num_lanes > 1:
                                    current_lane = int(veh_lane.split('_')[-1])
                                    target_lane = (current_lane + 1) % num_lanes
                                    traci.vehicle.changeLane(vehicle_id, target_lane, 5)
                                
                                # Reduce speed
                                traci.vehicle.setSpeed(vehicle_id, MAX_SPEED * 0.5)
                            except:
                                continue
                    except:
                        continue
                        
        except Exception as e:
            print(f"Warning in clear_path for {emergency_id}: {e}")
            # Continue execution even if there's an error
            pass

    def prepare_emergency_route(self, vehicle_id, tl_id, distance):
        """Prepare route for emergency vehicle approach"""
        try:
            current_lane = traci.vehicle.getLaneID(vehicle_id)
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            
            # Determine approach direction
            is_ns = any('A' in lane or 'B' in lane for lane in controlled_lanes 
                    if current_lane in lane)
            
            # Set appropriate phase with timing based on distance
            if is_ns:
                traci.trafficlight.setPhase(tl_id, 0)  # NS green
            else:
                traci.trafficlight.setPhase(tl_id, 2)  # EW green
                
        except traci.exceptions.TraCIException as e:
            print(f"Error preparing emergency route: {e}")

    def update_traffic_light(self, tl_id, current_time):
        """Enhanced hybrid RL + traffic detection control"""
        try:
            # Get current state
            state = self.rl_agent.get_state(tl_id)
            if state is None:
                return
                
            # Get current conditions
            current_phase = traci.trafficlight.getPhase(tl_id)
            last_change = self.rl_agent.last_action_time.get(tl_id, 0)
            min_phase_duration = 25  # Slightly reduced minimum phase time
            
            # Handle emergency vehicles (highest priority)
            if state[3] or state[7]:  # Emergency vehicle present
                if state[3] and current_phase != 0:  # NS emergency
                    traci.trafficlight.setPhase(tl_id, 0)
                    self.rl_agent.last_action_time[tl_id] = current_time
                elif state[7] and current_phase != 2:  # EW emergency
                    traci.trafficlight.setPhase(tl_id, 2)
                    self.rl_agent.last_action_time[tl_id] = current_time
                return

            # Calculate traffic pressure
            lanes = traci.trafficlight.getControlledLanes(tl_id)
            ns_lanes = [lane for lane in lanes if 'A' in lane or 'B' in lane]
            ew_lanes = [lane for lane in lanes if not ('A' in lane or 'B' in lane)]
            
            ns_pressure = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in ns_lanes) * 2 + \
                        sum(traci.lane.getWaitingTime(lane) for lane in ns_lanes) / 10
            ew_pressure = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in ew_lanes) * 2 + \
                        sum(traci.lane.getWaitingTime(lane) for lane in ew_lanes) / 10

            # Get RL action suggestion
            action = self.rl_agent.choose_action(state, tl_id)
            
            # Calculate reward for learning
            reward = self.rl_agent.calculate_reward(tl_id)
            
            # Decision making
            should_change = False
            
            # Check if minimum phase duration has elapsed
            if current_time - last_change >= min_phase_duration:
                # High pressure override
                if ns_pressure > 50 and current_phase != 0:  # High NS pressure
                    should_change = True
                    action = 0
                elif ew_pressure > 50 and current_phase != 2:  # High EW pressure
                    should_change = True
                    action = 2
                # RL-based decision with traffic validation
                elif self.should_change_phase(tl_id, state, action, current_phase):
                    should_change = True

            # Apply change if needed
            if should_change:
                traci.trafficlight.setPhase(tl_id, action)
                self.rl_agent.last_action_time[tl_id] = current_time
                
                # Learn from this action
                next_state = self.rl_agent.get_state(tl_id)
                self.rl_agent.learn(state, action, reward, next_state)
            
            # Debug output (every 100 steps)
            if current_time % 100 < 1:
                print(f"\nTL {tl_id} Status:")
                print(f"Phase: {current_phase} -> {action}")
                print(f"NS Pressure: {ns_pressure:.1f}, EW Pressure: {ew_pressure:.1f}")
                print(f"Last Change: {current_time - last_change:.1f}s ago")
                print(f"Reward: {reward:.2f}")
                
        except Exception as e:
            print(f"Error updating traffic light {tl_id}: {e}")

    def update_traffic_patterns(self, tl_id, state):
        """Update traffic history for pattern learning"""
        ns_traffic = state[0] + state[1] + state[2]  # Queue + waiting + vehicles
        ew_traffic = state[4] + state[5] + state[6]
        
        self.traffic_history[tl_id]['ns'].append(ns_traffic)
        self.traffic_history[tl_id]['ew'].append(ew_traffic)
        
        # Keep only recent history
        if len(self.traffic_history[tl_id]['ns']) > 100:
            self.traffic_history[tl_id]['ns'].pop(0)
            self.traffic_history[tl_id]['ew'].pop(0)

    def should_change_phase(self, tl_id, state, action, current_phase):
        """Enhanced phase change decision making"""
        # Get traffic levels
        ns_traffic = state[0] + state[1] + state[2]
        ew_traffic = state[4] + state[5] + state[6]
        
        # Get trends
        ns_trend = self.get_traffic_trend(tl_id, 'ns')
        ew_trend = self.get_traffic_trend(tl_id, 'ew')
        
        # Calculate imbalance ratio
        if current_phase in [0, 1]:  # NS phase
            imbalance = ew_traffic / max(1, ns_traffic)
            return (imbalance > 1.3 or  # Traffic imbalance
                    state[4] > 7 or     # Long EW queue
                    (ew_trend > 0 and ns_traffic < 3) or  # Growing EW traffic, little NS
                    (action == 2 and ew_traffic > 5))     # RL suggests change and sufficient EW traffic
        else:  # EW phase
            imbalance = ns_traffic / max(1, ew_traffic)
            return (imbalance > 1.3 or  # Traffic imbalance
                    state[0] > 7 or     # Long NS queue
                    (ns_trend > 0 and ew_traffic < 3) or  # Growing NS traffic, little EW
                    (action == 0 and ns_traffic > 5))     # RL suggests change and sufficient NS traffic

    def get_traffic_trend(self, tl_id, direction):
        """Calculate traffic trend (increasing/decreasing)"""
        history = self.traffic_history[tl_id][direction]
        if len(history) < 2:
            return 0
        return history[-1] - history[-2]

    def handle_emergency_state(self, tl_id, state):
        """Handle traffic light state when emergency vehicle present"""
        if state[3]:  # NS emergency
            if state[8] != 0:  # If not NS green
                traci.trafficlight.setPhase(tl_id, 0)
        elif state[7]:  # EW emergency
            if state[8] != 2:  # If not EW green
                traci.trafficlight.setPhase(tl_id, 2)

    def print_traffic_debug(self, tl_id, state, action, reward):
        """Print debug information"""
        print(f"\nTL {tl_id} Status:")
        print(f"NS Traffic: Queue={state[0]}, Wait={state[1]}, Vehicles={state[2]}")
        print(f"EW Traffic: Queue={state[4]}, Wait={state[5]}, Vehicles={state[6]}")
        print(f"Emergency: NS={state[3]}, EW={state[7]}")
        print(f"Action={action}, Reward={reward:.2f}")

def print_statistics(step, traffic_manager):
    """Print detailed simulation statistics"""
    # Time and general metrics
    current_time = traci.simulation.getTime()
    print(f"\n=== Simulation Statistics at Step {step} (Time: {current_time:.1f}s) ===")
    
    # System-wide metrics
    arrived = traci.simulation.getArrivedNumber()
    running = len(traci.vehicle.getIDList())
    mean_speed = sum(traci.vehicle.getSpeed(v) for v in traci.vehicle.getIDList()) / max(1, running)
    
    print(f"\nSystem Metrics:")
    print(f"Running vehicles: {running}")
    print(f"Arrived vehicles: {arrived}")
    print(f"Mean speed: {mean_speed:.2f} m/s")
    
    # Emergency vehicle statistics
    emergency_vehicles = [v for v in traci.vehicle.getIDList() if v.startswith("emergency")]
    print(f"\nEmergency Vehicles ({len(emergency_vehicles)}):")
    for ev in emergency_vehicles:
        try:
            speed = traci.vehicle.getSpeed(ev)
            delay = traci.vehicle.getAccumulatedWaitingTime(ev)
            distance = traci.vehicle.getDistance(ev)
            edge = traci.vehicle.getRoadID(ev)
            print(f"  {ev}:")
            print(f"    Speed: {speed:.1f} m/s")
            print(f"    Total delay: {delay:.1f} s")
            print(f"    Distance traveled: {distance:.1f} m")
            print(f"    Current edge: {edge}")
        except:
            continue
    
    # Traffic light performance
    print("\nTraffic Light Performance:")
    for tl_id in traci.trafficlight.getIDList():
        try:
            # Get controlled lanes
            lanes = traci.trafficlight.getControlledLanes(tl_id)
            ns_lanes = [l for l in lanes if 'A' in l or 'B' in l]
            ew_lanes = [l for l in lanes if not ('A' in l or 'B' in l)]
            
            # Calculate metrics for each direction
            ns_metrics = {
                'queue': sum(traci.lane.getLastStepHaltingNumber(l) for l in ns_lanes),
                'waiting': sum(traci.lane.getWaitingTime(l) for l in ns_lanes),
                'vehicles': sum(traci.lane.getLastStepVehicleNumber(l) for l in ns_lanes)
            }
            
            ew_metrics = {
                'queue': sum(traci.lane.getLastStepHaltingNumber(l) for l in ew_lanes),
                'waiting': sum(traci.lane.getWaitingTime(l) for l in ew_lanes),
                'vehicles': sum(traci.lane.getLastStepVehicleNumber(l) for l in ew_lanes)
            }
            
            print(f"\n  TL {tl_id}:")
            print(f"    Phase: {traci.trafficlight.getPhase(tl_id)}")
            print(f"    NS Direction:")
            print(f"      Queue length: {ns_metrics['queue']}")
            print(f"      Waiting time: {ns_metrics['waiting']:.1f} s")
            print(f"      Vehicles: {ns_metrics['vehicles']}")
            print(f"    EW Direction:")
            print(f"      Queue length: {ew_metrics['queue']}")
            print(f"      Waiting time: {ew_metrics['waiting']:.1f} s")
            print(f"      Vehicles: {ew_metrics['vehicles']}")
            
        except traci.exceptions.TraCIException as e:
            print(f"Error getting statistics for {tl_id}: {e}")
            continue

def run_simulation():
    """Main simulation loop with performance tracking"""
    try:
        # Initialize traffic manager and metrics
        traffic_manager = TrafficManager()
        step = 0
        total_waiting_time = 0
        total_vehicles = 0
        emergency_metrics = {
            'total_delay': 0,
            'total_travel_time': 0,
            'completed_trips': 0
        }
        
        print("Starting simulation...")
        
        while step < SIMULATION_STEPS:
            try:
                current_time = traci.simulation.getTime()
                vehicle_ids = traci.vehicle.getIDList()
                
                # Process vehicles
                for vehicle_id in vehicle_ids:
                    if vehicle_id.startswith("emergency"):
                        # Handle emergency vehicles
                        traffic_manager.handle_emergency_vehicle(vehicle_id)
                        emergency_metrics['total_delay'] += traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
                    else:
                        # Regular vehicle speed control
                        try:
                            speed = traci.vehicle.getSpeed(vehicle_id)
                            if speed > MAX_SPEED:
                                traci.vehicle.setSpeed(vehicle_id, MAX_SPEED)
                        except:
                            continue
                
                # Update traffic lights
                for tl_id in traci.trafficlight.getIDList():
                    traffic_manager.update_traffic_light(tl_id, current_time)
                
                # Collect metrics
                total_waiting_time += sum(traci.vehicle.getAccumulatedWaitingTime(v) 
                                        for v in vehicle_ids)
                total_vehicles = max(total_vehicles, len(vehicle_ids))
                
                # Track completed emergency vehicle trips
                arrived = traci.simulation.getArrivedIDList()
                emergency_metrics['completed_trips'] += sum(1 for v in arrived 
                                                        if v.startswith("emergency"))
                
                # Print statistics periodically
                if step % 100 == 0:
                    print_statistics(step, traffic_manager)
                    
                    # Print average metrics
                    if step > 0:
                        avg_waiting = total_waiting_time / (step * len(vehicle_ids)) if vehicle_ids else 0
                        print("\nCumulative Metrics:")
                        print(f"Average waiting time: {avg_waiting:.2f} s")
                        print(f"Peak vehicle count: {total_vehicles}")
                        print(f"Completed emergency trips: {emergency_metrics['completed_trips']}")
                
                traci.simulationStep()
                step += 1
                
            except traci.exceptions.TraCIException as e:
                print(f"Error at step {step}: {e}")
                if "connection closed by SUMO" in str(e):
                    break
                continue
            
        # Print final statistics
        print("\n=== Final Simulation Results ===")
        print(f"Total simulation steps: {step}")
        print(f"Average waiting time: {total_waiting_time/(step*total_vehicles) if total_vehicles else 0:.2f} s")
        print(f"Peak vehicle count: {total_vehicles}")
        print(f"Emergency vehicle metrics:")
        print(f"  Completed trips: {emergency_metrics['completed_trips']}")
        print(f"  Average delay: {emergency_metrics['total_delay']/max(1, emergency_metrics['completed_trips']):.2f} s")
        
    except Exception as e:
        print(f"Critical simulation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            traci.close()
            print("\nTraCI connection closed")
        except:
            pass

if __name__ == "__main__":
    try:
        # Configure and start SUMO
        sumoBinary = "sumo-gui"  # Use "sumo" for no GUI
        sumoCmd = [
            sumoBinary,
            "-c", "simulation.sumocfg",
            "--step-length", "0.1",
            "--collision.action", "warn",
            "--time-to-teleport", "-1",
            "--ignore-junction-blocker", "60",
            "--waiting-time-memory", "300",
            "--no-warnings",
            "--duration-log.statistics",
            "--log", "simulation.log",
            "--gui-settings-file", "gui-settings.xml",  # Optional: custom GUI settings
            "--device.emissions.probability", "1.0",    # Enable emissions tracking
            "--quit-on-end", "false",
            "--start", "true"
        ]
        
        print("Starting SUMO with command:", " ".join(sumoCmd))
        traci.start(sumoCmd)
        run_simulation()
        
    except Exception as e:
        print(f"Error starting simulation: {e}")
