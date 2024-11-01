import os
import sys
import traci
import random

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

class TrafficLight:
    def __init__(self, id):
        self.id = id
        self.waiting_time = 0
        self.vehicle_count = 0
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(id)[0]
        self.min_dur = min(phase.minDur for phase in logic.phases if hasattr(phase, 'minDur'))
        self.max_dur = max(phase.maxDur for phase in logic.phases if hasattr(phase, 'maxDur'))

    def update(self):
        lanes = traci.trafficlight.getControlledLanes(self.id)
        self.waiting_time = sum(traci.lane.getWaitingTime(lane) for lane in lanes)
        self.vehicle_count = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in lanes)

class AITrafficControl:
    def __init__(self):
        self.traffic_lights = {tl: TrafficLight(tl) for tl in traci.trafficlight.getIDList()}

    def update(self):
        for tl_id, tl in self.traffic_lights.items():
            tl.update()
            
            current_phase = traci.trafficlight.getPhase(tl_id)
            current_program = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]
            current_phase_def = current_program.phases[current_phase]
            
            if "G" in current_phase_def.state:  # Only adjust duration for green phases
                if tl.waiting_time > 50 or tl.vehicle_count > 5:
                    new_duration = min(tl.max_dur, current_phase_def.duration + 5)
                else:
                    new_duration = max(tl.min_dur, current_phase_def.duration - 5)
                
                traci.trafficlight.setPhaseDuration(tl_id, new_duration)

    def print_stats(self):
        print("\nTraffic Light Statistics:")
        for tl_id, tl in self.traffic_lights.items():
            phase = traci.trafficlight.getPhase(tl_id)
            duration = traci.trafficlight.getPhaseDuration(tl_id)
            print(f"  {tl_id}: Waiting Time = {tl.waiting_time:.2f}s, Vehicles = {tl.vehicle_count}, Phase = {phase}, Duration = {duration:.2f}s")

class VANETSimulation:
    def __init__(self):
        self.ai_control = AITrafficControl()

    def run(self):
        step = 0
        while step < 3600:  # Run for 1 hour of simulation time
            traci.simulationStep()
            
            self.ai_control.update()
            
            if step % 100 == 0:  # Print stats every 10 seconds
                self.ai_control.print_stats()
                self.print_vehicle_stats()
            
            step += 1
        
        traci.close()
        print("Simulation ended.")

    def print_vehicle_stats(self):
        vehicles = traci.vehicle.getIDList()
        print(f"\nVehicle Statistics (Total: {len(vehicles)}):")
        for v_id in random.sample(vehicles, min(5, len(vehicles))):
            speed = traci.vehicle.getSpeed(v_id)
            waiting_time = traci.vehicle.getAccumulatedWaitingTime(v_id)
            edge = traci.vehicle.getRoadID(v_id)
            route = traci.vehicle.getRoute(v_id)
            print(f"  {v_id}: Speed = {speed:.2f} m/s, Waiting Time = {waiting_time:.2f}s, Edge = {edge}")
            print(f"    Route: {' -> '.join(route)}")

if __name__ == "__main__":
    sumoBinary = "sumo-gui"
    sumoCmd = [sumoBinary, "-c", "simulation.sumocfg"]
    
    traci.start(sumoCmd)
    sim = VANETSimulation()
    sim.run()
