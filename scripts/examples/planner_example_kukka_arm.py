from scripts.simulation.SimulationWorld import SimulationWorld
import os
from scripts.Robot import Robot
from scripts.utils.utils import Utils as utils
import numpy as np
from scripts.TrajectoryOptimizationPlanner.TrajectoryOptimizationPlanner import TrajectoryOptimizationPlanner

class PlannerExample:
    def __init__(self):
        home = os.path.expanduser('~')

        location_prefix = home + '/masterThesis/bullet3/data/'

        urdf_file = location_prefix + "kuka_iiwa/model.urdf"

        self.planner = TrajectoryOptimizationPlanner(urdf_file, use_gui=True)

        # self.planner.world = Simulationplanner.world(urdf_file, use_gui=False)
        self.planner.world.set_gravity(0, 0, -10)
        self.planner.world.toggle_rendering(0)
        self.planner.world.load_urdf(urdf_file=location_prefix + "plane.urdf", position=[0, 0, 0.0])

        table_id = self.planner.add_constraint_from_urdf(urdf_file=location_prefix + "table/table.urdf", position=[0, 0, 0.0])


        self.box_id = self.planner.add_constraint(shape=self.planner.world.BOX, size=[0.1, 0.2, 0.45],
                                             position=[0.28, -0.43, 0.9], mass=100)

        self.planner.world.toggle_rendering(1)
        self.planner.world.step_simulation_for(0.01)

    def run_simulation(self):
        iteration_count = 0
        while 1:
            if iteration_count < 1:
                iteration_count += 1
                start_state = {}
                goal_state = {}

                start_state["lbr_iiwa_joint_1"] = -2.4823357809267463
                start_state["lbr_iiwa_joint_2"] = 1.4999975516996142
                start_state["lbr_iiwa_joint_3"] = -1.5762726255540713
                start_state["lbr_iiwa_joint_4"] = -0.8666279970481103
                start_state["lbr_iiwa_joint_5"] = 1.5855963769735366
                start_state["lbr_iiwa_joint_6"] = 1.5770985888989753
                start_state["lbr_iiwa_joint_7"] = 1.5704531145724918

                goal_state["lbr_iiwa_joint_1"] = -0.08180533826032865
                goal_state["lbr_iiwa_joint_2"] = 1.5474152457596664
                goal_state["lbr_iiwa_joint_3"] = -1.5873548294514912
                goal_state["lbr_iiwa_joint_4"] = -0.5791571346767671
                goal_state["lbr_iiwa_joint_5"] = 1.5979105177314896
                goal_state["lbr_iiwa_joint_6"] = 1.5857854098720727
                goal_state["lbr_iiwa_joint_7"] = 1.5726221954434347

                duration = 10
                samples = 20
                self.planner.world.reset_joint_states(self.planner.robot.id, start_state)
                self.planner.world.step_simulation_for(2)
                collision_check_distance = 0.1
                collision_safe_distance = 0.05
                group = goal_state.keys()

                status, trajectory = self.planner.get_trajectory(group, goal_state=goal_state, samples=samples, duration=duration,
                     collision_safe_distance=collision_safe_distance,
                     collision_check_distance=collision_check_distance)
                print("if trajectory has collision: ", status)
                self.planner.execute_trajectory()



                # import sys
                # sys.exit()

if __name__ == '__main__':
    example = PlannerExample()
    example.run_simulation()