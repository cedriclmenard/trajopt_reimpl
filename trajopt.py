from networkx.generators.classic import path_graph
import yaml
import os.path
from collections import OrderedDict
import time
import numpy as np
from urdfpy import URDF as URDFVis

from scripts.simulation.SimulationWorld import SimulationWorld
from scripts.Robot import Robot

from typing import (
    List,
    Union
)

class RobotConfiguration():
    def __init__(self,  robot: Robot,
                        planning_group_name: str = None, 
                        planning_group: List[str] = None, 
                        joint_configuration: str = None, 
                        joint_values: List[float] = None
            ):
        self.robot = robot
        self.planning_group_name = ""
        self.planning_group = []
        self.joint_configuration_name = ""
        self.joint_values = []

        if planning_group_name:
            self.set_planning_group_name(planning_group_name)
        if planning_group:
            self.set_planning_group(planning_group)
        if joint_configuration:
            self.set_joint_configuration(joint_configuration)
        if joint_values:
            self.set_joint_values(joint_values)
    
    def set_planning_group_name(self, name: str):
        self.planning_group_name = name
        self.planning_group = self.robot.get_planning_group_from_srdf(name)
        return self
    
    def set_planning_group(self, group: List[str]):
        self.planning_group = group
        self.planning_group_name = ""
        return self
    
    def set_joint_configuration(self, name: str):
        self.joint_configuration_name = name
        _, self.joint_values = self.robot.get_planning_group_joint_values(name, self.planning_group_name)
        return self
    
    def set_joint_values(self, values: List[float]):
        self.joint_values = values
        self.joint_configuration_name = ""
        return self
    
    def get_available_planning_group_names(self) -> List[str]:
        group_names = set()
        for (joint_configuration, group_name) in self.robot.group_states_map:
            group_names.add(group_names)
        return list(group_names)
    
    def get_available_joint_configurations(self, planning_group_name: str = None) -> List[str]:
        if not planning_group_name:
            planning_group_name = self.planning_group_name
        
        joint_configurations = set()
        for (joint_configuration, group_name) in self.robot.group_states_map:
            if group_name == planning_group_name:
                joint_configurations.add(joint_configuration)
        return list(joint_configurations)

    def visualize(self) -> None:
        robot = URDFVis.load(self.robot.urdf_file)
        cfg = dict(zip(self.planning_group, self.joint_values))
        robot.show(cfg)

class PathConfiguration(RobotConfiguration):
    def __init__(self,  robot: Robot,
                        planning_group_name: Union[List[str], str] = None, 
                        planning_group: Union[List[List[str]], List[str]] = None, 
                        joint_configuration: List[str] = None, 
                        joint_values: List[List[float]] = None
            ):
        self.robot = robot
        self.planning_group_name = []
        self.planning_group = []
        self.joint_configuration_name = []
        self.joint_values = []

        if planning_group_name:
            self.set_planning_group_name(planning_group_name)
        if planning_group:
            self.set_planning_group(planning_group)
        if joint_configuration:
            self.set_joint_configuration(joint_configuration)
        if joint_values:
            self.set_joint_values(joint_values)

    def set_planning_group_name(self, name: Union[List[str], str], n: int = 0):
        """Set planning group for path using its name

        Args:
            name (Union[List[str], str]): planning group name as described in the SRDF
            n (int, optional): Number of points in the path. Specify only if name is not a list. Defaults to 0.

        Returns:
            PathConfiguration: self
        """
        if type(name) is list:
            self.planning_group_name = name
            self.planning_group = [self.robot.get_planning_group_from_srdf(n) for n in name]
        else:
            assert n != 0, "n cannot be 0 when name is not a list"
            self.planning_group_name = [name for _ in range(n)]
            self.planning_group = [self.robot.get_planning_group_from_srdf(name) for _ in range(n)]
        return self
    
    def set_planning_group(self, group: Union[List[List[str]], List[str]], n: int = 0):
        if type(group) is list and all(type(g) is list for g in group):
            self.planning_group = group
            self.planning_group_name = ["" for _ in group]
        else:
            assert n != 0, "n cannot be 0 when group is not a list of lists"
            self.planning_group = [group for _ in range(n)]
            self.planning_group_name = ["" for _ in range(n)]
        return self
    
    def set_joint_configuration(self, name: List[str]):
        assert type(name) is list, "joint configurations should be a list of configurations"
        self.joint_configuration_name = name
        _, self.joint_values = [self.robot.get_planning_group_joint_values(n, self.planning_group_name) for n in name]
        if len(self.joint_values) > len(self.planning_group):
            # make sure we duplicate planning group if it is a single set
            self.planning_group = [self.planning_group for _ in self.joint_values]
            self.planning_group_name = [self.planning_group_name for _ in self.joint_values]
        return self
    
    def set_joint_values(self, values: List[List[float]]):
        assert type(values) is list and all(type(v) is list for v in values), "values should be a list of list of joint values (state)"
        self.joint_values = values
        self.joint_configuration_name = ["" for _ in values]

        if len(self.joint_values) > len(self.planning_group):
            # make sure we duplicate planning group if it is a single set
            self.planning_group = [self.planning_group for _ in self.joint_values]
            self.planning_group_name = [self.planning_group_name for _ in self.joint_values]
        return self
    
    def visualize(self) -> None:
        robot = URDFVis.load(self.robot.urdf_file)
        # groups = self.get
        cfg = {}
        for (pg, jvs) in zip(self.planning_group, self.joint_values):
            for jn, jv in zip(pg, jvs):
                if jn not in cfg:
                    cfg[jn] = [jv]
                else:
                    cfg[jn].append(jv)
        robot.animate(cfg)

def visualize_base_path(robot_urdf_file, path):
    robot = URDFVis.load(robot_urdf_file)
    cfg = dict(path.trajectory_by_name)
    robot.animate(cfg)

class TrajOpt():
    def __init__(self, robot_urdf, robot_srdf, robot_position, robot_orientation, sqp_config=None, use_gui=False) -> None:
        self.if_plot_traj = False

        # Load world
        self.world = SimulationWorld(use_gui=use_gui)

        # Load robot
        self.robot = Robot()
        self.robot.id = self.world.load_robot(robot_urdf, robot_position, robot_orientation, use_fixed_base=True)
        self.robot.load_robot_model(robot_urdf)
        self.load_robot_srdf(robot_srdf)

        

        # Load solver config
        with open(os.path.join(os.path.dirname(__file__), "config", "sqp_config.yaml"), 'r') as f:
            self.sqp_config = yaml.safe_load(f)["sqp"]

    def add_constraint(self, name, shape, mass, position, size=None, radius=None, height=None, orientation=None):
        shape_id = self.world.create_constraint(name, shape, mass, position, orientation, size, radius, height)
        self.world.add_collision_constraints(shape_id)
        return shape_id

    def add_constraint_heightfield(self, name, data, mass=None, position=None, orientation=None, mesh_scale=None, use_maximalcoordinates=True):
        shape_id = self.world.create_constraint_from_heightmap(name, data, mass, position, orientation, mesh_scale, use_maximalcoordinates)
        self.world.add_collision_constraints(shape_id)
        return shape_id

    def reset_robot_to(self, configuration: RobotConfiguration):
        self.world.reset_joint_states(self.robot.id, configuration.joint_values, configuration.planning_group)

    def generate_trajectory(self, start_configuration: RobotConfiguration, goal_configuration: RobotConfiguration, samples=20, duration=10, collision_safe_distance=0.05, collision_check_distance=0.1, ignore_goal_states=True):
        if type(ignore_goal_states) is bool:
            value = ignore_goal_states
            ignore_goal_states = [value for _ in range(len(start_configuration.joint_values))]
        self.reset_robot_to(start_configuration)
        status, is_collision_free, trajectory = "start state in collision", False, -1
        is_start_state_in_collision = self.world.is_given_state_in_collision(self.robot.id, start_configuration.joint_values, start_configuration.planning_group)
        if is_start_state_in_collision:
            print("is_start_state_in_collision", is_start_state_in_collision)
            status = "start state in collision"
            return status, is_collision_free, trajectory

        status, is_collision_free, trajectory = "goal state in collision", False, -1
        is_goal_in_collision = self.world.is_given_state_in_collision(self.robot.id, goal_configuration.joint_values, goal_configuration.planning_group)
        if is_goal_in_collision:
            print("is_goal_in_collision", is_goal_in_collision)
            status = "goal state in collision"
            return status, is_collision_free, trajectory

        self.robot.init_plan_trajectory(group=start_configuration.planning_group, current_state=start_configuration.joint_values,
                                        goal_state=goal_configuration.joint_values, samples=samples, duration=duration,
                                        collision_safe_distance=collision_safe_distance,
                                        collision_check_distance=collision_check_distance,
                                        solver_class=self.sqp_config["solver_class"],
                                        ignore_goal_states=ignore_goal_states
                                        )

        self.world.toggle_rendering_while_planning(False)
        _, planning_time, _ = self.robot.calulate_trajectory(self.callback_function_from_solver)
        trajectory = self.robot.planner.get_trajectory()

        is_collision_free = self.world.is_trajectory_collision_free(self.robot.id, self.robot.get_trajectory().final,
                                                                    start_configuration.planning_group,
                                                                    0.02)
        self.world.toggle_rendering_while_planning(True)
        self.elapsed_time = self.robot.planner.sqp_solver.solving_time + \
                            self.world.collision_check_time + self.robot.planner.prob_model_time
        status = "Optimal Trajectory has been found in " + str(self.elapsed_time) + " secs"
        # self.logger.info(status)
        # self.log_infos()

        if self.if_plot_traj:
            self.robot.planner.trajectory.plot_trajectories()

        return status, is_collision_free, trajectory

    # def get_group_names(self, group):
    #     if type(group) is str:
    #         group = self.robot_config["joints_groups"][group]
    #     if type(group) is dict or type(group) is OrderedDict:
    #         group = group.values()

    #     return group
    
    def callback_function_from_solver(self, new_trajectory, delta_trajectory=None):

        constraints, lower_limit, upper_limit = None, None, None
        new_trajectory = new_trajectory[:self.robot.planner.no_of_samples * self.robot.planner.num_of_joints]
        trajectory = np.split(new_trajectory, self.robot.planner.no_of_samples)
        self.robot.planner.trajectory.add_trajectory(trajectory)
        start = time.time()
        collision_infos = self.world.get_collision_infos(self.robot, trajectory, self.robot.planner.current_planning_joint_group,
                                                         distance=self.robot.planner.collision_check_distance)
        end = time.time()
        self.elapsed_time = (end - start) + self.robot.planner.sqp_solver.solving_time
        # self.elapsed_time = self.world.collision_check_time + self.robot.planner.sqp_solver.solving_time
        if len(collision_infos[2]) > 0:
            constraints, lower_limit, upper_limit = \
                self.robot.planner.problem_model.update_collision_infos(collision_infos, self.robot.planner.collision_safe_distance)
            self.robot.planner.update_prob()

        return constraints, lower_limit, upper_limit

    def get_group_and_state(self, group, goal_state=None):
        if type(group) is str:
            group = self.robot_config["joints_groups"][group]
        if type(goal_state) is str:
            goal_state = self.robot_config["joint_configurations"][goal_state]

        if type(goal_state) is dict or type(goal_state) is OrderedDict:
            goal_state = goal_state.values()

        return goal_state

    def load_robot_srdf(self, srdf_file):
        self.robot.load_srdf(srdf_file)
        self.world.ignored_collisions = self.robot.get_ignored_collsion()

    def execute_trajectory(self):
        # self.world.toggle_rendering(1)
        # self.world.toggle_rendering_while_planning(True)
        self.world.execute_trajectory(self.robot, self.robot.planner.get_trajectory())
        # self.world.toggle_rendering(0)

        return "Trajectory execution completed"