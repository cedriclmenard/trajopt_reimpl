import yaml
import os.path
from collections import OrderedDict
import time
import numpy as np

from scripts.simulation.SimulationWorld import SimulationWorld
from scripts.Robot import Robot


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

    def get_trajectory(self, group_in=None, start_state=None, goal_state=None, samples=20, duration=10, collision_safe_distance=0.05, collision_check_distance=0.1, ignore_goal_states=True):
        group = []
        group_name = group_in
        if group_name is not None:
            if type(group_name) is list:
                group = group_name
            if type(group_name) is str and group_name in self.robot_config["joints_groups"]:
                group = self.robot_config["joints_groups"][group_name]
            if not len(group):
                group = self.robot.get_planning_group_from_srdf(group_name)

        
        if start_state is not None and len(group):
            if type(start_state) is dict or type(start_state) is OrderedDict:
                start_state = start_state.values()
            if not type(start_state) is list:
                _, start_state = self.get_planning_group_and_corresponding_state("start_state", group=group_in)
            self.reset_robot_to(start_state, group, key="start_state")
            status, is_collision_free, trajectory = "start state in collision", False, -1
            is_start_state_in_collision = self.world.is_given_state_in_collision(self.robot.id, start_state, group)
            if is_start_state_in_collision:
                print("is_start_state_in_collision", is_start_state_in_collision)
                status = "start state in collision"
                return status, is_collision_free, trajectory
        elif len(group):
            start_state = self.world.get_current_states_for_given_joints(self.robot.id, group)

        if goal_state is not None and len(group):
            if type(goal_state) is dict or type(goal_state) is OrderedDict:
                goal_state = goal_state.values()
            if not type(goal_state) is list:
                _, goal_state = self.get_planning_group_and_corresponding_state("goal_state", group=group_in)
                status, is_collision_free, trajectory = "goal state in collision", False, -1
                is_goal_in_collision = self.world.is_given_state_in_collision(self.robot.id, goal_state, group)
                if is_goal_in_collision:
                    print("is_goal_in_collision", is_goal_in_collision)
                    status = "goal state in collision"
                    return status, is_collision_free, trajectory

        self.robot.init_plan_trajectory(group=group, current_state=start_state,
                                        goal_state=goal_state, samples=samples, duration=duration,
                                        collision_safe_distance=collision_safe_distance,
                                        collision_check_distance=collision_check_distance,
                                        solver_class=self.sqp_config["solver_class"],
                                        ignore_goal_states=ignore_goal_states
                                        )

        self.world.toggle_rendering_while_planning(False)
        _, planning_time, _ = self.robot.calulate_trajecotory(self.callback_function_from_solver)
        trajectory = self.robot.planner.get_trajectory()

        is_collision_free = self.world.is_trajectory_collision_free(self.robot.id, self.robot.get_trajectory().final,
                                                                    group,
                                                                    0.02)
        self.world.toggle_rendering_while_planning(True)
        self.elapsed_time = self.robot.planner.sqp_solver.solving_time + \
                            self.world.collision_check_time + self.robot.planner.prob_model_time
        status = "Optimal Trajectory has been found in " + str(self.elapsed_time) + " secs"
        self.logger.info(status)
        self.log_infos()

        if self.if_plot_traj:
            self.robot.planner.trajectory.plot_trajectories()

        return status, is_collision_free, trajectory

    def reset_robot_to(self, state, group, key="reset_state"):
        group, joint_states = self.get_planning_group_and_corresponding_state(key, group=group, reset_state=state)
        self.world.reset_joint_states(self.robot.id, joint_states, group)

    def get_group_names(self, group):
        if type(group) is str:
            group = self.robot_config["joints_groups"][group]
        if type(group) is dict or type(group) is OrderedDict:
            group = group.values()

        return group
    
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