from trajopt import TrajOpt, RobotConfiguration, visualize_base_path
import pybullet as p

if __name__ == '__main__':
    to = TrajOpt("config/kinova_m1n6s300/m1n6s300.urdf", "config/kinova_m1n6s300/m1n6s300.srdf", [0,0,0], [0,0,0], use_gui=True)
    to.add_constraint("test", shape=p.GEOM_BOX, position=[0.1,-0.1,0.6], size=[0.05,0.05,0.05], mass=100)
    robot = to.robot
    start_configuration = RobotConfiguration(robot, planning_group_name="arm", joint_configuration="Home")
    # start_configuration.visualize()
    goal_configuration = RobotConfiguration(robot, planning_group_name="arm", joint_configuration="Vertical")
    # goal_configuration.visualize()
    status, collision_free, trajectory = to.generate_trajectory(start_configuration, goal_configuration)
    if collision_free:
        to.execute_trajectory()
    else:
        print("Collision")
    # visualize_base_path("config/kinova_m1n6s300/m1n6s300.urdf", trajectory)
    print(trajectory)