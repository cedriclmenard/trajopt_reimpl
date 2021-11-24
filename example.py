from trajopt import TrajOpt

if __name__ == '__main__':
    to = TrajOpt("config/kinova_m1n6s300/m1n6s300.urdf", "config/kinova_m1n6s300/m1n6s300.srdf", [0,0,0], [0,0,0])
    status, collision_free, trajectory = to.generate_trajectory(group="arm", start_state="Home", goal_state="Vertical")
    print(trajectory)