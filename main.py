from algorithms.dfs import run_dfs
from algorithms.wall_follower import run_wall_follower

def main():
    algorithm = "dfs"  # Change this to switch between algorithms

    if algorithm == "dfs":
        run_dfs()
    elif algorithm == "wall_follower":
        run_wall_follower()

if __name__ == "__main__":
    main()