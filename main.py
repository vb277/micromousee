from algorithms.dfs import run_dfs
from algorithms.wall_follower import run_wall_follower
from algorithms.flood_fill import run_flood_fill
from algorithms.flood_fill_6 import run_flood_fill_6  # Import the 6x6 flood fill function

def main():
    algorithm = "flood_fill_6"  # Change this to switch between algorithms

    if algorithm == "dfs":
        run_dfs()
    elif algorithm == "wall_follower":
        run_wall_follower()
    elif algorithm == "flood_fill":
        run_flood_fill()
    elif algorithm == "flood_fill_6":
        run_flood_fill_6()

if __name__ == "__main__":
    main()
