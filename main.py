from algorithms.dfs import run_dfs
from algorithms.wall_follower import run_wall_follower
from algorithms.flood_fill import run_flood_fill
from algorithms.flood_fill_6 import run_flood_fill_6
from algorithms.ff_manhattan_6 import run_ff_manhattan_6
from algorithms.ff_manhattan import run_ff_manhattan
from algorithms.directional_heuristic import run_directional_heuristic
from algorithms.directional_heuristic_6 import run_directional_heuristic_6
# from algorithms.ff_and_m_6 import ff_and_m_6
# from algorithms.ff_and_m import ff_and_m

def main():
    algorithm = "ff_manhattan"  

    if algorithm == "dfs":
        run_dfs()
    elif algorithm == "wall_follower":
        run_wall_follower()
    elif algorithm == "flood_fill":
        run_flood_fill()
    elif algorithm == "flood_fill_6":
        run_flood_fill_6()
    elif algorithm == "ff_manhattan_6":
        run_ff_manhattan_6()
    elif algorithm == "ff_manhattan":
        run_ff_manhattan()
    elif algorithm == "directional":
        run_directional_heuristic()
    elif algorithm == "directional":
        run_directional_heuristic_6()
    # elif algorithm == "ff_and_m_6":
    #     ff_and_m_6()
    # elif algorithm == "ff_and_m":
    #     ff_and_m()

if __name__ == "__main__":
    main()
