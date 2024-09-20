from algorithms.dfs import run_dfs
from algorithms.wall_follower import run_wall_follower
from algorithms.flood_fill import run_flood_fill
from algorithms.flood_fill_6 import run_flood_fill_6
from algorithms.ff_manhattan_6 import run_ff_manhattan_6
from algorithms.ff_manhattan import run_ff_manhattan
from algorithms.directional_heuristic import run_directional_heuristic
from algorithms.directional_heuristic_6 import run_directional_heuristic_6
from algorithms.d_lite_6 import run_d_lite_6
from algorithms.d_lite import run_d_lite
from algorithms.d_lite_km import run_d_lite_km
from algorithms.d_lite_4 import run_d_lite_4
from algorithms.d_lite_ff_6 import run_d_lite_ff_6
from algorithms.d_lite_ff import run_d_lite_ff
from algorithms.d_lite_32 import run_d_lite_32
from algorithms.flood_fill_32 import run_flood_fill_32
from algorithms.d_lite_ff_32 import run_d_lite_ff_32

def main():
    algorithm = "dlite6"  
    # dlite dliteff flood_fill
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
    elif algorithm == "directional6":
        run_directional_heuristic_6()
    elif algorithm == "dlite6":
        run_d_lite_6()
    elif algorithm == "dlite":
        run_d_lite()
    elif algorithm == "dlitekm":
        run_d_lite_km()
    elif algorithm == "dlite4":
        run_d_lite_4()
    elif algorithm == "dliteff6":
        run_d_lite_ff_6()
    elif algorithm == "dliteff":
        run_d_lite_ff()
    elif algorithm == "dlite32":
        run_d_lite_32()
    elif algorithm == "ff32":
        run_flood_fill_32()
    elif algorithm == "dliteff32":
        run_d_lite_ff_32()
        

if __name__ == "__main__":
    main()