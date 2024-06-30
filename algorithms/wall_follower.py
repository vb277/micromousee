import API
import sys

def log(string):
    sys.stderr.write("{}\n".format(string))
    sys.stderr.flush()

DIRECTIONS = ["NORTH", "EAST", "SOUTH", "WEST"]
MOVE_MAP = {
    "NORTH": (0, 1),
    "EAST": (1, 0),
    "SOUTH": (0, -1),
    "WEST": (-1, 0)
}

def is_goal(x, y):
    goal_cells = [(7, 7), (8, 7), (7, 8), (8, 8)]
    return (x, y) in goal_cells

def wall_follower(x, y, current_direction):
    move_counter = 0
    turn_counter = 0

    while not is_goal(x, y):
        if not API.wallLeft():
            API.turnLeft()
            turn_counter += 1
            current_direction = DIRECTIONS[(DIRECTIONS.index(current_direction) - 1) % 4]
            if not API.wallFront():
                API.moveForward()
                move_counter += 1
                x, y = x + MOVE_MAP[current_direction][0], y + MOVE_MAP[current_direction][1]
        elif not API.wallFront():
            API.moveForward()
            move_counter += 1
            x, y = x + MOVE_MAP[current_direction][0], y + MOVE_MAP[current_direction][1]
        else:
            API.turnRight()
            turn_counter += 1
            current_direction = DIRECTIONS[(DIRECTIONS.index(current_direction) + 1) % 4]
    
    return move_counter, turn_counter

def run_wall_follower():
    log("Running Wall Follower...")
    API.setColor(0, 0, "G")
    API.setText(0, 0, "Start")

    start_x, start_y = 0, 0 
    log(f"Starting at ({start_x}, {start_y})")

    current_x, current_y = start_x, start_y
    current_direction = "NORTH"

    move_counter, turn_counter = wall_follower(current_x, current_y, current_direction)

    total_distance = API.getStat("total-distance")
    total_turns = API.getStat("total-turns")
    log(f"Total distance: {total_distance} (Counted: {move_counter})")
    log(f"Total turns: {total_turns} (Counted: {turn_counter})")