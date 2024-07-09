import API
import sys
from collections import deque

def log(string):
    """
    Log messages for debugging.
    """
    sys.stderr.write("{}\n".format(string))
    sys.stderr.flush()

def valid_position(x, y, width, height):
    """
    Check if a given position is within the bounds of the maze.
    
    Args:
    x (int): The x-coordinate of the position.
    y (int): The y-coordinate of the position.
    width (int): The width of the maze.
    height (int): The height of the maze.
    
    Returns:
    bool: True if the position is valid, False otherwise.
    """
    return 0 <= x < width and 0 <= y < height

def flood_fill(maze, width, height):
    """
    Perform the flood fill algorithm to propagate the distance values from the goal cells.
    
    Args:
    maze (list): The 2D list representing the maze.
    width (int): The width of the maze.
    height (int): The height of the maze.
    """
    # Define the possible directions of movement: NORTH, EAST, SOUTH, WEST
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    # Define the goal cells
    goal_cells = [(7, 7), (8, 7), (7, 8), (8, 8)]
    # Initialize a queue with the goal cells
    queue = deque(goal_cells)

    while queue:
        x, y = queue.popleft()  # Get the current cell from the queue
        current_distance = maze[y][x]  # Get the distance value of the current cell

        # Set the distance value in the simulator using setText
        API.setText(x, y, str(int(current_distance)))

        # Check all possible directions from the current cell
        for dx, dy in directions:
            nx, ny = x + dx, y + dy  # Calculate the new cell coordinates
            if valid_position(nx, ny, width, height) and maze[ny][nx] == float('inf'):
                # If the new cell is within bounds and not yet visited, update its distance
                maze[ny][nx] = current_distance + 1
                queue.append((nx, ny))  # Add the new cell to the queue for further processing

def check_wall(direction):
    """
    Check for a wall in a given direction relative to the current orientation.
    
    Args:
    direction (int): The direction to check (0 = NORTH, 1 = EAST, 2 = SOUTH, 3 = WEST).
    
    Returns:
    bool: True if there is a wall, False otherwise.
    """
    if direction == 0:  # NORTH
        return API.wallFront()
    elif direction == 1:  # EAST
        return API.wallRight()
    elif direction == 2:  # SOUTH (behind the robot, not directly visible)
        return False  # Placeholder, should be handled differently in actual implementation
    elif direction == 3:  # WEST
        return API.wallLeft()

def update_walls(x, y, direction, has_wall, horizontal_walls, vertical_walls):
    """
    Update the internal map with the detected walls and print debug statements.
    
    Args:
    x (int): The x-coordinate of the current cell.
    y (int): The y-coordinate of the current cell.
    direction (int): The direction of the wall (0 = NORTH, 1 = EAST, 2 = SOUTH, 3 = WEST).
    has_wall (bool): True if there is a wall, False otherwise.
    horizontal_walls (list): 2D list representing horizontal walls.
    vertical_walls (list): 2D list representing vertical walls.
    """
    if direction == 0:  # NORTH
        if has_wall:
            horizontal_walls[y + 1][x] = 1
            API.setWall(x, y, 'n')
            log(f"Added wall in cell ({x}, {y}, N)")
            if valid_position(x, y + 1, 16, 16):
                API.setWall(x, y + 1, 's')
                log(f"Added wall in cell ({x}, {y + 1}, S)")
    elif direction == 1:  # EAST
        if has_wall:
            vertical_walls[y][x + 1] = 1
            API.setWall(x, y, 'e')
            log(f"Added wall in cell ({x}, {y}, E)")
            if valid_position(x + 1, y, 16, 16):
                API.setWall(x + 1, y, 'w')
                log(f"Added wall in cell ({x + 1}, {y}, W)")
    elif direction == 3:  # WEST
        if has_wall:
            vertical_walls[y][x] = 1
            API.setWall(x, y, 'w')
            log(f"Added wall in cell ({x}, {y}, W)")
            if valid_position(x - 1, y, 16, 16):
                API.setWall(x - 1, y, 'e')
                log(f"Added wall in cell ({x - 1}, {y}, E)")

def scan_and_update_walls(x, y, horizontal_walls, vertical_walls):
    """
    Scan for walls in the current cell and update the internal map with print statements.
    
    Args:
    x (int): The x-coordinate of the current cell.
    y (int): The y-coordinate of the current cell.
    horizontal_walls (list): 2D list representing horizontal walls.
    vertical_walls (list): 2D list representing vertical walls.
    """
    directions = [0, 1, 3]  # NORTH, EAST, WEST
    for direction in directions:
        has_wall = check_wall(direction)
        update_walls(x, y, direction, has_wall, horizontal_walls, vertical_walls)

def can_move(x, y, direction, horizontal_walls, vertical_walls):
    """
    Check if the mouse can move from the current position in the given direction.
    
    Args:
    x (int): The x-coordinate of the current cell.
    y (int): The y-coordinate of the current cell.
    direction (int): The direction of the movement (0 = NORTH, 1 = EAST, 2 = SOUTH, 3 = WEST).
    horizontal_walls (list): 2D list representing horizontal walls.
    vertical_walls (list): 2D list representing vertical walls.
    
    Returns:
    bool: True if the move is possible, False otherwise.
    """
    width, height = 16, 16  # Fixed size for the maze
    
    if direction == 0:  # NORTH
        if valid_position(x, y + 1, width, height):
            can_move_north = horizontal_walls[y + 1][x] == 0
            log(f"Checking NORTH: can move to ({x}, {y + 1}): {can_move_north}")
            return can_move_north
        return False
    elif direction == 1:  # EAST
        if valid_position(x + 1, y, width, height):
            can_move_east = vertical_walls[y][x + 1] == 0
            log(f"Checking EAST: can move to ({x + 1}, {y}): {can_move_east}")
            return can_move_east
        return False
    elif direction == 2:  # SOUTH
        if valid_position(x, y - 1, width, height):
            can_move_south = horizontal_walls[y][x] == 0
            log(f"Checking SOUTH: can move to ({x}, {y - 1}): {can_move_south}")
            return can_move_south
        return False
    elif direction == 3:  # WEST
        if valid_position(x - 1, y, width, height):
            can_move_west = vertical_walls[y][x] == 0
            log(f"Checking WEST: can move to ({x - 1}, {y}): {can_move_west}")
            return can_move_west
        return False
    return False

def get_accessible_neighbors(x, y, horizontal_walls, vertical_walls):
    """
    Get accessible neighboring cells from the current position using the can_move function.
    
    Args:
    x (int): The x-coordinate of the current cell.
    y (int): The y-coordinate of the current cell.
    horizontal_walls (list): 2D list representing horizontal walls.
    vertical_walls (list): 2D list representing vertical walls.
    
    Returns:
    list: A list of accessible neighboring cells as (x, y) tuples.
    """
    neighbors = []
    
    for direction in range(4):
        if can_move(x, y, direction, horizontal_walls, vertical_walls):
            if direction == 0:  # NORTH
                neighbors.append((x, y + 1))
            elif direction == 1:  # EAST
                neighbors.append((x + 1, y))
            elif direction == 2:  # SOUTH
                neighbors.append((x, y - 1))
            elif direction == 3:  # WEST
                neighbors.append((x - 1, y))
    
    return neighbors

def move_to_lowest_neighbor(x, y, maze, horizontal_walls, vertical_walls):
    """
    Move to the neighboring cell with the lowest value in the maze.
    
    Args:
    x (int): The x-coordinate of the current cell.
    y (int): The y-coordinate of the current cell.
    maze (list): The 2D list representing the maze distances.
    horizontal_walls (list): 2D list representing horizontal walls.
    vertical_walls (list): 2D list representing vertical walls.
    
    Returns:
    tuple: The coordinates of the new cell after moving.
    """
    neighbors = get_accessible_neighbors(x, y, horizontal_walls, vertical_walls)
    lowest_value = float('inf')
    next_x, next_y = x, y
    
    for nx, ny in neighbors:
        if maze[ny][nx] < lowest_value:
            lowest_value = maze[ny][nx]
            next_x, next_y = nx, ny

    log(f"Moving from ({x}, {y}) to ({next_x}, {next_y}) with value {lowest_value}")

    if next_x == x and next_y == y + 1:
        API.moveForward()
    elif next_x == x + 1 and next_y == y:
        API.turnRight()
        API.moveForward()
        API.turnLeft()
    elif next_x == x and next_y == y - 1:
        API.turnLeft()
        API.turnLeft()
        API.moveForward()
        API.turnLeft()
        API.turnLeft()
    elif next_x == x - 1 and next_y == y:
        API.turnLeft()
        API.moveForward()
        API.turnRight()

    return next_x, next_y

def print_walls(horizontal_walls, vertical_walls):
    """
    Print the internal wall representation for debugging purposes.
    
    Args:
    horizontal_walls (list): 2D list representing horizontal walls.
    vertical_walls (list): 2D list representing vertical walls.
    """
    log("Horizontal Walls:")
    for row in horizontal_walls[::-1]:
        log(" ".join(map(str, row)))
    
    log("Vertical Walls:")
    for row in vertical_walls[::-1]:
        log(" ".join(map(str, row)))

def run_flood_fill():
    """
    Main function to run the flood fill algorithm.
    """
    width, height = 16, 16  # Fixed size for the maze
    maze = [[float('inf')] * width for _ in range(height)]
    horizontal_walls = [[0] * width for _ in range(height + 1)]
    vertical_walls = [[0] * (width + 1) for _ in range(height)]

    # Initialize goal cells in the maze
    goal_cells = [(7, 7), (8, 7), (7, 8), (8, 8)]
    for gx, gy in goal_cells:
        maze[gy][gx] = 0

    # Initialize the borders of the maze
    for i in range(width):
        horizontal_walls[0][i] = 1  # Bottom border
        horizontal_walls[height][i] = 1  # Top border
    for i in range(height):
        vertical_walls[i][0] = 1  # Left border
        vertical_walls[i][width] = 1  # Right border

    flood_fill(maze, width, height)  # Perform the flood fill algorithm

    # Start scanning and updating walls from the start position
    x, y = 0, 0
    while (x, y) not in goal_cells:
        log(f"Scanning and updating walls at ({x}, {y})")
        scan_and_update_walls(x, y, horizontal_walls, vertical_walls)
        
        log(f"Determining next move from ({x}, {y})")
        x, y = move_to_lowest_neighbor(x, y, maze, horizontal_walls, vertical_walls)
        log(f"Moved to ({x}, {y})")
        
        # Print the distance map and wall maps after each move
        log("Distance map:")
        for row in maze[::-1]:
            log(" ".join([str(cell) for cell in row]))
        
        print_walls(horizontal_walls, vertical_walls)

    # Print the final distance map for debugging
    log("Final distance map:")
    for row in maze[::-1]:
        log(" ".join([str(cell) for cell in row]))

    # Print the final wall maps for debugging
    print_walls(horizontal_walls, vertical_walls)

if __name__ == "__main__":
    run_flood_fill()
