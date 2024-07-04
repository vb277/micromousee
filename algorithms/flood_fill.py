# flood_fill.py

import API
import sys
from collections import deque

def log(string):
    """
    Log messages for debugging.
    """
    sys.stderr.write("{}\n".format(string))
    sys.stderr.flush()

def init_maze(width, height):
    """
    Initialize the maze with all cells set to infinity (indicating they are not reachable yet),
    except for the goal cells which are set to 0 (distance to themselves is 0).
    
    Args:
    width (int): The width of the maze.
    height (int): The height of the maze.
    
    Returns:
    list: A 2D list representing the initialized maze.
    """
    maze = [[float('inf')] * width for _ in range(height)]
    # Define the goal cells
    goal_cells = [(7, 7), (8, 7), (7, 8), (8, 8)]
    for gx, gy in goal_cells:
        maze[gy][gx] = 0  # Set distance to goal cells as 0
    return maze

def print_maze(maze):
    """
    Print the maze for debugging purposes. The maze is printed from top to bottom.
    
    Args:
    maze (list): The 2D list representing the maze.
    """
    for row in maze[::-1]:  # Reverse the rows for correct visualization (top to bottom)
        log(" ".join([str(cell) for cell in row]))

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

    print_maze(maze)  # Print the maze for debugging

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
        # No direct method to check back, assuming it is checked as part of movement or another strategy
        return False  # Placeholder, should be handled differently in actual implementation
    elif direction == 3:  # WEST
        return API.wallLeft()

def update_walls(x, y, direction, has_wall, width, height):
    """
    Update the internal map with the detected walls and print debug statements.
    
    Args:
    x (int): The x-coordinate of the current cell.
    y (int): The y-coordinate of the current cell.
    direction (int): The direction of the wall (0 = NORTH, 1 = EAST, 2 = SOUTH, 3 = WEST).
    has_wall (bool): True if there is a wall, False otherwise.
    width (int): The width of the maze.
    height (int): The height of the maze.
    """
    if direction == 0:  # NORTH
        if has_wall:
            API.setWall(x, y, 'n')
            log(f"Added wall in cell ({x}, {y}, N)")
            if valid_position(x, y + 1, width, height):
                API.setWall(x, y + 1, 's')
                log(f"Added wall in cell ({x}, {y + 1}, S)")
    elif direction == 1:  # EAST
        if has_wall:
            API.setWall(x, y, 'e')
            log(f"Added wall in cell ({x}, {y}, E)")
            if valid_position(x + 1, y, width, height):
                API.setWall(x + 1, y, 'w')
                log(f"Added wall in cell ({x + 1}, {y}, W)")
    elif direction == 3:  # WEST
        if has_wall:
            API.setWall(x, y, 'w')
            log(f"Added wall in cell ({x}, {y}, W)")
            if valid_position(x - 1, y, width, height):
                API.setWall(x - 1, y, 'e')
                log(f"Added wall in cell ({x - 1}, {y}, E)")

def scan_and_update_walls(x, y, width, height):
    """
    Scan for walls in the current cell and update the internal map with print statements.
    
    Args:
    x (int): The x-coordinate of the current cell.
    y (int): The y-coordinate of the current cell.
    width (int): The width of the maze.
    height (int): The height of the maze.
    """
    directions = [0, 1, 3]  # NORTH, EAST, WEST
    for direction in directions:
        has_wall = check_wall(direction)
        update_walls(x, y, direction, has_wall, width, height)

def run_flood_fill():
    """
    Main function to run the flood fill algorithm.
    """
    width = API.mazeWidth()  # Get the width of the maze
    height = API.mazeHeight()  # Get the height of the maze
    maze = init_maze(width, height)  # Initialize the maze

    flood_fill(maze, width, height)  # Perform the flood fill algorithm

    # Start scanning and updating walls from the start position
    x, y = 0, 0
    scan_and_update_walls(x, y, width, height)
    
    # Print the distance map for debugging
    print_maze(maze)

if __name__ == "__main__":
    run_flood_fill()
