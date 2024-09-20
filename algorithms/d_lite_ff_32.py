import API
import sys
import heapq
from collections import deque
import tracemalloc

import random

# Constants for directions
NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3
direction_map = {NORTH: 'n', EAST: 'e', SOUTH: 's', WEST: 'w'}
current_orientation = NORTH

# Mouse position
x, y = 0, 0

# Maze dimensions
maze_width, maze_height = 32, 32

# Wall data structures to track where walls are in the maze
horizontal_walls = [[0] * maze_width for _ in range(maze_height + 1)]
vertical_walls = [[0] * (maze_width + 1) for _ in range(maze_height)]

# Set to track which cells have been explored
explored_cells = set()

initial_run_cells = 0
return_run_cells = 0
final_run_cells = 0

# Function to log messages for debugging purposes
def log(string, skip=False):
    if not skip:
        sys.stderr.write("{}\n".format(string))
        sys.stderr.flush()

# Functions to control the mouse's movement and update its orientation

def turn_left():
    global current_orientation
    API.turnLeft()
    current_orientation = (current_orientation - 1) % 4
    # log(f"Turned left. New orientation: {current_orientation}")

def turn_right():
    global current_orientation
    API.turnRight()
    current_orientation = (current_orientation + 1) % 4
    # log(f"Turned right. New orientation: {current_orientation}")

def turn_around():
    global current_orientation
    API.turnLeft()
    API.turnLeft()
    current_orientation = (current_orientation + 2) % 4
    # log(f"Turned around. New orientation: {current_orientation}")

# Function to move the mouse forward and update its position in the maze
def move_forward():
    global x, y, current_orientation
    # log(f"Attempting to move forward. Current position: ({x}, {y}), orientation: {current_orientation}")
    API.moveForward()

    # Update the mouse's position based on its current orientation
    if current_orientation == NORTH:
        y += 1
    elif current_orientation == EAST:
        x += 1
    elif current_orientation == SOUTH:
        y -= 1
    elif current_orientation == WEST:
        x -= 1
    # log("----------------------------------------------------------------")
    # log(f"Moved forward to ({x}, {y}). Updated position: ({x}, {y})")
    # log(f"New orientation: {current_orientation}")


class LexicographicPriority:
    def __init__(self, primary_key, secondary_key):
        self.primary_key = primary_key
        self.secondary_key = secondary_key
        self.v = (primary_key, secondary_key)

    def __lt__(self, other):
        if self.primary_key == other.primary_key:
            return self.secondary_key < other.secondary_key
        return self.primary_key < other.primary_key

    def __repr__(self):
        return f"(primary_key={self.primary_key}, secondary_key={self.secondary_key})"



class QueueNode:
    def __init__(self, priority, vertex):
        self.priority = priority
        self.vertex = vertex

    def __lt__(self, other):
        return self.priority < other.priority


# Class to represent the D* Lite algorithm
class DStarLite:
    def __init__(self, start, goals, graph):
        # Initialize the D* Lite algorithm with a start point, goal points, and a graph of the maze
        self.start = start
        self.goals = goals
        self.graph = graph
        self.k_m = 0  # Km is a heuristic adjustment factor
        self.g =    {v: float('inf') for v in graph.get_all_nodes()}
        self.rhs =  {v: float('inf') for v in graph.get_all_nodes()}
        self.priority_queue = PriorityQueue()
        self.initialize(start, goals)

    # Initialize the D* Lite algorithm
    def initialize(self, start, goals):
        log("Initializing D* Lite")
        for goal in goals:
            # Set the rhs value of the goal nodes to 0 (known cost to reach goal is zero)
            self.rhs[goal] = 0
            # Add the goal nodes to the priority queue with their calculated key
            heuristic_value = self.heuristic(self.start, goal)
            key = LexicographicPriority(heuristic_value, 0)
            self.priority_queue.insert(goal, key)

        log("D* Lite initialization complete.")
        for goal in goals:
            # log(f"g[{goal}] = {self.g[goal]}")
            # log(f"rhs[{goal}] = {self.rhs[goal]}")
            # log(f"Added {goal} to the priority queue with key {self.calculate_key(goal)}")
            # Update the simulator with the rhs value at each goal location
            API.setText(goal[0], goal[1], str(int(self.rhs[goal])))
            # log(f"Post-init check - rhs[{goal}] = {self.rhs[goal]}")

    # Calculate the priority key for a given vertex in the graph
    def calculate_key(self, vertex):
        g_rhs_min = min(self.g[vertex], self.rhs[vertex])
        heuristic_value = self.heuristic(self.start, vertex)
        key = LexicographicPriority(g_rhs_min + heuristic_value + self.k_m, g_rhs_min)
        
        # log(f"Calculating key for {vertex}:")
        # log(f"  g[{vertex}] = {self.g[vertex]}, rhs[{vertex}] = {self.rhs[vertex]}")
        # log(f"  Heuristic from start to {vertex}: {heuristic_value}")
        # log(f"  Key = (primary: {key.primary_key}, secondary: {key.secondary_key})")
        
        return key


    # Heuristic function that uses Manhattan distance
    def heuristic(self, a, b):
        x1, y1 = a
        x2, y2 = b
        return abs(x2 - x1) + abs(y2 - y1)
    
    def update_vertex(self, vertex, restrict_to_explored=False):
        if vertex not in self.goals:
            neighbors = self.graph.get_accessible_neighbors(vertex, restrict_to_explored)
            new_rhs = float('inf')
            for neighbor in neighbors:
                if self.graph.is_connected(vertex, neighbor):
                    cost = self.graph.cost(vertex, neighbor)
                    candidate_rhs = self.g[neighbor] + cost
                    if candidate_rhs < new_rhs:
                        new_rhs = candidate_rhs
            self.rhs[vertex] = new_rhs

        # Update priority queue logic as before...

        # log(f"values = {self.g[vertex]} and {self.rhs[vertex]}")

        if vertex in self.priority_queue.vertex_set:
            # log(f"YEP 2")
            self.priority_queue.remove(vertex)

        if self.g[vertex] != self.rhs[vertex]:
            # log(f"YEP 3")
            key = self.calculate_key(vertex)
            self.priority_queue.insert(vertex, key)

        if self.rhs[vertex] == float('inf'):
            API.setText(vertex[0], vertex[1], "inf")
        else:
            API.setText(vertex[0], vertex[1], str(int(self.rhs[vertex])))

    def compute_shortest_path(self, restrict_to_explored=False):
        # log(f"Length of queue = {len(self.priority_queue.heap)}")
        while (self.priority_queue.peek_priority() < self.calculate_key(self.start)) or (self.rhs[self.start] != self.g[self.start]):
            kold = self.priority_queue.peek_priority()
            u = self.priority_queue.extract_min()
            if kold < self.calculate_key(u):
                self.priority_queue.insert(u, self.calculate_key(u))
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for s in self.graph.get_parents(u):
                    self.update_vertex(s, restrict_to_explored)
            else:
                self.g[u] = float('inf')
                self.update_vertex(u, restrict_to_explored)
                for s in self.graph.get_parents(u):
                    self.update_vertex(s, restrict_to_explored)

        show(self.g, self.rhs, self.priority_queue)


class MazeGraph:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.graph = {}
        self.initialize_graph()
        self.initialize_full_connections()

    # Initialize the graph structure
    def initialize_graph(self):
        for x in range(self.width):
            for y in range(self.height):
                self.graph[(x, y)] = []

    # Create full connections between neighboring cells
    def initialize_full_connections(self):
        for x in range(self.width):
            for y in range(self.height):
                if y < self.height - 1:
                    self.add_edge((x, y), (x, y + 1))  # Connect to north
                if y > 0:
                    self.add_edge((x, y), (x, y - 1))  # Connect to south
                if x < self.width - 1:
                    self.add_edge((x, y), (x + 1, y))  # Connect to east
                if x > 0:
                    self.add_edge((x, y), (x - 1, y))  # Connect to west

    # Add an edge between two cells
    def add_edge(self, u, v):
        not_exists = v not in self.graph[u]
        if not_exists:        
            self.graph[u].append(v)
            self.graph[v].append(u)
        return not_exists

    # Remove an edge between two cells
    def remove_edge(self, u, v):
        exists = v in self.graph[u]
        if exists:
            self.graph[u] = [i for i in self.graph[u] if i != v]
            self.graph[v] = [i for i in self.graph[v] if i != u]
        return exists
    
    def get_accessible_neighbors(self, node, restrict_to_explored=False):
        neighbors = self.graph[node]
        if restrict_to_explored:
            return [neighbor for neighbor in neighbors if neighbor in explored_cells]
        return neighbors

    def get_parents(self, node):
        parents = []
        for u in self.graph.keys():
            if node in self.graph[u]:
                parents.append(u)
        return parents

    def get_all_neighbors(self, node):
        # Returns all neighbors in all four directions, regardless of current walls
        x, y = node
        potential_neighbors = []
        if y < self.height - 1:
            potential_neighbors.append((x, y + 1))  # NORTH
        if y > 0:
            potential_neighbors.append((x, y - 1))  # SOUTH
        if x < self.width - 1:
            potential_neighbors.append((x + 1, y))  # EAST
        if x > 0:
            potential_neighbors.append((x - 1, y))  # WEST
        return potential_neighbors

    # Check if two cells are connected
    def is_connected(self, u, v):
        return v in self.graph[u]

    # Get the cost of moving between two cells
    def cost(self, u, v):
        if not self.is_connected(u, v):
            return float('inf')
        return 1

    # Get all nodes in the graph
    def get_all_nodes(self):
        return list(self.graph.keys())


class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.vertex_set = set()

    def peek(self):
        return self.heap[0].vertex

    def peek_priority(self):
        if len(self.heap) == 0:
            return LexicographicPriority(float('inf'), float('inf'))
        return self.heap[0].priority

    def extract_min(self):
        item = heapq.heappop(self.heap)
        self.vertex_set.remove(item.vertex)
        return item.vertex


    def insert(self, vertex, priority):
        if vertex in self.vertex_set:
            self.update(vertex, priority)
        else:
            heapq.heappush(self.heap, QueueNode(priority, vertex))
            self.vertex_set.add(vertex)

    def remove(self, vertex):
        if vertex in self.vertex_set:
            self.vertex_set.remove(vertex)
            for index, node in enumerate(self.heap):
                if node.vertex == vertex:
                    # Replace the node with the last element
                    self.heap[index] = self.heap[-1]
                    self.heap.pop()
                    # Restore the heap property
                    heapq.heapify(self.heap)
                    break

    def update(self, vertex, priority):
        if vertex in self.vertex_set:
            # Find the vertex in the heap
            for index, node in enumerate(self.heap):
                if node.vertex == vertex:
                    self.heap[index].priority = priority
                    # Restore the heap property
                    heapq.heapify(self.heap)
                    break
        else:
            self.insert(vertex, priority)

    def is_empty(self):
        return len(self.heap) == 0

    # def log_queue_state(self):
    #     log("Current Priority Queue:")
    #     for node in sorted(self.heap, key=lambda x: (x.priority.primary_key, x.priority.secondary_key)):
    #         log(f"  Vertex: {node.vertex}, Priority: ({node.priority.primary_key}, {node.priority.secondary_key})")

def scan_and_update_walls(start, last, dstar_lite):
    x, y = start
    global explored_cells
    explored_cells.add((x, y))
    directions = [0, 1, 3]  # NORTH, EAST, WEST
    # directions = [0, 1, 2, 3]  # NORTH, EAST, WEST
    # log(f"Scanning walls at ({x}, {y}) with orientation {current_orientation}")
    
    wall_detected = False
    updated_vertices = set()

    graph_changed = False

    for direction in directions:
        has_wall = check_wall(direction)
        
        nx, ny = None, None
        actual_direction = (current_orientation + direction) % 4
        
        if actual_direction == NORTH: 
            nx, ny = x, y + 1
        elif actual_direction == EAST: 
            nx, ny = x + 1, y
        elif actual_direction == SOUTH: 
            nx, ny = x, y - 1
        elif actual_direction == WEST: 
            nx, ny = x - 1, y

        # log(f"Checking wall in direction {direction}: {has_wall}, coords = {(nx, ny)}")

        if valid_position(nx, ny, maze_width, maze_height):
            if has_wall:
                wall_detected = True
                if dstar_lite.graph.remove_edge((x, y), (nx, ny)):
                    graph_changed = True
                # log(f"Removing edge: {(x, y)} and {(nx, ny)} and {graph_changed}")
                API.setWall(x, y, direction_map[actual_direction])
                updated_vertices.add((x, y))
            else:
                if dstar_lite.graph.add_edge((x, y), (nx, ny)):
                    graph_changed = True
                # log(f"Added edge between ({x}, {y}) and ({nx, ny}) - no wall detected.")
                updated_vertices.add((x, y))

    # log(f"Wall has been detected: {wall_detected}")

    if graph_changed:
        dstar_lite.k_m += dstar_lite.heuristic(last, start)
        dstar_lite.update_vertex(start)
        for u in dstar_lite.graph.get_all_neighbors(start):
            dstar_lite.update_vertex(u)
        dstar_lite.compute_shortest_path()
        return start
    else:
        return last

# Function to check if there is a wall in a specified direction
def check_wall(direction):
    # log(f"Checking wall. Current orientation: {current_orientation}, direction: {direction}")
    if direction == 0:  # Front
        return API.wallFront()
    elif direction == 1:  # Right
        return API.wallRight()
    # elif direction == 2:
    #     return API.wallBack()
    elif direction == 3:  # Left
        return API.wallLeft()

# Function to validate if a given coordinate is within the maze bounds
def valid_position(x, y, width, height):
    return 0 <= x < width and 0 <= y < height

# Move to next position in the simulator
def mms_move_to(prev_position, next_position):
    x, y = prev_position
    next_x, next_y = next_position
    if next_x == x and next_y == y + 1:
        target_orientation = NORTH
    elif next_x == x + 1 and next_y == y:
        target_orientation = EAST
    elif next_x == x and next_y == y - 1:
        target_orientation = SOUTH
    elif next_x == x - 1 and next_y == y:
        target_orientation = WEST
    else:
        # log(f"Unexpected move: from ({x}, {y}) to ({next_x}, {next_y}), which is not adjacent.")
        return x, y  # Abort move if something is wrong

    # Adjust the mouse's orientation to face the target direction
    while current_orientation != target_orientation:
        if (target_orientation - current_orientation) % 4 == 1:
            turn_right()
        elif (target_orientation - current_orientation) % 4 == 3:
            turn_left()
        elif (target_orientation - current_orientation) % 4 == 2:
            turn_around()

    # Move forward after turning
    move_forward()

def find_next_position(s_start, graph, g, dstar_lite, restrict_to_explored=False):
    neighbors = graph.get_accessible_neighbors((x, y), restrict_to_explored)
    # log(f"Evaluating neighbors for move from ({x}, {y}): {neighbors}")

    lowest_g = float('inf')
    next_x, next_y = x, y

    # Find the neighbor with the lowest g value
    for nx, ny in neighbors:
        # log(f"Neighbor ({nx}, {ny}) has g value {g[(nx, ny)]}")
        if g[(nx, ny)] < lowest_g:
            # Ensure the neighbor is adjacent before selecting it
            if abs(nx - x) + abs(ny - y) == 1:  # Manhattan distance of 1
                lowest_g = g[(nx, ny)]
                next_x, next_y = nx, ny

    # if (next_x, next_y) == (x, y):
    #     log("No valid adjacent move found, staying in place.")
    
    return (next_x, next_y)



def show(g, rhs, priority_queue=None):
    max_x = max([coord[0] for coord in g.keys()])
    max_y = max([coord[1] for coord in g.keys()])
    
    # Get all the cells currently in the priority queue
    priority_cells = set(item.vertex for item in priority_queue.heap) if priority_queue else set()

    for y in range(max_y, -1, -1):  # Start from max_y down to 0 to display from top to bottom
        for x in range(max_x + 1):  # Start from 0 to max_x
            g_value = g.get((x, y), float('inf'))
            rhs_value = rhs.get((x, y), float('inf'))
            
            # Condense g and rhs values into a single string
            g_str = "i" if g_value == float('inf') else str(int(g_value))
            rhs_str = "i" if rhs_value == float('inf') else str(int(rhs_value))
            cell_text = f"g{g_str}r{rhs_str}"
            API.setText(x, y, cell_text)
            
            # Highlight cells that are in the priority queue
            # if (x, y) in priority_cells:
            #     API.setColor(x, y, 'y')  # Highlight the cell in yellow
            
def mark_unexplored_cells_as_walls(graph):
    global maze_width, maze_height

    # Loop through each cell in the maze
    for x in range(maze_width):
        for y in range(maze_height):
            if (x, y) not in explored_cells:
                # Get all neighbors of the unexplored cell
                neighbors = graph.get_all_neighbors((x, y))

                # Set walls around the unexplored cell by removing edges to all its neighbors
                for neighbor in neighbors:
                    if graph.remove_edge((x, y), neighbor):
                        # Calculate the direction of the neighbor relative to the current cell
                        nx, ny = neighbor
                        if ny == y + 1:
                            API.setWall(x, y, 'n')  # North wall
                        elif ny == y - 1:
                            API.setWall(x, y, 's')  # South wall
                        elif nx == x + 1:
                            API.setWall(x, y, 'e')  # East wall
                        elif nx == x - 1:
                            API.setWall(x, y, 'w')  # West wall

                log(f"Marked unexplored cell ({x}, {y}) as a wall by removing edges and setting API walls.")

def flood_fill(graph, goals, maze_width, maze_height):
    # Initialize flood values
    flood_values = [[float('inf')] * maze_width for _ in range(maze_height)]

    # Create a queue for BFS (starting from the goal)
    queue = deque()
    for goal in goals:
        queue.append(goal)
        flood_values[goal[1]][goal[0]] = 0  # Mark goal cells with distance 0

    # Perform BFS to flood fill the maze
    while queue:
        x, y = queue.popleft()
        current_value = flood_values[y][x]

        # Get neighbors and flood them
        neighbors = graph.get_accessible_neighbors((x, y))
        for nx, ny in neighbors:
            if flood_values[ny][nx] == float('inf'):  # If the cell has not been flooded yet
                flood_values[ny][nx] = current_value + 1
                queue.append((nx, ny))

    return flood_values

def run_flood_fill_final_path(graph, start, goals, maze_width, maze_height):
    # Perform flood fill from the goal
    flood_values = flood_fill(graph, goals, maze_width, maze_height)

    # Start backtracking from the initial start position
    x, y = start
    path_length = 0

    # While not at the goal
    while (x, y) not in goals:
        # Get neighbors
        neighbors = graph.get_accessible_neighbors((x, y))
        # Find the neighbor with the smallest flood value
        next_x, next_y = min(neighbors, key=lambda pos: flood_values[pos[1]][pos[0]])

        # Mark the path (green for the final run)
        API.setColor(next_x, next_y, 'g')

        # Move to the next position
        x, y = next_x, next_y
        path_length += 1

    log(f"Final path length using flood fill: {path_length}")

# Initialize counters for each phase
initial_run_cells = 0
return_run_cells = 0
final_run_cells = 0

def run_d_lite_ff_32():
    try:
        tracemalloc.start()
        global x, y, current_orientation
        global initial_run_cells, return_run_cells, final_run_cells

        width, height = 32, 32
        initial_start_position = (0, 0)  # Initial start point
        start = initial_start_position
        x, y = start

        # Multiple goal cells
        goals = [(15, 15), (16, 15), (15, 16), (16, 16)]

        log(f"Goals = {goals}", False)

        # Initialize the MazeGraph
        graph = MazeGraph(width, height)

        # Initialize the D* Lite algorithm
        dstar_lite = DStarLite(start, goals, graph)

        # Compute the initial shortest path using D* Lite (first phase)
        log("Running D* Lite to calculate the initial shortest path.")
        dstar_lite.compute_shortest_path()

        # Initially scan for walls before making the first move
        last = start
        last = scan_and_update_walls(start, last, dstar_lite)

        # Main movement loop to reach one of the goals (first phase - initial run)
        while start not in goals:

            if dstar_lite.g[start] == float("inf"):
                log("No path exists to the goal.")
                break

            # Find the next position and move the mouse
            prev = start
            start = find_next_position(start, graph, dstar_lite.g, dstar_lite)
            dstar_lite.start = start
            mms_move_to(prev, start)

            # Mark cell as yellow for the initial run
            API.setColor(start[0], start[1], 'y')  # Yellow color

            # Update mouse position
            x, y = start

            # Increment the counter for the initial run
            initial_run_cells += 1

            # Scan for walls and update the graph
            last = scan_and_update_walls(start, last, dstar_lite)

            # Display the current g and rhs values in the simulator
            show(dstar_lite.g, dstar_lite.rhs, dstar_lite.priority_queue)

        log("Mouse has reached the goal.")

        # ===========================
        # Switch to reverse journey (second phase - return run)
        # ===========================
        if start in goals:
            log("Switching to return journey.")
            dstar_lite.goals = [initial_start_position]  # Reverse the target to the initial start
            dstar_lite.start = start  # New start is the current goal

            # Reset g and rhs values for all nodes
            for node in dstar_lite.graph.get_all_nodes():
                dstar_lite.g[node] = float('inf')
                dstar_lite.rhs[node] = float('inf')
            dstar_lite.rhs[initial_start_position] = 0  # Set the rhs value for the new goal (initial start)

            # Reinitialize and compute the path for the return journey
            dstar_lite.initialize(dstar_lite.start, dstar_lite.goals)
            dstar_lite.compute_shortest_path()

        # Movement loop for the return journey (second phase - return run)
        while start != initial_start_position:

            if dstar_lite.g[start] == float("inf"):
                log("No path exists to return to the start.")
                break

            # Find the next position and move the mouse
            prev = start
            start = find_next_position(start, graph, dstar_lite.g, dstar_lite)
            dstar_lite.start = start
            mms_move_to(prev, start)

            # Mark cell as blue for the return run
            API.setColor(start[0], start[1], 'b')  # Blue color

            # Update mouse position
            x, y = start

            # Increment the counter for the return run
            return_run_cells += 1

            # Scan for walls and update the graph
            last = scan_and_update_walls(start, last, dstar_lite)

            # Display the current g and rhs values in the simulator
            show(dstar_lite.g, dstar_lite.rhs, dstar_lite.priority_queue)

        log("Mouse has returned to the initial start.")

        # ===========================
        # Final run to the goal (flood fill, final optimized run)
        # ===========================
        log("Final run to the goal using flood fill.")
        dstar_lite.goals = goals  # Set goals back to the original goals
        dstar_lite.start = initial_start_position  # Reset start to the initial start

        # Mark all unexplored cells as walls
        mark_unexplored_cells_as_walls(dstar_lite.graph)

        # Perform flood fill from the goal
        flood_values = flood_fill(dstar_lite.graph, goals, maze_width, maze_height)

        # Display the flood values on the grid
        for y in range(maze_height):
            for x in range(maze_width):
                if flood_values[y][x] != float('inf'):
                    API.setText(x, y, str(flood_values[y][x]))  # Display the flood fill value

        # Initialize path length counter
        path_length = 0

        # Movement loop for the final run (backtracking using flood fill)
        while start not in goals:

            # Get neighbors
            neighbors = dstar_lite.graph.get_accessible_neighbors(start)

            # Find the neighbor with the smallest flood value
            next_x, next_y = min(neighbors, key=lambda pos: flood_values[pos[1]][pos[0]])

            # Move the mouse to the next position
            prev = start
            start = (next_x, next_y)
            mms_move_to(prev, start)

            # Mark cell as green for the final run
            API.setColor(start[0], start[1], 'g')  # Green color

            # Increment path length
            path_length += 1

            # Increment the counter for the final run
            final_run_cells += 1

        log(f"Final path length using flood fill: {path_length}")
        log("Mouse has completed the final run to the goal.")

        # ===========================
        # Retrieve and log API statistics at the end
        # ===========================
        log_final_stats()

    except Exception as e:
        log(f"Error during D* Lite execution: {e}")
        raise  # Re-raise the exception for further investigation


def log_final_stats():
    """
    Log both manually tracked and API-retrieved statistics after the final run.
    """
    # API-based statistics
    stats = {
        "total-distance": API.getStat("total-distance"),
        "total-turns": API.getStat("total-turns"),
        "best-run-distance": API.getStat("best-run-distance"),
        "best-run-turns": API.getStat("best-run-turns"),
        "current-run-distance": API.getStat("current-run-distance"),
        "current-run-turns": API.getStat("current-run-turns"),
        "total-effective-distance": API.getStat("total-effective-distance"),
        "best-run-effective-distance": API.getStat("best-run-effective-distance"),
        "current-run-effective-distance": API.getStat("current-run-effective-distance"),
        "score": API.getStat("score")
    }
    
    # Log API-based statistics
    for key, value in stats.items():
        log(f"{key}: {value}")
    
    # Log manually tracked statistics for cell traversals
    log(f"Cells traversed in initial run: {initial_run_cells}")
    log(f"Cells traversed in return run: {return_run_cells}")
    log(f"Cells traversed in final run: {final_run_cells}")

    # Retrieve and log memory usage statistics
    current, peak = tracemalloc.get_traced_memory()
    log(f"Current memory usage: {current / 10**6} MB; Peak memory usage: {peak / 10**6} MB")

    # Stop tracking memory usage after logging
    tracemalloc.stop()


# Entry point of the program
if __name__ == "__main__":
    try:
        log("Starting D* Lite algorithm with flood fill final run...")
        run_d_lite_ff_32()
        log("Finished running D* Lite algorithm with flood fill final run.")
    except Exception as e:
        log(f"Error in main execution: {e}")
