import API
import sys
import heapq

# Constants for directions
NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3
direction_map = {NORTH: 'n', EAST: 'e', SOUTH: 's', WEST: 'w'}
current_orientation = NORTH

# Mouse position
x, y = 0, 0

# Maze dimensions
maze_width, maze_height = 6, 6

# Wall data structures to track where walls are in the maze
horizontal_walls = [[0] * maze_width for _ in range(maze_height + 1)]
vertical_walls = [[0] * (maze_width + 1) for _ in range(maze_height)]

# Set to track which cells have been explored
explored_cells = set()

# Function to log messages for debugging purposes
def log(string):
    sys.stderr.write("{}\n".format(string))
    sys.stderr.flush()

# Functions to control the mouse's movement and update its orientation

def turn_left():
    global current_orientation
    API.turnLeft()
    current_orientation = (current_orientation - 1) % 4
    log(f"Turned left. New orientation: {current_orientation}")

def turn_right():
    global current_orientation
    API.turnRight()
    current_orientation = (current_orientation + 1) % 4
    log(f"Turned right. New orientation: {current_orientation}")

def turn_around():
    global current_orientation
    API.turnLeft()
    API.turnLeft()
    current_orientation = (current_orientation + 2) % 4
    log(f"Turned around. New orientation: {current_orientation}")

# Function to move the mouse forward and update its position in the maze
def move_forward():
    global x, y, current_orientation
    log(f"Attempting to move forward. Current position: ({x}, {y}), orientation: {current_orientation}")
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
    log("----------------------------------------------------------------")
    log(f"Moved forward to ({x}, {y}). Updated position: ({x}, {y})")
    log(f"New orientation: {current_orientation}")


class LexicographicPriority:
    def __init__(self, primary_key, secondary_key):
        self.primary_key = primary_key
        self.secondary_key = secondary_key

    def __lt__(self, other):
        if self.primary_key < other.primary_key:
            return True
        if self.primary_key == other.primary_key:
            return self.secondary_key < other.secondary_key
        return False

    def __le__(self, other):
        if self.primary_key < other.primary_key:
            return True
        if self.primary_key == other.primary_key:
            return self.secondary_key <= other.secondary_key
        return False

    def __eq__(self, other):
        return self.primary_key == other.primary_key and self.secondary_key == other.secondary_key
    def __repr__(self):
            return f"LexicographicPriority(primary_key={self.primary_key}, secondary_key={self.secondary_key})"


class QueueNode:
    def __init__(self, priority, vertex):
        self.priority = priority
        self.vertex = vertex

    def __lt__(self, other):
        return self.priority < other.priority

    def __le__(self, other):
        return self.priority <= other.priority

    def __eq__(self, other):
        return self.priority == other.priority


# Class to represent the D* Lite algorithm
class DStarLite:
    def __init__(self, start, goals, graph):
        # Initialize the D* Lite algorithm with a start point, goal points, and a graph of the maze
        self.start = start
        self.goals = goals
        self.graph = graph
        self.k_m = 0  # Km is a heuristic adjustment factor
        # Initialize g and rhs values for all nodes in the graph
        self.g = {v: float('inf') for v in graph.get_all_nodes()}
        self.rhs = {v: float('inf') for v in graph.get_all_nodes()}
        # Priority queue to manage nodes to be processed
        self.priority_queue = PriorityQueue()

        # Initialize the algorithm with the start and goal points
        self.initialize(start, goals)

    # Initialize the D* Lite algorithm
    def initialize(self, start, goals):
        log("Initializing D* Lite")
        for goal in goals:
            # Set the rhs value of the goal nodes to 0 (known cost to reach goal is zero)
            self.rhs[goal] = 0
            # Add the goal nodes to the priority queue with their calculated key
            self.priority_queue.insert(goal, self.calculate_key(goal))

        log("D* Lite initialization complete.")
        for goal in goals:
            log(f"g[{goal}] = {self.g[goal]}")
            log(f"rhs[{goal}] = {self.rhs[goal]}")
            log(f"Added {goal} to the priority queue with key {self.calculate_key(goal)}")
            # Update the simulator with the rhs value at each goal location
            API.setText(goal[0], goal[1], str(int(self.rhs[goal])))
            log(f"Post-init check - rhs[{goal}] = {self.rhs[goal]}")

    # Calculate the priority key for a given vertex in the graph
    def calculate_key(self, vertex):
        g_rhs_min = min(self.g[vertex], self.rhs[vertex])
        return LexicographicPriority(g_rhs_min + self.heuristic(self.start, vertex) + self.k_m, g_rhs_min)

    # Heuristic function that uses Manhattan distance
    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def update_vertex(self, vertex):
        # Ensure vertex is a tuple and exists in g and rhs
        if not isinstance(vertex, tuple) or vertex not in self.g:
            log(f"Error: Vertex {vertex} is not valid or not initialized.")
            return

        log(f"Updating vertex {vertex}. Initial g = {self.g[vertex]}, rhs = {self.rhs[vertex]}")

        if vertex not in self.goals:
            # Recalculate rhs for this vertex based on its neighbors
            old_rhs = self.rhs[vertex]
            self.rhs[vertex] = min(
                [self.g[neighbor] + self.graph.cost(vertex, neighbor) for neighbor in self.graph.get_all_neighbors(vertex)]
            )
            log(f"Recalculated rhs for vertex {vertex}. Old rhs = {old_rhs}, New rhs = {self.rhs[vertex]}")

        # Check if the vertex is in the priority queue before removing it
        if vertex in self.priority_queue.vertex_set:
            log(f"Removing vertex {vertex} from the priority queue.")
            self.priority_queue.remove(vertex)

        # If g != rhs, reinsert with updated priority
        if self.g[vertex] != self.rhs[vertex]:
            key = self.calculate_key(vertex)
            self.priority_queue.insert(vertex, key)
            log(f"Reinserted vertex {vertex} with new key = ({key.primary_key}, {key.secondary_key}) into the priority queue.")
            log(f"Updated vertex {vertex}: g = {self.g[vertex]}, rhs = {self.rhs[vertex]}")
        else:
            log(f"No update required for vertex {vertex}: g = {self.g[vertex]}, rhs = {self.rhs[vertex]}")

        # Log the current state of the priority queue
        self.priority_queue.log_queue_state()

        # Update the display with the new rhs value in the simulator
        if self.rhs[vertex] == float('inf'):
            API.setText(vertex[0], vertex[1], "inf")
        else:
            API.setText(vertex[0], vertex[1], str(int(self.rhs[vertex])))


    def compute_shortest_path(self):
        log("Starting to compute the shortest path...")
        
        while not self.priority_queue.is_empty() and (self.priority_queue.peek_priority() < self.calculate_key(self.start) or self.rhs[self.start] != self.g[self.start]):
            u = self.priority_queue.extract_min()
            log(f"Popping node {u} with g[{u}] = {self.g[u]} and rhs[{u}] = {self.rhs[u]} from the priority queue")
            self.priority_queue.log_queue_state()

            if self.g[u] > self.rhs[u]:  # Overconsistent case
                log(f"Node {u} is overconsistent. g[{u}] > rhs[{u}]. Updating g[{u}] to rhs[{u}]")
                self.g[u] = self.rhs[u]  # Relax the g value
                for s in self.graph.get_accessible_neighbors(u):  # Update neighbors
                    log("----------------------------------------------------------------")
                    log(f"Updating neighbor {s} of node {u} before rhs[{s}] = {self.rhs[s]}")
                    self.update_vertex(s)
                    log(f"Updated neighbor {s} of node {u} after rhs[{s}] = {self.rhs[s]}")
                    log("----------------------------------------------------------------")
            else:  # Underconsistent case
                log(f"Node {u} is underconsistent. Setting g[{u}] to infinity.")
                self.g[u] = float('inf')
                self.update_vertex(u)
                for s in self.graph.get_accessible_neighbors(u):
                    log(f"Updating neighbor {s} of node {u} before rhs[{s}] = {self.rhs[s]}")
                    self.update_vertex(s)
                    log(f"Updated neighbor {s} of node {u} after rhs[{s}] = {self.rhs[s]}")

            log(f"Finished processing node {u} with new g[{u}] = {self.g[u]} and rhs[{u}] = {self.rhs[u]}")
            log(f"=======================================================================================")

        log("Shortest path computation complete.")
        
        # Display the g and rhs values and highlight the priority queue cells in the simulator
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
        if v not in self.graph[u]:
            self.graph[u].append(v)
        if u not in self.graph[v]:
            self.graph[v].append(u)

    # Remove an edge between two cells
    def remove_edge(self, u, v):
        if v in self.graph[u]:
            self.graph[u].remove(v)
        if u in self.graph[v]:
            self.graph[v].remove(u)

    def get_accessible_neighbors(self, node):
        # Returns only neighbors that are connected via an open path (no wall)
        return self.graph[node]

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
        self.vertex_set.remove(vertex)
        for index, node in enumerate(self.heap):
            if node.vertex == vertex:
                self.heap[index] = self.heap[-1]
                self.heap.pop()
                heapq.heapify(self.heap)
                break

    def update(self, vertex, priority):
        self.remove(vertex)
        self.insert(vertex, priority)

    def is_empty(self):
        return len(self.heap) == 0

    def log_queue_state(self):
        log("Current Priority Queue:")
        for node in sorted(self.heap, key=lambda x: (x.priority.primary_key, x.priority.secondary_key)):
            log(f"  Vertex: {node.vertex}, Priority: ({node.priority.primary_key}, {node.priority.secondary_key})")


def move_and_replan(start_position):
    """
    Navigate from the current position towards the goal, updating the plan as obstacles are encountered.
    """
    path = [start_position]
    current_position = start_position
    last_position = current_position
    DStarLite.compute_shortest_path()

    while current_position not in DStarLite.goals:
        # Ensure there's a valid path
        assert (DStarLite.rhs[current_position] != float('inf')), "No known path to the goal!"

        # Find the neighbor with the lowest cost
        neighbors = DStarLite.graph.get_all_neighbors(current_position)
        next_position = min(neighbors, key=lambda pos: DStarLite.graph.cost(current_position, pos) + DStarLite.g[pos])

        # Move to the best next position
        current_position = next_position
        path.append(current_position)

        # Scan the environment for any changes (e.g., obstacles)
        changed_edges = []
        for neighbor in neighbors:
            old_cost = DStarLite.graph.cost(last_position, current_position)
            new_cost = DStarLite.graph.cost(current_position, neighbor)
            if old_cost != new_cost:
                changed_edges.append((last_position, current_position))
                changed_edges.append((current_position, neighbor))

        # If any edges have changed, update the cost estimates
        if changed_edges:
            DStarLite.k_m += DStarLite.heuristic(last_position, current_position)
            last_position = current_position

            # Re-evaluate all vertices affected by the edge cost changes
            for (u, v) in changed_edges:
                new_cost = DStarLite.graph.cost(u, v)
                if u != DStarLite.goals:
                    DStarLite.rhs[u] = min(DStarLite.rhs[u], new_cost + DStarLite.g[v])
                elif DStarLite.rhs[u] == new_cost + DStarLite.g[v]:
                    if u != DStarLite.goals:
                        min_cost = float('inf')
                        neighbor_nodes = DStarLite.graph.get_all_neighbors(u)
                        for neighbor in neighbor_nodes:
                            temp_cost = DStarLite.graph.cost(u, neighbor) + DStarLite.g[neighbor]
                            if min_cost > temp_cost:
                                min_cost = temp_cost
                        DStarLite.rhs[u] = min_cost
                    DStarLite.update_vertex(u)

        # Recompute the shortest path with the updated information
        DStarLite.compute_shortest_path()

    log("Path to goal found!")
    return path

def scan_and_update_walls(x, y, dstar_lite):
    global explored_cells
    explored_cells.add((x, y))
    directions = [0, 1, 3]  # NORTH, EAST, WEST
    log(f"Scanning walls at ({x}, {y}) with orientation {current_orientation}")
    
    wall_detected = False
    updated_vertices = set()

    for direction in directions:
        has_wall = check_wall(direction)
        log(f"Checking wall in direction {direction}: {has_wall}")
        
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
            
        if valid_position(nx, ny, maze_width, maze_height):
            if has_wall:
                wall_detected = True
                dstar_lite.graph.remove_edge((x, y), (nx, ny))
                log(f"Removed edge between ({x}, {y}) and ({nx, ny}) due to wall.")
                API.setWall(x, y, direction_map[actual_direction])
                updated_vertices.add((x, y))
            else:
                dstar_lite.graph.add_edge((x, y), (nx, ny))
                log(f"Added edge between ({x}, {y}) and ({nx, ny}) - no wall detected.")
                updated_vertices.add((x, y))

    if wall_detected:
        log("Wall detected; recalculating shortest path.")
        for u in updated_vertices:
            log(f"Updating vertex {u}.") 
            dstar_lite.update_vertex(u)
            # Use get_all_neighbors to ensure all potential neighbors are considered
            neighbors = dstar_lite.graph.get_all_neighbors(u)
            for s in neighbors:
                log(f"Updating neighbor {s} of vertex {u}.")  
                dstar_lite.update_vertex(s)
        dstar_lite.compute_shortest_path()



# Function to check if there is a wall in a specified direction
def check_wall(direction):
    log(f"Checking wall. Current orientation: {current_orientation}, direction: {direction}")
    if direction == 0:  # Front
        return API.wallFront()
    elif direction == 1:  # Right
        return API.wallRight()
    elif direction == 3:  # Left
        return API.wallLeft()

# Function to validate if a given coordinate is within the maze bounds
def valid_position(x, y, width, height):
    return 0 <= x < width and 0 <= y < height

# Function to determine the next move based on the shortest path calculation
def move_to_next(x, y, graph, g, dstar_lite):
    neighbors = graph.get_accessible_neighbors((x, y))
    log(f"Evaluating neighbors for move from ({x}, {y}): {neighbors}")

    lowest_g = float('inf')
    next_x, next_y = x, y

    # Find the neighbor with the lowest g value
    for nx, ny in neighbors:
        log(f"Neighbor ({nx}, {ny}) has g value {g[(nx, ny)]}")
        if g[(nx, ny)] < lowest_g:
            lowest_g = g[(nx, ny)]
            next_x, next_y = nx, ny

    # If all neighbors have higher g values, the mouse should not move there
    if lowest_g >= g[(x, y)]:
        log(f"All neighbors have higher or equal g values. Replanning required.")
        dstar_lite.compute_shortest_path()
        return x, y  # Replan and stay in the same position

    log(f"Next move determined: from ({x}, {y}) to ({next_x}, {next_y}) with g value {lowest_g}")

    # Determine the direction to move
    if next_x == x and next_y == y + 1:
        target_orientation = NORTH
    elif next_x == x + 1 and next_y == y:
        target_orientation = EAST
    elif next_x == x and next_y == y - 1:
        target_orientation = SOUTH
    elif next_x == x - 1 and next_y == y:
        target_orientation = WEST
    else:
        log(f"Unexpected move: from ({x}, {y}) to ({next_x}, {next_y}), which is not adjacent.")
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

    return next_x, next_y


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
            if (x, y) in priority_cells:
                API.setColor(x, y, 'y')  # Highlight the cell in yellow


def run_d_lite_6():
    try:
        global x, y, current_orientation

        width, height = 6, 6  # Fixed size for the maze
        start = (0, 0)
        goals = [(2, 2), (3, 2), (2, 3), (3, 3)]  # Multiple goal cells

        # Initialize the MazeGraph
        graph = MazeGraph(width, height)

        # Initialize the D* Lite algorithm
        dstar_lite = DStarLite(start, goals, graph)

        # Compute the initial shortest path
        log("Running D* Lite to calculate the initial shortest path.")
        dstar_lite.compute_shortest_path()

        # Main movement loop
        while (x, y) not in goals:
            log(f"Starting loop with mouse at ({x}, {y}), current orientation: {current_orientation}.")

            # Step 1: Scan for walls
            log(f"Mouse at ({x}, {y}), orientation: {current_orientation}. Scanning for walls.")
            scan_and_update_walls(x, y, dstar_lite)

            # Step 2: Recompute the shortest path if necessary
            if dstar_lite.priority_queue.peek_priority() < dstar_lite.calculate_key(start):
                log("Wall detected; recalculating shortest path.")
                dstar_lite.compute_shortest_path()

            # Step 3: Determine and execute the next move
            log(f"Determining and executing the next move from ({x}, {y}).")
            x, y = move_to_next(x, y, graph, dstar_lite.g, dstar_lite)

            # Display the state of g and rhs values in the simulator
            show(dstar_lite.g, dstar_lite.rhs, dstar_lite.priority_queue)

        log("Mouse has reached the goal.")

    except Exception as e:
        log(f"Error during D* Lite execution: {e}")
        raise  # Re-raise the exception for further investigation


# Entry point of the program
if __name__ == "__main__":
    try:
        log("Starting D* Lite algorithm...")
        run_d_lite_6()
        log("Finished running D* Lite algorithm.")
    except Exception as e:
        log(f"Error in main execution: {e}")