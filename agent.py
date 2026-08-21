# agent.py
import random
from collections import deque
import heapq


class SimpleReflexAgent:
    """A simple reflex agent using purely IF-THEN rules."""

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here'):
            return 'Stay'
        elif percept.get('wall_ahead'):
            return 'Up'
        else:
            return 'Right'


class ModelBasedAgent:
    """A model-based agent that uses internal memory (state) to escape loops."""

    def __init__(self):
        self.last_action = None
        self.stuck_count = 0

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here'):
            self.last_action = 'Stay'
            return 'Stay'

        if percept.get('wall_ahead'):
            self.stuck_count += 1
            if self.stuck_count == 1:
                self.last_action = 'Up'
            elif self.stuck_count == 2:
                self.last_action = 'Left'
            elif self.stuck_count == 3:
                self.last_action = 'Down'
            else:
                self.last_action = 'Right'
            return self.last_action
        else:
            self.stuck_count = 0
            if self.last_action in ['Up', 'Down', 'Left', 'Right']:
                return self.last_action
            else:
                self.last_action = 'Right'
                return 'Right'


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        pos = percept.get('agent_pos')
        return random.choice(self.actions_pool)


class SearchAgent:
    """A Goal-Based / Planning Agent supporting BFS, DFS, and UCS."""

    def __init__(self, active_algo: str = 'BFS'):
        self.plan = []
        self.active_algo = active_algo

    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        """Breadth-First Search using a FIFO queue."""
        start = tuple(start_pos)
        goal = tuple(goal_pos)
        width, height = grid_size
        walls_set = set(tuple(w) for w in walls)

        if start == goal:
            return []

        frontier = deque([(start, [])])
        visited = {start}

        actions = [
            ('Up', (0, 1)),
            ('Down', (0, -1)),
            ('Left', (-1, 0)),
            ('Right', (1, 0))
        ]

        while frontier:
            curr_pos, path = frontier.popleft()

            if curr_pos == goal:
                return path

            for action, (dx, dy) in actions:
                next_pos = (curr_pos[0] + dx, curr_pos[1] + dy)
                if 0 <= next_pos[0] < width and 0 <= next_pos[1] < height:
                    if next_pos not in walls_set and next_pos not in visited:
                        visited.add(next_pos)
                        frontier.append((next_pos, path + [action]))

        return []

    def dfs_search(self, start_pos, goal_pos, walls, grid_size):
        """Depth-First Search using a LIFO stack."""
        start = tuple(start_pos)
        goal = tuple(goal_pos)
        width, height = grid_size
        walls_set = set(tuple(w) for w in walls)

        if start == goal:
            return []

        frontier = [(start, [])]
        visited = set()

        actions = [
            ('Up', (0, 1)),
            ('Down', (0, -1)),
            ('Left', (-1, 0)),
            ('Right', (1, 0))
        ]

        while frontier:
            curr_pos, path = frontier.pop()

            if curr_pos == goal:
                return path

            if curr_pos in visited:
                continue
            visited.add(curr_pos)

            for action, (dx, dy) in actions:
                next_pos = (curr_pos[0] + dx, curr_pos[1] + dy)
                if 0 <= next_pos[0] < width and 0 <= next_pos[1] < height:
                    if next_pos not in walls_set and next_pos not in visited:
                        frontier.append((next_pos, path + [action]))

        return []

    def ucs_search(self, start_pos, goal_pos, walls, grid_size):
        """Uniform-Cost Search using a Priority Queue."""
        start = tuple(start_pos)
        goal = tuple(goal_pos)
        width, height = grid_size
        walls_set = set(tuple(w) for w in walls)

        if start == goal:
            return []

        count = 0
        frontier = [(0, count, start, [])]
        heapq.heapify(frontier)
        best_cost = {start: 0}

        actions = [
            ('Up', (0, 1)),
            ('Down', (0, -1)),
            ('Left', (-1, 0)),
            ('Right', (1, 0))
        ]

        while frontier:
            cost, _, curr_pos, path = heapq.heappop(frontier)

            if curr_pos == goal:
                return path

            if cost > best_cost.get(curr_pos, float('inf')):
                continue

            for action, (dx, dy) in actions:
                next_pos = (curr_pos[0] + dx, curr_pos[1] + dy)
                if 0 <= next_pos[0] < width and 0 <= next_pos[1] < height:
                    if next_pos not in walls_set:
                        next_cost = cost + 1
                        if next_cost < best_cost.get(next_pos, float('inf')):
                            best_cost[next_pos] = next_cost
                            count += 1
                            heapq.heappush(frontier, (next_cost, count, next_pos, path + [action]))

        return []

    def sense_and_act(self, percept: dict) -> str:
        """Executes offline plan step-by-step or computes a new plan to the closest food."""
        if not self.plan:
            all_food = percept.get('all_food', [])
            if not all_food:
                return 'Stay'

            agent_pos = tuple(percept.get('agent_pos', (0, 0)))
            walls = percept.get('walls', [])
            grid_size = percept.get('grid_size', (10, 10))

            # Find closest food pellet using Manhattan distance
            closest_food = min(
                all_food,
                key=lambda f: abs(f[0] - agent_pos[0]) + abs(f[1] - agent_pos[1])
            )

            # Execute the search method matching self.active_algo
            if self.active_algo == 'BFS':
                self.plan = self.bfs_search(agent_pos, closest_food, walls, grid_size)
            elif self.active_algo == 'DFS':
                self.plan = self.dfs_search(agent_pos, closest_food, walls, grid_size)
            elif self.active_algo == 'UCS':
                self.plan = self.ucs_search(agent_pos, closest_food, walls, grid_size)
            else:
                self.plan = self.bfs_search(agent_pos, closest_food, walls, grid_size)

        if self.plan:
            return self.plan.pop(0)

        return 'Stay'