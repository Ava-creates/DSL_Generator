import numpy as np
import time
import collections
import env_factory
import craft
import env
def collect(env, primitive):    # Step 1: Extract the current state from the environment
  import numpy as np
  from collections import deque

  # Get the index of the target primitive
  primitive_index = env.world.cookbook.index[primitive]

  # Helper function to check if a cell is blocked
  def is_blocked(x, y):
    return any(env._current_state.grid[x, y] > 0)

  # Directions and their corresponding actions
  directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
  actions = {(-1, 0): craft.LEFT, (1, 0): craft.RIGHT, 
             (0, -1): craft.DOWN, (0, 1): craft.UP}

  # BFS to find the shortest path to a cell containing the primitive
  start_x, start_y = env._current_state.pos
  visited = set()
  queue = deque([(start_x, start_y, [])])  # (x, y, path)
  
  while queue:
    x, y, path = queue.popleft()
    
    if (x, y) in visited:
      continue
    
    visited.add((x, y))
    
    for dx, dy in directions:
      nx, ny = x + dx, y + dy
      
      # Check bounds
      if 0 <= nx < env._current_state.grid.shape[0] and 0 <= ny < env._current_state.grid.shape[1]:
        # If the cell contains the primitive, return the path to it
        if env._current_state.grid[nx, ny][primitive_index] > 0:
          move_action = actions[(dx, dy)]
          use_action = craft.USE
          return path + [move_action, use_action]
        
        # If the cell is not blocked and hasn't been visited, add to queue
        if not is_blocked(nx, ny):
          move_action = actions[(dx, dy)]
          queue.append((nx, ny, path + [move_action]))
  
  # If no path found, return an empty list (though this should not happen in a valid scenario)
  return []
