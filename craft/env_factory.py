"""Factory to sample new Craft environments."""

from __future__ import division
from __future__ import print_function

import collections
import json
import numpy as np
import yaml

from . import craft, env
from .misc import util

Task = collections.namedtuple("Task", ["goal", "steps"])


class EnvironmentFactory(object):
  """Factory instantiating Craft environments."""

  def __init__(self,
               recipes_path,
               hints_path,
               env_type,
               max_steps=300,
               seed=0,
               visualise=False,
               reuse_environments=False,
               custom_grid_path=None):
    self.subtask_index = util.Index()
    self.task_index = util.Index()
    self._max_steps = max_steps
    self._visualise = visualise
    self._reuse_environments = reuse_environments

    # Per task, we reuse the same environment, with same layouts.
    # Should generates much easier tasks where agents can overfit.
    if self._reuse_environments:
      self._env_cache = {}

    # create World
    self.world = craft.CraftWorld(recipes_path, env_type, seed)

    # Optional: load a custom grid spec for deterministic scenarios
    self._custom_grid_spec = None
    if custom_grid_path:
      with open(custom_grid_path, "r", encoding="utf-8") as f:
        self._custom_grid_spec = json.load(f)

    # Load the tasks with sub-steps (== hints)
    with open(hints_path) as hints_f:
      self.hints = yaml.load(hints_f, Loader=yaml.FullLoader)

    # Setup all possible tasks
    self._init_tasks()

  def _init_tasks(self):
    """Build the list of tasks and subtasks."""
    # organize task and subtask indices
    self.tasks_by_subtask = collections.defaultdict(list)
    self.tasks = {}
    for hint_key, hint in self.hints.items():
      # hint_key: make[plank], hint/steps: get_wood, makeAtToolshed
      goal = util.parse_fexp(hint_key)
      # goal: (make, plank)
      goal = (self.subtask_index.index(goal[0]),
              self.world.cookbook.index[goal[1]])
      steps = tuple(self.subtask_index.index(s) for s in hint)
      task = Task(goal, steps)
      for subtask in steps:
        self.tasks_by_subtask[subtask].append(task)

      self.tasks[hint_key] = task
      self.task_index.index(task)

    self.task_names = sorted(self.tasks.keys())

    if self._reuse_environments:
      # Trying to handle random seed weirdness by preallocating everything.
      for task_name in self.task_names:
        self.sample_environment(task_name)

  def _create_environment(self, task_name):
    # Get the task
    
    task = self.tasks[task_name]
    # print(self.tasks, "printing out th etasks")
    goal_arg = task.goal[1]

    # Sample a world (== scenario for them...) or use a custom scenario
    if self._custom_grid_spec is not None:
      scenario = craft.scenario_from_spec(self._custom_grid_spec, self.world)
    else:
      scenario = self.world.sample_scenario_with_goal(goal_arg)

    # Wrap it into an environment and return
    return env.CraftLab(
        scenario,
        task_name,
        task,
        max_steps=self._max_steps,
        visualise=self._visualise)

  def sample_environment(self, task_name=None):
    if task_name is None:
      task_name = np.random.choice(self.task_names)

    if self._reuse_environments:
      return self._env_cache.setdefault(task_name,
                                        self._create_environment(task_name))
    else:
      return self._create_environment(task_name)
