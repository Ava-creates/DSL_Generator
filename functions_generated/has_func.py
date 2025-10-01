def has(env, item):
        count = env._current_state.inventory[item]
        return count > 0 