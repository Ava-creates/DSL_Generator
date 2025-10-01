def move(env, dir):
  # print(dir)
  if dir == "UP":
    return 1
  elif dir == "DOWN":
    return 0
  elif dir == "LEFT":
    return 2
  elif dir == "RIGHT":
    return 3
  return 0