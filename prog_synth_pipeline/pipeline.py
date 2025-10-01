from funsearch.implementation.funsearch import FunSearch
from funsearch.implementation import config as config_lib
from cfg_parser import CFGParser
import funsearch

funsearch = FunSearch(
    model_type='ollama',
    model_path=""
)

with open('specification.txt', 'r') as f:
    specification = f.read()

#parse cfg
cfg_parser = CFGParser('cfg.txt')
terminal_functions = [func_name for func_name, _ in cfg_parser.get_terminal_functions()]

config = config_lib.Config()

#implement each terminal function
for function_name in terminal_functions:
    if(function_name == "move"):
        inputs = ["UP", "DOWN", "LEFT", "RIGHT"]
    elif(function_name == "has"):
        inputs=["wood"]
    else:
        inputs = [3]
    funsearch.run(
        specification=specification,
        inputs=inputs,
        config=config,
        function_to_implement=function_name
    )

#program synthesis 