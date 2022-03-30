from config import Config
import time
from solution import Solution
from cube import \
    Cube, solved_cubes, turns, cube_rotations, \
    apply_action, apply_algorythm, parse_algorythm

from tree import \
    HeuristicNode, HeuristicTree, Tree, Node

from heuristics import sticker_groups, move_count_combination
import sys
# sys.setrecursionlimit(20000)

config = Config.parse()

tic = time.time()
solution = config.method(config.state)
toc = time.time()

if solution:
    for node in solution.solution.get_branch():
        if node.action is not None:
            print(f"""
    |
    {node.action}
    |
""")
        # print(node.heuristic)
        print(str(node.state))

    processing_time = toc - tic
    print("Process time: " + str(processing_time))

    with \
            open('solution.json', 'w') as file,\
            open('visualization.txt', 'w') as visualization_file:
        Solution(config, node, solution.expanded_count,
                 solution.border_count, processing_time).save(file, visualization_file)
