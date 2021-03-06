from solution import Solution
from cube import \
    Cube, generate_scrambled, solved_cubes, turns, cube_rotations, \
    apply_action, apply_algorythm, parse_algorythm

from tree import \
    HeuristicNode, HeuristicTree, Tree, Node

from heuristics import sticker_groups, move_count_combination
import sys
# sys.setrecursionlimit(20000)

import time

cube = generate_scrambled(10)

tic = time.time()

t = Tree(Node(cube), Cube.is_solved)
# node = t.bpa()
# node = t.bpp()
# node = t.bpp_non_recursive()
# node = t.bppv(8)
# node = t.global_heuristic()
node = t.bppv(2)

toc = time.time()

if node:
    for node in node.get_branch():
        if node.action is not None:
            print(f"""
    |
    {node.action}
    |
""")
        # print(node.heuristic)
        print(str(node.state))

    print("Process time: " + str(toc-tic))    

    with open('solution.txt', 'w') as file:
        Solution(node).save(file)
