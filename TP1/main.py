from solution import Solution
from cube import \
    Cube, generate_scrambled, solved_cubes, turns, cube_rotations, \
    apply_action, apply_algorythm, parse_algorythm

from tree import \
    HeuristicNode, HeuristicTree, Tree, Node

from heuristics import sticker_groups
import sys
sys.setrecursionlimit(20000)


cube = generate_scrambled(5)

t = HeuristicTree(HeuristicNode(cube, sticker_groups, True), True)
# node = t.bpa()
# node = t.bpp()
# node = t.bppv(8)
# node = t.global_heuristic()
node = t.global_heuristic()

if node:
    for state, action in node.get_branch():
        if action is not None:
            print(f"""
    |
    {action}
    |
""")
        print(str(state))

    with open('solution.txt', 'w') as file:
        Solution(node).save(file)
