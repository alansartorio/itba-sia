from solution import Solution
from cube import \
    Cube, generate_scrambled, solved_cubes, turns, cube_rotations, \
    apply_action, apply_algorythm, parse_algorythm

from tree import \
    HeuristicNode, HeuristicTree, Tree, Node

from heuristics import sticker_groups, move_count_combination
import sys
sys.setrecursionlimit(20000)


cube = generate_scrambled(20)

t = HeuristicTree(HeuristicNode(cube, move_count_combination, True), True, Cube.is_solved)
# node = t.bpa()
# node = t.bpp()
# node = t.bppv(8)
node = t.global_heuristic()
# node = t.local_heuristic()

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
