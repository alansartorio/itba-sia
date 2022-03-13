from cube import \
    Cube, generate_scrambled, solved_cubes, turns, cube_rotations, \
    apply_action, apply_algorythm, parse_algorythm

from tree import \
    Tree, Node

import sys
sys.setrecursionlimit(20000)


cube = generate_scrambled(3)

t = Tree(Node(cube))
# node = t.bpa()
# node = t.bpp()
# node = t.bppv(8)
# node = t.global_heuristic()
node = t.local_heuristic()

if node:
    for state, action in node.get_branch():
        if action is not None:
            print(f"""
    |
    {action}
    |
""")
        print(str(state))

