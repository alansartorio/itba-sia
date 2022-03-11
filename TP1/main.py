from cube import \
    Cube, solved_cubes, turns, cube_rotations, \
    apply_action, apply_algorythm, parse_algorythm

from tree import \
    Tree, Node


cube = tuple(solved_cubes)[0]

n = Node(str(cube))


cube = apply_algorythm(cube, parse_algorythm("R"))
print(cube.is_solved())
print(str(cube))
print()


s = Node(cube.clone())
n.add_child(s)


cube = apply_algorythm(cube, parse_algorythm("R'"))
print(cube.is_solved())
print(str(cube))
print()

t = Node(cube.clone())
s.add_child(t)

cube = apply_algorythm(
    cube, parse_algorythm("R U R' U' R' F R2 U' R' U' R U R' F'"))
print(cube.is_solved())
print(str(cube))
print()

r = Node(cube.clone())
t.add_child(r)

t = Tree(n)
t.bfs()
t.dfs(t.root)

