from cube import Cube, solved_cubes, turns, cube_rotations, apply_action, apply_algorythm, parse_algorythm

cube = tuple(solved_cubes)[0]


cube = apply_algorythm(cube, parse_algorythm("R"))
print(cube.is_solved())
print(str(cube))
print()

cube = apply_algorythm(cube, parse_algorythm("R'"))
print(cube.is_solved())
print(str(cube))
print()

cube = apply_algorythm(cube, parse_algorythm("R U R' U' R' F R2 U' R' U' R U R' F'"))
print(cube.is_solved())
print(str(cube))
print()
