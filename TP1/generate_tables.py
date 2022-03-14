from typing import Any, Iterable
from cube import Cube, Piece
from tree import Tree, Node
from itertools import permutations, combinations
from multiprocessing import Pool
from tqdm import tqdm
from more_itertools import take
from math import factorial


def all_orientations(count=8):
    if count == 0:
        yield ()
        return
    for a in all_orientations(count - 1):
        for i in range(3):
            yield a + (i,)


def all_positions():
    return permutations(range(8))


def moves_to_solve_orientations(orientations):
    cube = Cube(tuple(Piece(0, o) for o in orientations))  # type: ignore
    tree = Tree(Node(cube), Cube.are_orientations_solved)
    sol = tree.bpa()

    return orientations, sol.get_depth() if sol is not None else None


def map_to_positions(cube: Cube) -> str:
    return cube.get_state_string()[::2]


def moves_to_solve_positions(positions):
    cube = Cube(tuple(Piece(id, 0) for id in positions))  # type: ignore

    tree = Tree(Node(cube), Cube.are_positions_solved, map_to_positions)
    sol = tree.bpa()

    return positions, sol.get_depth() if sol is not None else None


def calculate(iter: Iterable, length: int, func: Any, file_name: str):
    with Pool(12) as p,\
            open(file_name, 'w') as file:
        for a, steps in tqdm(p.imap(func, iter), total=length):
            if steps is None:
                continue
            s = ''.join(map(str, a)) + '=' + str(steps) + '\n'
            file.write(s)


calculate(all_positions(), factorial(8),
          moves_to_solve_positions, "positions_table.txt")
# calculate(take(20, all_orientations()), 20, moves_to_solve_orientations, "orientations_table.txt")
