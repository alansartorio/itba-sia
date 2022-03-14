
from cube import Cube, solved_cubes, generate_scrambled
import itertools
from collections import Counter
from tree import Tree, Node


def sticker_groups(cube: Cube):
    group_points = {
        1: 20,
        2: 5,
        3: 2,
        4: 0,
    }
    faces = cube.colored_faces()
    return sum(
        sum(
            map(group_points.__getitem__, Counter(
                tile for row in face for tile in row).values())
        ) for face in faces
    )

