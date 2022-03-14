
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




def load_file(file_name: str):
    with open(file_name, 'r') as file:
        for line in file.readlines():
            state, steps = line.split('=')
            yield (state, int(steps))

orientations = dict(load_file('orientations_table.txt'))
positions = dict(load_file('positions_table.txt'))

def move_count_combination(cube: Cube):
    state = cube.get_state_string()
    position_state = state[::2]
    orientation_state = state[1::2]
    return max(orientations[orientation_state], positions[position_state])
