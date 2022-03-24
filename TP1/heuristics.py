
from cube import Cube, apply_action, solved_cubes, cube_rotations, all_rotations
from collections import Counter


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


def manhattan_distance(cube: Cube):
    def find_3d_position(id):
        return ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))[id]

    def manhattan_distance_3d(p1, p2):
        return abs(p1[0] - p2[0]) +\
            abs(p1[1] - p2[1]) +\
            abs(p1[2] - p2[2])

    def find_orientation(cube: Cube):
        def is_oriented():
            return cube.pieces[0].id == 0 and cube.pieces[0].orientation == 0
        for cube in all_rotations(cube):
            if is_oriented():
                return cube
    cube = find_orientation(cube)  # type: ignore
    assert cube is not None

    dist = []
    for position, piece in enumerate(cube.pieces):
        desired_position = piece.id
        d = manhattan_distance_3d(find_3d_position(position), find_3d_position(desired_position))
        dist.append(d)

    return max(max(dist), 1) if not cube.is_solved() else 0
