from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Iterable


class Colors(Enum):
    RED = 'r'
    ORANGE = 'o'
    WHITE = 'w'
    YELLOW = 'y'
    BLUE = 'b'
    GREEN = 'g'

    def __str__(self):
        return '\033[1;' + {
            Colors.RED: '31m',
            Colors.ORANGE: '36m',
            Colors.WHITE: '37m',
            Colors.YELLOW: '33m',
            Colors.BLUE: '34m',
            Colors.GREEN: '32m',
        }[self] + '█\033[0m'

    def default_position(self):
        return {
            Colors.RED: 'u',
            Colors.ORANGE: 'd',
            Colors.WHITE: 'r',
            Colors.YELLOW: 'l',
            Colors.BLUE: 'f',
            Colors.GREEN: 'b',
        }[self]


piece_colors = (
    (Colors.RED, Colors.WHITE, Colors.BLUE),
    (Colors.RED, Colors.WHITE, Colors.GREEN),
    (Colors.RED, Colors.YELLOW, Colors.BLUE),
    (Colors.RED, Colors.YELLOW, Colors.GREEN),

    (Colors.ORANGE, Colors.WHITE, Colors.BLUE),
    (Colors.ORANGE, Colors.WHITE, Colors.GREEN),
    (Colors.ORANGE, Colors.YELLOW, Colors.BLUE),
    (Colors.ORANGE, Colors.YELLOW, Colors.GREEN),
)


def rotate_3(a, b, c):
    return c, a, b


def invert_3(a, b, c):
    return a, c, b


class Piece:
    def __init__(self, id: int, orientation: int) -> None:
        self.id = id
        self.orientation = orientation

    @classmethod
    def parse(cls, s: str):
        return cls(int(s[0]), int(s[1]))

    def __repr__(self) -> str:
        return f'{self.id}{self.orientation}'

    def get_colors(self, invert: bool) -> tuple[Colors, Colors, Colors]:
        colors = piece_colors[self.id]
        if invert:
            colors = invert_3(*colors)
        for _ in range(self.orientation):
            colors = rotate_3(*colors)
        return colors


class Cube:
    def __init__(self, pieces: tuple[Piece, Piece, Piece, Piece, Piece, Piece, Piece, Piece]) -> None:
        self.pieces = list(pieces)

    def is_solved(self):
        return self in solved_cubes

    def are_positions_solved(self):
        ids = tuple(piece.id for piece in self.pieces)
        return ids in solved_positions

    def are_orientations_solved(self):
        orientations = (piece.orientation for piece in self.pieces)
        first = next(orientations)
        return all(o == first for o in orientations)

    @classmethod
    def parse(cls, s: str):
        assert len(s) == 16
        l = [Piece.parse(s[i:i+2]) for i in range(0, 16, 2)]
        return cls((l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7]))

    def get_state_string(self) -> str:
        return ''.join(map(repr, self.pieces))

    def __repr__(self) -> str:
        return self.get_state_string()

    def clone(self) -> 'Cube':
        return Cube.parse(repr(self))

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, str):
            return self.get_state_string() == __o
        if isinstance(__o, Cube):
            return self == __o.get_state_string()
        return False

    def __hash__(self) -> int:
        return hash(repr(self))

    def colored_faces(self):
        pc = []
        ccw = (1, 2, 4, 7)
        for i, piece in enumerate(self.pieces):
            invert = (i in ccw) != (piece.id in ccw)
            pc.append(piece.get_colors(invert))

        up = ((pc[3][0], pc[1][0]), (pc[2][0], pc[0][0]))
        down = ((pc[6][0], pc[4][0]), (pc[7][0], pc[5][0]))
        right = ((pc[0][1], pc[1][1]), (pc[4][1], pc[5][1]))
        left = ((pc[3][1], pc[2][1]), (pc[7][1], pc[6][1]))
        front = ((pc[2][2], pc[0][2]), (pc[6][2], pc[4][2]))
        back = ((pc[1][2], pc[3][2]), (pc[5][2], pc[7][2]))

        return (up, down, right, left, front, back)

    def get_faces(self):
        s = ""
        for face in self.colored_faces():
            a, b, c, d = (color.default_position() for row in face for color in row)
            s += f'{a}{b}\n{c}{d}\n\n'
        return s

    def __str__(self):
        (up, down, right, left, front, back) = self.colored_faces()
        faces = {
            'u': up,
            'd': down,
            'r': right,
            'l': left,
            'f': front,
            'b': back,
        }

        text = ["  uu    ",
                "  uu    ",
                "llffrrbb",
                "llffrrbb",
                "  dd    ",
                "  dd    "]
        text = [[c for c in row] for row in text]

        counters = {k: 0 for k in "udrlfb"}
        for y, row in enumerate(text):
            for x, c in enumerate(row):
                if c in counters:
                    count = counters[c]
                    text[y][x] = str(faces[c][count // 2][count % 2])
                    counters[c] += 1

        return "\n".join(["".join(row) for row in text])

class PositionsCube(Cube):
    def get_state_string(self) -> str:
        return super().get_state_string()[::2]


class Action(metaclass=ABCMeta):
    @abstractmethod
    def apply(self, cube: Cube) -> Cube: ...
    @abstractmethod
    def undo(self, cube: Cube) -> Cube: ...

    def reversed(self) -> 'ReversedAction':
        return ReversedAction(self)

    def doubled(self) -> 'DoubleAction':
        return DoubleAction(self)


class DoubleAction(Action):
    def __init__(self, action: Action):
        self.action = action

    def apply(self, cube: Cube) -> Cube:
        for _ in range(2):
            cube = self.action.apply(cube)
        return cube

    def undo(self, cube: Cube) -> Cube:
        for _ in range(2):
            cube = self.action.undo(cube)
        return cube


class ReversedAction(Action):
    def __init__(self, action: Action) -> None:
        self.action = action

    def apply(self, cube: Cube) -> Cube:
        return self.action.undo(cube)

    def undo(self, cube: Cube) -> Cube:
        return self.action.apply(cube)


class Turn(Action, metaclass=ABCMeta):
    def doubled(self) -> 'DoubleTurn':
        return DoubleTurn(self)

    def reversed(self) -> 'ReversedTurn':
        return ReversedTurn(self)


class SingleTurn(Turn):
    def __init__(self, cycle: tuple[int, int, int, int], orientation: int):
        self.cycle = cycle
        self.orientation = orientation

    def _apply_cycle(self, cycle: tuple[int, int, int, int], cube: Cube):
        cube = cube.clone()
        c0, c1, c2, c3 = cycle
        tmp = cube.pieces[c0]
        cube.pieces[c0] = cube.pieces[c3]
        cube.pieces[c3] = cube.pieces[c2]
        cube.pieces[c2] = cube.pieces[c1]
        cube.pieces[c1] = tmp
        for i in cycle:
            o = cube.pieces[i].orientation

            if o != self.orientation:
                o = (o + 1) % 3
                if o == self.orientation:
                    o = (o + 1) % 3
            cube.pieces[i].orientation = o
        return cube

    def apply(self, cube: Cube) -> Cube:
        return self._apply_cycle(self.cycle, cube)

    def undo(self, cube: Cube) -> Cube:
        c0, c1, c2, c3 = self.cycle
        return self._apply_cycle((c3, c2, c1, c0), cube)


class DoubleTurn(Turn, DoubleAction):
    def __init__(self, turn: Turn):
        super().__init__(turn)


class ReversedTurn(Turn, ReversedAction):
    def __init__(self, turn) -> None:
        super().__init__(turn)


class CubeRotation(Action):
    def doubled(self) -> 'DoubleCubeRotation':
        return DoubleCubeRotation(self)

    def reversed(self) -> 'ReversedCubeRotation':
        return ReversedCubeRotation(self)


class SingleCubeRotation(CubeRotation):
    def __init__(self, turns: tuple[Turn, Turn]):
        self.turns = turns

    def apply(self, cube: Cube) -> Cube:
        return self.turns[1].apply(self.turns[0].apply(cube))

    def undo(self, cube: Cube) -> Cube:
        return self.turns[1].undo(self.turns[0].undo(cube))


class DoubleCubeRotation(DoubleAction, CubeRotation):
    def __init__(self, cube_rotation: CubeRotation):
        super().__init__(cube_rotation)


class ReversedCubeRotation(CubeRotation, ReversedAction):
    def __init__(self, turn) -> None:
        super().__init__(turn)


def apply_action(cube: Cube, action: Action):
    return action.apply(cube)


base_turns = {
    'U': SingleTurn((2, 3, 1, 0), 0),
    'D': SingleTurn((4, 5, 7, 6), 0),
    'R': SingleTurn((0, 1, 5, 4), 1),
    'L': SingleTurn((6, 7, 3, 2), 1),
    'F': SingleTurn((0, 4, 6, 2), 2),
    'B': SingleTurn((3, 7, 5, 1), 2),
}

turns: dict[str, Turn] = {}

for notation, turn in list(base_turns.items()):
    turns[notation] = turn
    turns[notation + "'"] = turn.reversed()
    turns[notation + "2"] = turn.doubled()

base_cube_rotations = {
    'y': SingleCubeRotation((turns["D'"], turns["U"])),
    'x': SingleCubeRotation((turns["L'"], turns["R"])),
    'z': SingleCubeRotation((turns["B'"], turns["F"])),
}

cube_rotations: dict[str, CubeRotation] = {}

for notation, cube_rotation in list(base_cube_rotations.items()):
    cube_rotations[notation] = cube_rotation
    cube_rotations[notation + "'"] = cube_rotation.reversed()
    cube_rotations[notation + "2"] = cube_rotation.doubled()


def parse_algorythm(algo: str):
    return tuple(map((turns | cube_rotations).__getitem__, algo.split()))


def apply_algorythm(cube: Cube, alg: Iterable[Action]):
    for turn in alg:
        cube = apply_action(cube, turn)
    return cube

def all_rotations(original: Cube):
    rotations = [
        original,
        apply_action(original, cube_rotations['z']),
        apply_action(original, cube_rotations["x'"]),
    ]
    # Rotate 180° each solved state
    for cube in tuple(rotations):
        cube = apply_algorythm(
            cube, (cube_rotations['x2'], cube_rotations['y\'']))
        rotations.append(cube)

    for cube in tuple(rotations):
        for _ in range(3):
            cube = apply_action(cube, cube_rotations['y'])
            rotations.append(cube)

    return set(rotations)


def calculate_solved_cubes():
    original = Cube.parse("0010203040506070")
    return all_rotations(original)

solved_cubes = calculate_solved_cubes()

def calculate_solved_positions():
    positions = set()
    for cube in solved_cubes:
        positions.add(tuple(p.id for p in cube.pieces))
    return positions

solved_positions = calculate_solved_positions()

