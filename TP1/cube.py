from abc import ABCMeta, abstractmethod
from enum import Enum, auto
from typing import Iterable


class Colors(Enum):
    RED = auto()
    ORANGE = auto()
    WHITE = auto()
    YELLOW = auto()
    BLUE = auto()
    GREEN = auto()

    def __str__(self):
        return '\033[1;' + {
            Colors.RED: '31m',
            Colors.ORANGE: '36m',
            Colors.WHITE: '37m',
            Colors.YELLOW: '33m',
            Colors.BLUE: '34m',
            Colors.GREEN: '32m',
        }[self] + '█\033[0m'


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

    @classmethod
    def solved(cls):
        return [cls.parse("0010203040506070")]

    def is_solved(self):
        pass

    @classmethod
    def parse(cls, s: str):
        assert len(s) == 16
        l = [Piece.parse(s[i:i+2]) for i in range(0, 16, 2)]
        return cls((l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7]))

    def __repr__(self) -> str:
        return ''.join(map(repr, self.pieces))

    def clone(self) -> 'Cube':
        return Cube.parse(repr(self))

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


class Turn(metaclass=ABCMeta):
    @abstractmethod
    def apply(self, cube: Cube) -> Cube: ...


class SingleTurn(Turn):
    def __init__(self, cycle: tuple[int, int, int, int], orientation: int):
        self.cycle = cycle
        self.orientation = orientation

    def doubled(self) -> 'DoubleTurn':
        return DoubleTurn(self)

    def reversed(self) -> 'SingleTurn':
        c0, c1, c2, c3 = self.cycle
        return SingleTurn((c3, c2, c1, c0), self.orientation)

    def apply(self, cube: Cube) -> Cube:
        cube = cube.clone()
        c0, c1, c2, c3 = self.cycle
        tmp = cube.pieces[c0]
        cube.pieces[c0] = cube.pieces[c3]
        cube.pieces[c3] = cube.pieces[c2]
        cube.pieces[c2] = cube.pieces[c1]
        cube.pieces[c1] = tmp
        for i in self.cycle:
            o = cube.pieces[i].orientation

            if o != self.orientation:
                o = (o + 1) % 3
                if o == self.orientation:
                    o = (o + 1) % 3
            cube.pieces[i].orientation = o
        return cube


class DoubleTurn(Turn):
    def __init__(self, turn: SingleTurn):
        self.turn = turn

    def apply(self, cube: Cube) -> Cube:
        for _ in range(2):
            cube = self.turn.apply(cube)
        return cube


def apply_turn(cube: Cube, turn: Turn):
    return turn.apply(cube)


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


def parse_algorythm(algo: str):
    return tuple(map(turns.__getitem__, algo.split()))


def apply_algorythm(cube: Cube, alg: Iterable[Turn]):
    for turn in alg:
        cube = apply_turn(cube, turn)
    return cube


solved = Cube.solved()[0]
print(str(solved))
print()
print(str(apply_turn(solved, turns["F'"])))
T = parse_algorythm("R U R' U' R' F R2 U' R' U' R U R' F'")
print(str(apply_algorythm(solved, T)))
