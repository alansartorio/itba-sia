import itertools
import functools
from enum import Enum, auto
from typing import DefaultDict


class Colors(Enum):
    RED = auto()
    ORANGE = auto()
    WHITE = auto()
    YELLOW = auto()
    BLUE = auto()
    GREEN = auto()

    def to_string(self):
        return '\033[1;' + {
            Colors.RED: '31m',
            Colors.ORANGE: '36m',
            Colors.WHITE: '37m',
            Colors.YELLOW: '33m',
            Colors.BLUE: '34m',
            Colors.GREEN: '32m',
        }[self] + '█\033[0m'
        # return {
            # Colors.RED: 'r',
            # Colors.ORANGE: 'o',
            # Colors.WHITE: 'w',
            # Colors.YELLOW: 'y',
            # Colors.BLUE: 'b',
            # Colors.GREEN: 'g',
        # }[self]


piece_colors = (
    (Colors.RED, Colors.WHITE, Colors.BLUE),
    (Colors.GREEN, Colors.WHITE, Colors.RED),
    (Colors.BLUE, Colors.YELLOW, Colors.RED),
    (Colors.RED, Colors.YELLOW, Colors.GREEN),

    (Colors.BLUE, Colors.WHITE, Colors.ORANGE),
    (Colors.ORANGE, Colors.WHITE, Colors.GREEN),
    (Colors.ORANGE, Colors.YELLOW, Colors.BLUE),
    (Colors.GREEN, Colors.YELLOW, Colors.ORANGE),
)


def rotate_3(a, b, c):
    return c, a, b


class Piece:
    def __init__(self, id: int, orientation: int) -> None:
        self.id = id
        self.orientation = orientation

    @classmethod
    def parse(cls, s: str):
        return cls(int(s[0]), int(s[1]))

    def __repr__(self) -> str:
        return f'{self.id}{self.orientation}'

    def get_colors(self) -> tuple[Colors, Colors, Colors]:
        colors = piece_colors[self.id]
        # for _ in range(self.orientation):
            # colors = rotate_3(*colors)
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
        l = [Piece.parse(s[i:i+2]) for i in range(0, 16, 2)]
        return cls((l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7]))

    def __repr__(self) -> str:
        return ''.join(map(repr, self.pieces))

    def clone(self) -> 'Cube':
        return Cube.parse(repr(self))

    def colored_faces(self):
        pc = []
        for i, piece in enumerate(self.pieces):
            a, b, c = piece.get_colors()
            if i in (1, 2, 4, 7):
                a, b, c = c, b, a
            a, b, c = rotate_3(a, b, c)
            pc.append((a, b, c))

        up = ((pc[3][0], pc[1][0]), (pc[2][0], pc[0][0]))
        down = ((pc[6][0], pc[4][0]), (pc[7][0], pc[5][0]))
        right = ((pc[0][1], pc[1][1]), (pc[4][1], pc[5][1]))
        left = ((pc[3][1], pc[2][1]), (pc[7][1], pc[6][1]))
        front = ((pc[2][2], pc[0][2]), (pc[6][2], pc[4][2]))
        back = ((pc[1][2], pc[3][2]), (pc[5][2], pc[7][2]))

        return (up, down, right, left, front, back)

    def to_string(self):
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
                    text[y][x] = faces[c][count // 2][count % 2].to_string()
                    counters[c] += 1

        return "\n".join(["".join(row) for row in text])


class Turn:
    def __init__(self, cycle: tuple[int, int, int, int], orientation: int):
        self.cycle = cycle
        self.orientation = orientation

    def reverse(self) -> 'Turn':
        c0, c1, c2, c3 = self.cycle
        return Turn((c3, c2, c1, c0), self.orientation)

    def next(self, cube: Cube) -> Cube:
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


turns = {
    'R': Turn((0, 1, 5, 4), 1),
    'L': Turn((6, 7, 3, 2), 1),
}

print(Cube.solved()[0].to_string())
print(turns['R'].next(Cube.solved()[0]).to_string())
print(turns['R'].next(Cube.solved()[0]))
