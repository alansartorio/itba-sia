import itertools
import functools


class Piece:
    def __init__(self, id: int, orientation: int) -> None:
        self.id = id
        self.orientation = orientation

    @classmethod
    def parse(cls, s: str):
        return cls(int(s[0]), int(s[1]))

    def __repr__(self) -> str:
        return f'{self.id}{self.orientation}'


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
        cube.pieces[c0] = cube.pieces[c1]
        cube.pieces[c1] = cube.pieces[c2]
        cube.pieces[c2] = cube.pieces[c3]
        cube.pieces[c3] = tmp
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

print(turns['R'].next(Cube.solved()[0]))
