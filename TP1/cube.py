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


def flatten(pieces: tuple[tuple[tuple[Piece, Piece], tuple[Piece, Piece]], tuple[tuple[Piece, Piece], tuple[Piece, Piece]]]):
    return (pieces[0][0][0], pieces[0][0][1], 
            pieces[0][1][0], pieces[0][1][1], 
            pieces[1][0][0], pieces[1][0][1], 
            pieces[1][1][0], pieces[1][1][1])


class Cube:
    def __init__(self, pieces: tuple[tuple[tuple[Piece, Piece], tuple[Piece, Piece]], tuple[tuple[Piece, Piece], tuple[Piece, Piece]]]) -> None:
        self.pieces = pieces

    @classmethod
    def solved(cls):
        return [cls.parse("0010203040506070")]

    def is_solved(self):
        pass

    @classmethod
    def parse(cls, s: str):
        l = [Piece.parse(s[i:i+2]) for i in range(0, 16, 2)]
        return cls((((l[0], l[1]), (l[2], l[3])), ((l[4], l[5]), (l[6], l[7]))))

    def __repr__(self) -> str:
        return ''.join((repr(piece) for piece in flatten(self.pieces)))


print(Cube.solved()[0])
