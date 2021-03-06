from cube import Cube
from scramble import generate_scrambles_by_solve_depth
import json


def load():
    with open('scrambles.json') as file:
        return dict(enumerate(list(map(Cube.parse, cubes)) for cubes in json.load(file)))


if __name__ == '__main__':
    try:
        old_scrambles = load()
    except FileNotFoundError:
        old_scrambles = {}
    scrambles = generate_scrambles_by_solve_depth(100, 300, old_scrambles)


    with open('scrambles.json', 'w') as file:
        json.dump([[scramble.get_state_string() for scramble in scrambles[i]]
                  if i in scrambles else [] for i in range(20)], file, indent=2)

