import json
from methods import methods, heuristic_methods
from cube import Cube
from heuristics import manhattan_distance, move_count_combination, sticker_groups


class Config:
    def __init__(self, state, method, heuristic):
        self.state = state
        match heuristic:
            case 'sticker_groups':
                self.heuristic = sticker_groups
            case 'move_count_combination':
                self.heuristic = move_count_combination
            case 'manhattan_distance':
                self.heuristic = manhattan_distance
        if heuristic:
            self.method = lambda cube:  heuristic_methods[method](
                cube, self.heuristic)
        else:
            self.method = methods[method]

    @classmethod
    def parse(cls):
        with open('./config.json', 'r') as f:
            data = json.load(f)
        state = Cube.parse(data['state'])

        return cls(state, data['method'], data['heuristic'] if 'heuristic' in data else None)
