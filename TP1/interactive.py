from functools import partial
from inquirer import List
from cube import Cube
import methods
import inquirer
import heuristics
from methods import time_solve_reduced


q = [
    List(
        "Method",
        "Which method to use?", [
            ("Breadth first search (BPA)", methods.test_bpa),
            ("Depth first search (BPP)", methods.test_bpp),
            ("Iterative deepening depth first search (BPPV)", methods.test_bppv),
            ("Local heuristic search", methods.test_local_heuristics),
            ("Global heuristic search", methods.test_global_heuristics),
            ("A* search", methods.test_global_heuristics_cost),
        ]
    ),
    List("Heuristic", "Which heuristic to use?", [
        ("Sticker Groups", heuristics.sticker_groups),
        ("Move Count Combination", heuristics.move_count_combination),
        ("Manhattan Distance", heuristics.manhattan_distance),
    ],
        ignore=lambda a:a['Method'] not in methods.heuristic_methods.values()
    ),
]

answers = inquirer.prompt(q)

if answers is None:
    exit(1)

search_method = answers['Method']
heuristic = answers['Heuristic']


def method(cube: Cube):
    return time_solve_reduced(search_method if heuristic is None else partial(search_method, heuristic), cube)


try:
    while True:
        cube = input("Cube state:")
        try:
            output = method(Cube.parse(cube))
            print(output)
        except KeyboardInterrupt:
            print("Search cancelled.")
except KeyboardInterrupt:
    pass
