
from collections import defaultdict

from tqdm.std import tqdm
from cube import solved_cubes, apply_action, turns
import random
from methods import test_global_heuristics_cost
from heuristics import move_count_combination
from multiprocessing.pool import Pool, ThreadPool
from itertools import count

def generate_scrambled(moves: int = 30):
    cube = next(iter(solved_cubes))
    allowed_moves = list(map(turns.__getitem__, "R R' L L' U U' D D' F F' B B'".split()))
    for _ in range(moves):
        move = random.sample(allowed_moves, 1)[0]
        cube = apply_action(cube, move)
    return cube

def generate_scramble_by_optimum_solve_depth(depth: int):
    while True:
        scramble = generate_scrambled(depth)
        solution, _, _ = test_global_heuristics_cost(scramble, move_count_combination)
        if solution is not None and solution.get_depth() == depth:
            return scramble


def generate_scramble_with_solution(moves: int):
    scramble = generate_scrambled(random.randint(1, 20))
    solution, _, _ = test_global_heuristics_cost(scramble, move_count_combination)
    assert solution is not None
    depth = solution.get_depth()
    return scramble, depth

def multithreaded_generate():
    def random_move_count():
        while True:
            yield random.randint(1, 20)
    with Pool(12) as pool:
        yield from pool.imap_unordered(generate_scramble_with_solution, random_move_count())
    return

def generate_scrambles_by_solve_depth(min_each: int, max_each: int):
    scrambles_by_depth = defaultdict(lambda:[])

    try:
        generator = multithreaded_generate()
        while min(map(len, scrambles_by_depth.values()), default=0) < min_each:
            scramble, depth = next(generator)
            print("Found solution with depth =", depth)
            if len(scrambles_by_depth[depth]) < max_each:
                scrambles_by_depth[depth].append(scramble)
    except KeyboardInterrupt:
        pass
    finally:
        return dict(scrambles_by_depth)


# for _ in range(100):
    # print(repr(generate_scrambled(30)))
