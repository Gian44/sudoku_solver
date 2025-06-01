import numpy as np
import random
from abc_solver import ABCSolver

class MSABCSolver:
    """Multi-Swarm ABC Solver for Sudoku."""
    def __init__(self, puzzle_input, size, population_size, num_swarms, rcloud, max_cycles=10000):
        self.size = size
        self.population_size = population_size
        self.num_swarms = num_swarms
        self.max_cycles = max_cycles
        self.rcloud = rcloud
        self.swarms = [ABCSolver(puzzle_input, self.size, self.population_size) for _ in range(num_swarms)]
        self.global_best = None
        self.global_fitness = -np.inf

    def solve(self):
        """Run the multi-swarm optimization."""
        for cycle in range(self.max_cycles):
            # --- Phase 1: Parallel Sub-swarm Optimization ---
            for swarm in self.subswarms:
                swarm.abc.solve()  # Run one ABC iteration
                swarm.update_best()

                # Update global best
                if swarm.best_fitness > self.global_fitness:
                    self.global_best = swarm.best_solution.copy()
                    self.global_fitness = swarm.best_fitness

            yield self.global_best, self.global_fitness

            # Early termination
            if self.global_fitness == 1.0:
                break

    def get_solution(self):
        """Return the solved Sudoku as a dictionary."""
        if self.global_best is None:
            return None
        return {f"{chr(65 + r)}{c + 1}": str(self.global_best[r, c]) 
                for r in range(self.size) for c in range(self.size)}