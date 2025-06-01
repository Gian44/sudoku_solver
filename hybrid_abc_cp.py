import random
import numpy as np
from copy import deepcopy
from sudo import initialize_structures, assign, eliminate

class HybridABCSolver:
    def __init__(self, puzzle_dict, size=9):
        self.size = size
        (self.rows, self.cols, self.squares, self.unitlist, 
         self.units, self.peers, self.subgrid_rows, 
         self.subgrid_cols, self.digits) = initialize_structures(size)
        
        self.puzzle = self.dict_to_values(puzzle_dict)
        self.fixed_cells = {s for s in self.squares if len(self.puzzle[s]) == 1}
        self.max_cycles = 1000
        self.employed_bees = 10
        self.onlooker_bees = 20
        self.scout_limit = 10
        
        self.population = [self.create_cp_solution() for _ in range(self.employed_bees)]
        self.fitnesses = [self.evaluate(sol) for sol in self.population]
        self.best_solution = self.population[np.argmax(self.fitnesses)]
        self.best_fitness = max(self.fitnesses)
        self.trials = [0] * self.employed_bees

    def dict_to_values(self, puzzle_dict):
        values = {s: self.digits.copy() for s in self.squares}
        for s, d in puzzle_dict.items():
            if d in self.digits:
                values = assign(values.copy(), s, d, self.size, self.units, self.peers)
        return values

    def create_cp_solution(self):
        values = deepcopy(self.puzzle)
        for s in self.squares:
            if len(values[s]) == 1:
                continue
            for d in random.sample(values[s], len(values[s])):
                new_values = assign(values.copy(), s, d, self.size, self.units, self.peers)
                if new_values:
                    values = new_values
                    break
        return values

    def evaluate(self, values):
        fixed = sum(1 for s in self.squares if len(values[s]) == 1)
        if fixed == self.size**2:
            return 1.0
        return fixed / (self.size**2)

    def neighbor_search(self, current_solution):
        new_values = deepcopy(current_solution)
        while True:
            s = random.choice(self.squares)
            if s not in self.fixed_cells and len(new_values[s]) > 1:
                break
                
        for d in random.sample(new_values[s], len(new_values[s])):
            candidate = assign(new_values.copy(), s, d, self.size)
            if candidate:
                return candidate
                
        return new_values

    def scout_phase(self):
        idx = self.trials.index(max(self.trials))
        self.population[idx] = self.create_cp_solution()
        self.fitnesses[idx] = self.evaluate(self.population[idx])
        self.trials[idx] = 0

    def solve(self):
        for cycle in range(self.max_cycles):
            if self.best_fitness >= 0.999:
                yield self.best_solution, 1.0
                return

            # Employed bees phase
            for i in range(self.employed_bees):
                neighbor = self.neighbor_search(self.population[i])
                neighbor_fit = self.evaluate(neighbor)
                
                if neighbor_fit > self.fitnesses[i]:
                    self.population[i] = neighbor
                    self.fitnesses[i] = neighbor_fit
                    self.trials[i] = 0
                else:
                    self.trials[i] += 1
                    
                if neighbor_fit > self.best_fitness:
                    self.best_solution = neighbor
                    self.best_fitness = neighbor_fit
                    yield self.best_solution, self.best_fitness

            # Onlooker bees phase
            total_fitness = sum(self.fitnesses)
            if total_fitness > 0:
                for i in range(self.employed_bees):
                    prob = self.fitnesses[i] / total_fitness
                    for _ in range(int(prob * self.onlooker_bees)):
                        neighbor = self.neighbor_search(self.population[i])
                        neighbor_fit = self.evaluate(neighbor)
                        
                        if neighbor_fit > self.fitnesses[i]:
                            self.population[i] = neighbor
                            self.fitnesses[i] = neighbor_fit
                            self.trials[i] = 0
                            
                        if neighbor_fit > self.best_fitness:
                            self.best_solution = neighbor
                            self.best_fitness = neighbor_fit
                            yield self.best_solution, self.best_fitness

            # Scout bees phase
            if max(self.trials) > self.scout_limit:
                self.scout_phase()
                new_fit = self.fitnesses[self.trials.index(0)]
                if new_fit > self.best_fitness:
                    self.best_solution = self.population[self.trials.index(0)]
                    self.best_fitness = new_fit
                    yield self.best_solution, self.best_fitness

            self.best_solution_yield = {k: v[0] for k, v in self.best_solution.items()}
            yield self.best_solution_yield, self.best_fitness