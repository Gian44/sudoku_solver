import random
import numpy as np
from sudo import initialize_structures

class ABCSolver:
    def __init__(self, puzzle_input, size=9, population_size=50):
        self.size = size
        (self.rows, self.cols, self.squares, self.unitlist, 
         self.units, self.peers, self.subgrid_rows, 
         self.subgrid_cols, self.digits) = initialize_structures(size)
        
        self.grid = self.parse_input(puzzle_input)
        self.fixed_cells = (self.grid != 0)
        
        self.max_cycles = 100000
        self.population_size = population_size
        self.employed_bees = self.population_size
        self.onlooker_bees = self.employed_bees * 2
        self.scout_limit = 10
        
        self.population = self.initialize_population()
        self.fitnesses = [self.evaluate(sol) for sol in self.population]
        self.best_solution = max(self.population, key=lambda x: self.evaluate(x))
        self.best_fitness = self.evaluate(self.best_solution)
        self.trials = [0] * self.employed_bees

    def parse_input(self, puzzle_input):
        grid = np.zeros((self.size, self.size), dtype=int)
        if isinstance(puzzle_input, dict):
            for cell, value in puzzle_input.items():
                if value in self.digits:
                    row = ord(cell[0].upper()) - ord('A')
                    col = int(cell[1:]) - 1
                    grid[row, col] = self.digits.index(value) + 1
        else:
            grid = np.array(puzzle_input, dtype=int)
        return grid

    def initialize_population(self):
        population = []
        for _ in range(self.employed_bees):
            solution = self.grid.copy()
            for box_r in range(0, self.size, self.subgrid_rows):
                for box_c in range(0, self.size, self.subgrid_cols):
                    existing = set()
                    for r in range(box_r, box_r + self.subgrid_rows):
                        for c in range(box_c, box_c + self.subgrid_cols):
                            if self.fixed_cells[r, c]:
                                existing.add(solution[r, c])
                    
                    missing = [i+1 for i in range(self.size) if (i+1) not in existing]
                    random.shuffle(missing)
                    
                    idx = 0
                    for r in range(box_r, box_r + self.subgrid_rows):
                        for c in range(box_c, box_c + self.subgrid_cols):
                            if not self.fixed_cells[r, c]:
                                solution[r, c] = missing[idx]
                                idx += 1
            population.append(solution)
        return population

    def evaluate(self, solution):
        penalty = 0
        for row in solution:
            penalty += self.size - len(set(row))
        for col in solution.T:
            penalty += self.size - len(set(col))
        return 1 / (1 + penalty)

    def neighbor_search(self, current_solution):
        new_solution = current_solution.copy()
        while True:
            row = random.randint(0, self.size-1)
            col = random.randint(0, self.size-1)
            if not self.fixed_cells[row, col]:
                break
        
        block_row = (row // self.subgrid_rows) * self.subgrid_rows
        block_col = (col // self.subgrid_cols) * self.subgrid_cols
        
        neighbor_solution = random.choice(self.population)
        while np.array_equal(neighbor_solution, current_solution):
            neighbor_solution = random.choice(self.population)
        
        x_ij = current_solution[row, col]
        x_kj = neighbor_solution[row, col]
        v_ij = x_ij + random.random() * abs(x_ij - x_kj)
        v_ij = int(round(v_ij))
        
        if v_ij > self.size:
            v_ij = (v_ij % self.size) + 1
        elif v_ij < 1:
            v_ij = self.size - (abs(v_ij) % self.size)
        
        block = current_solution[block_row:block_row+self.subgrid_rows, 
                               block_col:block_col+self.subgrid_cols]
        if v_ij in block:
            positions = np.argwhere(block == v_ij)
            for pos in positions:
                swap_r, swap_c = pos
                abs_r = block_row + swap_r
                abs_c = block_col + swap_c
                if not self.fixed_cells[abs_r, abs_c]:
                    new_solution[abs_r, abs_c] = x_ij
                    new_solution[row, col] = v_ij
                    break
        else:
            new_solution[row, col] = v_ij
        
        return new_solution

    def solve(self):
        for cycle in range(self.max_cycles):
            if self.best_fitness == 1:
                yield self.best_solution, 1.0
                return
                
            # Employed bees phase
            for i in range(self.employed_bees):
                neighbor = self.neighbor_search(self.population[i])
                neighbor_fit = self.evaluate(neighbor)
                
                if neighbor_fit >= self.fitnesses[i]:
                    self.population[i] = neighbor
                    self.fitnesses[i] = neighbor_fit
                    self.trials[i] = 0
                else:
                    self.trials[i] += 1
                    
                if neighbor_fit >= self.best_fitness:
                    self.best_solution = neighbor.copy()
                    self.best_fitness = neighbor_fit
            
            # Onlooker bees phase
            sum_fitness = sum(self.fitnesses)
            if sum_fitness > 0:
                for i in range(self.employed_bees):
                    probability = self.fitnesses[i] / sum_fitness
                    for _ in range(int(probability * self.onlooker_bees)):
                        neighbor = self.neighbor_search(self.population[i])
                        neighbor_fit = self.evaluate(neighbor)
                        
                        if neighbor_fit >= self.fitnesses[i]:
                            self.population[i] = neighbor
                            self.fitnesses[i] = neighbor_fit
                            self.trials[i] = 0
                            
                        if neighbor_fit >= self.best_fitness:
                            self.best_solution = neighbor.copy()
                            self.best_fitness = neighbor_fit
            
            # Scout bees phase
            if max(self.trials) > self.scout_limit:
                idx = self.trials.index(max(self.trials))
                self.population[idx] = self.initialize_population()[0]
                self.fitnesses[idx] = self.evaluate(self.population[idx])
                self.trials[idx] = 0
                
                if self.fitnesses[idx] >= self.best_fitness:
                    self.best_solution = self.population[idx].copy()
                    self.best_fitness = self.fitnesses[idx]
            
            yield self.best_solution, self.best_fitness