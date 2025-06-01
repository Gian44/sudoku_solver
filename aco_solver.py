import random
import numpy as np
from sudo import initialize_structures

class ACOSolver:
    def __init__(self, puzzle_dict, size=9, num_ants=50, evaporation_rate=0.5, greediness=0.9, bve_rate=0.1):
        self.size = size
        (self.rows, self.cols, self.squares, self.unitlist, 
         self.units, self.peers, self.subgrid_rows, 
         self.subgrid_cols, self.digits) = initialize_structures(size)
        
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.greediness = greediness
        self.bve_rate = bve_rate
        
        self.values = self._initialize_values(puzzle_dict)
        if self.values is False:
            raise ValueError("Invalid initial puzzle")
        
        self.pheromone = np.ones((self.size**2, self.size)) * (1.0 / self.size**2)
        self.best_pheromone_to_add = 0
        self.best_solution = None
        self.best_fixed = 0

    def _initialize_values(self, grid):
        values = {s: self.digits.copy() for s in self.squares}
        for s, d in grid.items():
            if d in self.digits:
                values = self._assign_value(values, s, d)
                if values is False:
                    return False
        return values

    def _assign_value(self, values, s, d):
        other_values = [v for v in values[s] if v != d]
        for v in other_values:
            values = self._eliminate_value(values, s, v)
            if values is False:
                return False
        return values

    def _eliminate_value(self, values, s, d):
        if d not in values[s]:
            return values
        
        values[s].remove(d)
        
        if not values[s]:
            return False
        
        if len(values[s]) == 1:
            d2 = values[s][0]
            for s2 in self.peers[s]:
                values = self._eliminate_value(values, s2, d2)
                if values is False:
                    return False
        
        for u in self.units[s]:
            dplaces = [s for s in u if d in values[s]]
            if len(dplaces) == 0:
                return False
            if len(dplaces) == 1:
                values = self._assign_value(values, dplaces[0], d)
                if values is False:
                    return False
        return values

    def solve(self):
        while True:
            ants = []
            for _ in range(self.num_ants):
                ant_values = {s: self.values[s].copy() for s in self.squares}
                ants.append({'values': ant_values, 'fixed': 0, 'solution': {}, 'valid': True})
            
            starting_cells = random.sample(self.squares, self.num_ants)
            for i, ant in enumerate(ants):
                ant['current_cell'] = starting_cells[i]
            
            for _ in range(self.size**2):
                for ant in ants:
                    if not ant['valid']:
                        continue
                        
                    s = ant['current_cell']
                    
                    if len(ant['values'][s]) > 1:
                        chosen_value = self._choose_value(ant['values'], s, self.squares.index(s))
                        new_values = self._assign_value(ant['values'].copy(), s, chosen_value)
                        if new_values is False:
                            ant['valid'] = False
                            continue
                            
                        ant['values'] = new_values
                        ant['solution'][s] = chosen_value
                        self._local_pheromone_update(self.squares.index(s), self.digits.index(chosen_value))
                    
                    ant['fixed'] = sum(1 for s in self.squares if len(ant['values'][s]) == 1)
                    current_index = self.squares.index(ant['current_cell'])
                    ant['current_cell'] = self.squares[(current_index + 1) % (self.size**2)]
            
            valid_ants = [ant for ant in ants if ant['valid']]
            if not valid_ants:
                yield {}, 0.0
                continue
                
            iteration_best = max(valid_ants, key=lambda x: x['fixed'])
            
            if iteration_best['fixed'] >= self.best_fixed:
                self.best_fixed = iteration_best['fixed']
                self.best_solution = iteration_best['values']
                delta_tau = (self.size**2) / (self.size**2 - self.best_fixed + 1e-6)
                if delta_tau > self.best_pheromone_to_add:
                    self.best_pheromone_to_add = delta_tau
            
            self._global_pheromone_update()
            self.best_pheromone_to_add *= (1 - self.bve_rate)

            yield self.best_solution, self.best_fixed / (self.size**2)
            
            if self.best_fixed == self.size**2:
                break
        
        yield self.best_solution, 1.0
    
    def _choose_value(self, values, s, cell_idx):
        possible_values = [d for d in values[s] if d in self.digits]
        if len(possible_values) == 1:
            return possible_values[0]
        
        pheromone_levels = [self.pheromone[cell_idx, self.digits.index(d)] for d in possible_values]
        
        if random.random() < self.greediness:
            chosen_index = np.argmax(pheromone_levels)
        else:
            total = sum(pheromone_levels)
            probabilities = [p/total for p in pheromone_levels]
            chosen_index = np.random.choice(len(possible_values), p=probabilities)
        
        return possible_values[chosen_index]
    
    def _local_pheromone_update(self, cell_idx, value_idx):
        xi = 0.1
        tau0 = 1.0 / (self.size**2)
        self.pheromone[cell_idx, value_idx] = (1 - xi) * self.pheromone[cell_idx, value_idx] + xi * tau0
    
    def _global_pheromone_update(self):
        if not self.best_solution or self.best_pheromone_to_add <= 0:
            return
        
        for cell_idx, s in enumerate(self.squares):
            if s in self.best_solution and len(self.best_solution[s]) == 1:
                value = self.digits.index(self.best_solution[s])
                self.pheromone[cell_idx, value] = (
                    (1 - self.evaporation_rate) * self.pheromone[cell_idx, value] + 
                    self.evaporation_rate * self.best_pheromone_to_add
                )