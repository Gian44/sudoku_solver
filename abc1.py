import random
import numpy as np
from abc_solver import ABCSolver
from hybrid_abc_cp import HybridABCSolver

def display_sudoku(grid):
    """Display Sudoku grid - accepts either dict or numpy array"""
    if isinstance(grid, np.ndarray):
        # Convert array to dict for display
        temp_dict = {}
        for i in range(9):
            row = 'ABCDEFGHI'[i]
            for j in range(9):
                cell = f"{row}{j+1}"
                temp_dict[cell] = str(grid[i,j]) if grid[i,j] != 0 else ' '
        grid = temp_dict
    
    print("+" + "---+"*9)
    for i in range(9):
        row = 'ABCDEFGHI'[i]
        print("|", end="")
        for j in range(1, 10):
            cell = f"{row}{j}"
            val = grid.get(cell, " ")
            print(f" {val} |", end="")
        print("\n+" + "---+"*9)

def main():
    # Test puzzle (AI Escargot - one of the hardest Sudoku puzzles)
    puzzle_str = "1....7.9..3..2...8..96..5....53..9...1..8...26....4...3......1..41.....7..7...3.."
    
    # Convert to grid dictionary format
    puzzle = {}
    for i, char in enumerate(puzzle_str):
        row = 'ABCDEFGHI'[i // 9]
        col = (i % 9) + 1
        puzzle[f"{row}{col}"] = char if char != '.' else ''
    
    print("Original Puzzle:")
    display_sudoku(puzzle)
    
    # Initialize and run solver
    solver = HybridABCSolver(puzzle)
    print("\nSolving with Hybrid Artificial Bee Colony...")
    
    solution = solver.solve()
    
    if solution is not None:
        print("\nSolution Found:")
        display_sudoku(solution)
        
        # Verify solution
        valid = True
        # Check rows and columns
        for i in range(9):
            # Check rows
            if len(set(solution[i,:])) != 9 or 0 in solution[i,:]:
                valid = False
            # Check columns
            if len(set(solution[:,i])) != 9 or 0 in solution[:,i]:
                valid = False
        
        # Check 3x3 subgrids
        for box_r in range(0, 9, 3):
            for box_c in range(0, 9, 3):
                subgrid = solution[box_r:box_r+3, box_c:box_c+3].flatten()
                if len(set(subgrid)) != 9 or 0 in subgrid:
                    valid = False
        
        if valid:
            print("\nVALID SOLUTION!")
        else:
            print("\nINVALID SOLUTION - Constraints violated")
    else:
        print("\nNo solution found within the iteration limit")

if __name__ == "__main__":
    main()