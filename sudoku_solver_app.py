# sudoku_solver_app.py
import streamlit as st
import time
import numpy as np
from sudo import solve as norvig_solve, grid_values, initialize_structures
from abc_solver import ABCSolver
from hybrid_abc_cp import HybridABCSolver
from aco_solver import ACOSolver

# Custom CSS with error highlighting
st.markdown("""
<style>
.sudoku-container {
    display: inline-block;
    margin: 20px 0;
}
.sudoku-grid {
    border: 3px solid #000;
    display: inline-block;
    background-color: white;
}
.sudoku-row {
    display: flex;
}
.sudoku-cell {
    width: 50px;
    height: 50px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    font-weight: bold;
    color: #000 !important;
    border: 1px solid #999;
    background-color: white;
}
.sudoku-cell.fixed {
    background-color: #f0f0f0;
}
.sudoku-cell.solved {
    background-color: #e6f7ff;
}
.sudoku-cell.error {
    background-color: #ffdddd;
    color: #ff0000 !important;
}
.border-right-thick {
    border-right: 3px solid #000 !important;
}
.border-bottom-thick {
    border-bottom: 3px solid #000 !important;
}
</style>
""", unsafe_allow_html=True)

def is_incorrect(grid, row, col, value, size, subgrid_rows, subgrid_cols):
    """Check if a value violates Sudoku rules in the current grid"""
    if value == 0:
        return False
    
    # Check row
    if np.sum(grid[row, :] == value) > 1:
        return True
    
    # Check column
    if np.sum(grid[:, col] == value) > 1:
        return True
    
    # Check subgrid
    box_row = (row // subgrid_rows) * subgrid_rows
    box_col = (col // subgrid_cols) * subgrid_cols
    box = grid[box_row:box_row+subgrid_rows, box_col:box_col+subgrid_cols]
    if np.sum(box == value) > 1:
        return True
    
    return False

def display_sudoku(grid, is_solution=False, original_grid=None, size=9):
    """Render Sudoku grid with error highlighting"""
    rows, cols, squares, unitlist, units, peers, subgrid_rows, subgrid_cols, digits = initialize_structures(size)
    
    if isinstance(grid, np.ndarray):
        grid_dict = {}
        for i in range(size):
            for j in range(size):
                cell_id = f"{rows[i]}{cols[j]}"
                value = grid[i,j] if grid[i,j] != 0 else ''
                grid_dict[cell_id] = str(value) if value != '' else ''
        grid = grid_dict
    
    # Convert original grid to dict if provided
    original_dict = None
    if original_grid is not None and isinstance(original_grid, np.ndarray):
        original_dict = {}
        for i in range(size):
            for j in range(size):
                cell_id = f"{rows[i]}{cols[j]}"
                value = original_grid[i,j] if original_grid[i,j] != 0 else ''
                original_dict[cell_id] = str(value) if value != '' else ''
    
    html = '<div class="sudoku-container"><div class="sudoku-grid">'
    for i in range(size):
        html += '<div class="sudoku-row">'
        for j in range(size):
            cell_id = f"{rows[i]}{cols[j]}"
            value = grid.get(cell_id, '')
            display_value = value if value not in ['', '0', '.'] else ''
            
            classes = "sudoku-cell"
            
            # Mark fixed cells (original puzzle clues)
            if original_dict and original_dict.get(cell_id, '') not in ['', '0', '.']:
                classes += " fixed"
            
            # Mark solved cells (final solution)
            if is_solution:
                classes += " solved"
            
            # Mark errors in intermediate solutions
            if not is_solution and original_dict and value not in ['', '0', '.']:
                if original_dict.get(cell_id, '') in ['', '0', '.']:  # Only check filled cells
                    # Convert to numpy array for checking
                    temp_grid = np.zeros((size,size), dtype=int)
                    for r in range(size):
                        for c in range(size):
                            cell = f"{rows[r]}{cols[c]}"
                            val = grid.get(cell, '0')
                            temp_grid[r,c] = int(val) if val not in ['', '.', '0'] else 0
                    
                    if is_incorrect(temp_grid, i, j, int(value), size, subgrid_rows, subgrid_cols):
                        classes += " error"
            
            # Add thick borders
            classes += " border-right-thick" if (j+1) % subgrid_cols == 0 and j != size-1 else ""
            classes += " border-bottom-thick" if (i+1) % subgrid_rows == 0 and i != size-1 else ""
            
            html += f'<div class="{classes}">{display_value}</div>'
        html += '</div>'
    html += '</div></div>'
    return html

def parse_puzzle(content):
    """Parse puzzle input that starts with size and has space-separated values"""
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    if not lines or not lines[0].isdigit():
        raise ValueError("First line must be grid size")
    
    size = int(lines[0])
    puzzle_lines = lines[1:]
    
    # Parse all cell values (handling multi-digit numbers)
    cells = []
    for line in puzzle_lines:
        for val in line.split():
            if val == '0' or val == '.':
                cells.append('.')
            else:
                try:
                    num = int(val)
                    if 1 <= num <= size:
                        cells.append(str(num) if num <= 9 else chr(ord('A') + num - 10))
                    else:
                        cells.append('.')
                except ValueError:
                    cells.append('.')
    
    # Convert to single string (Norvig format)
    puzzle_str = ''.join(cells)
    
    if len(puzzle_str) != size * size:
        raise ValueError(f"Expected {size*size} cells, got {len(puzzle_str)}")
    
    return size, puzzle_str

# Streamlit UI
st.title("🧩 Variable-Size Sudoku Solver")

method = st.selectbox(
    "Select solving method:",
    [
        "Norvig's Constraint Propagation", 
        "Artificial Bee Colony", 
        "Hybrid Artificial Bee Colony + Constraint Propagation",
        "Ant Colony Optimization"
    ]
)

uploaded_file = st.file_uploader(
    "Upload Sudoku puzzle (TXT format)",
    type=["txt", "sav"],
    help="First line: size (e.g., 6 for 6x6). Following lines: puzzle (1-9 for clues, 0 or . for empty cells)"
)

sample_puzzles = {
    "Select sample puzzle": "",
    "Easy 6x6": "6\n0 4 0 2 0 0\n2 0 5 0 0 0\n0 6 1 0 0 0\n0 0 0 4 6 0\n0 0 0 5 0 3\n0 0 3 0 1 0",
    "Hard 9x9": "9\n8 5 . . . 2 4 . .\n7 2 . . . . . . 9\n. . 4 . . . . . .\n. . . 1 . 7 . . 2\n3 . 5 . . . 9 . .\n. 4 . . . . . . .\n. . . 8 . . 7 . .\n1 7 . . . . . . .\n. . 3 6 . 4 . 8 .",
    "Easy 12x12": "12\n0 0 8 0 0 4 9 1 0 0 0 2\n0 0 2 0 11 8 0 0 1 0 0 6\n0 10 9 1 0 2 0 0 0 12 0 8\n4 12 3 10 0 0 0 8 2 0 0 0\n0 0 0 0 5 0 6 4 8 1 3 0\n0 0 0 5 2 3 11 10 0 0 9 0\n3 0 10 0 0 0 2 0 9 0 4 0\n0 2 4 12 0 6 1 11 0 0 0 0\n11 0 0 0 0 0 5 0 6 0 0 0\n0 11 12 9 1 0 0 3 0 6 0 0\n0 6 0 0 10 11 0 0 3 8 5 0\n0 3 0 2 6 0 0 0 0 10 0 12",
    "Al Escargot": "9\n1 . . . . 7 . 9 .\n. 3 . . 2 . . . 8\n. . 9 6 . . 5 . .\n. . 5 3 . . 9 . .\n. 1 . . 8 . . . 2\n6 . . . . 4 . . .\n3 . . . . . . 1 .\n. 4 1 . . . . . 7\n. . 7 . . . 3 . ."
}

puzzle_choice = st.selectbox("Or try sample puzzle:", list(sample_puzzles.keys()))
puzzle_size, puzzle_str  = parse_puzzle(sample_puzzles[puzzle_choice]) if puzzle_choice != "Select sample puzzle" else (9, "")

if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    try:
        puzzle_size, puzzle_str = parse_puzzle(content)
    except ValueError as e:
        st.error(f"Invalid puzzle input: {e}")
        st.stop()

if puzzle_str:
    rows, cols, squares, *_ = initialize_structures(puzzle_size)
    puzzle_dict = grid_values(f"{puzzle_size}\n{puzzle_str}", puzzle_size)
    
    puzzle_grid = np.zeros((puzzle_size, puzzle_size), dtype=int)
    for i in range(puzzle_size):
        for j in range(puzzle_size):
            cell = f"{rows[i]}{cols[j]}"
            val = puzzle_dict.get(cell, '0')
            puzzle_grid[i,j] = int(val) if val.isdigit() else 0
    
    st.subheader("Original Puzzle")
    st.markdown(display_sudoku(puzzle_grid, size=puzzle_size), unsafe_allow_html=True)
    
    if st.button("Solve Sudoku", type="primary"):
        with st.spinner("Solving..."):
            start_time = time.time()
            
            if method == "Norvig's Constraint Propagation":
                solution = norvig_solve(puzzle_str, puzzle_size)
                if isinstance(solution, str):
                    solution_dict = {squares[i]: solution[i] for i in range(len(solution))}
                elif isinstance(solution, dict):
                    solution_dict = solution
                else:
                    st.error("Unexpected solution format")
                    st.stop()
                
                solution_grid = np.zeros((puzzle_size, puzzle_size), dtype=int)
                for i in range(puzzle_size):
                    for j in range(puzzle_size):
                        cell = f"{rows[i]}{cols[j]}"
                        val = solution_dict.get(cell, '0')
                        # Handle both digit strings and letters
                        if val.isdigit():
                            solution_grid[i,j] = int(val)
                        elif val in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                            solution_grid[i,j] = 10 + ord(val.upper()) - ord('A')
                        else:
                            solution_grid[i,j] = 0
                
                st.subheader("Solution")
                st.markdown(display_sudoku(solution_grid, is_solution=True, size=puzzle_size), unsafe_allow_html=True)
                st.success(f"Solved in {time.time() - start_time:.2f} seconds!")
            
            elif method == "Ant Colony Optimization":
                solver = ACOSolver(puzzle_dict, puzzle_size)
                progress_bar = st.progress(0)
                time_text = st.empty()
                fitness_text = st.empty()
                solution_placeholder = st.empty()
                
                for solution, fitness in solver.solve():
                    progress_bar.progress(int(fitness * 100))
                    time_text.text(f"Elapsed time: {time.time() - start_time:.2f}s")
                    fitness_text.text(f"Fitness: {fitness:.4f}")
                    
                    solution_grid = np.zeros((puzzle_size, puzzle_size), dtype=int)
                    for i in range(puzzle_size):
                        for j in range(puzzle_size):
                            cell = f"{rows[i]}{cols[j]}"
                            val = solution.get(cell, '0') if isinstance(solution, dict) else solution[i,j]
                            solution_grid[i,j] = int(val) if str(val).isdigit() else 0
                    
                    solution_placeholder.markdown(
                        display_sudoku(solution_grid, original_grid=puzzle_grid, size=puzzle_size), 
                        unsafe_allow_html=True
                    )
                    
                    if fitness >= 0.999:
                        st.success(f"Solved in {time.time() - start_time:.2f} seconds!")
                        break
            
            else:  # ABC methods
                solver = HybridABCSolver(puzzle_dict, puzzle_size) if "Hybrid" in method else ABCSolver(puzzle_dict, puzzle_size)
                progress_bar = st.progress(0)
                time_text = st.empty()
                fitness_text = st.empty()
                solution_placeholder = st.empty()
                
                for solution, fitness in solver.solve():
                    progress_bar.progress(int(fitness * 100))
                    time_text.text(f"Elapsed time: {time.time() - start_time:.2f}s")
                    fitness_text.text(f"Fitness: {fitness:.4f}")
                    
                    solution_grid = np.zeros((puzzle_size, puzzle_size), dtype=int)
                    if isinstance(solution, dict):
                        for i in range(puzzle_size):
                            for j in range(puzzle_size):
                                cell = f"{rows[i]}{cols[j]}"
                                val = solution.get(cell, '0')
                                solution_grid[i,j] = int(val) if str(val).isdigit() else 0
                    else:
                        solution_grid = solution
                        
                    
                    
                    solution_placeholder.markdown(
                        display_sudoku(solution_grid, original_grid=puzzle_grid, size=puzzle_size), 
                        unsafe_allow_html=True
                    )
                    
                    if fitness >= 0.999:
                        st.success(f"Solved in {time.time() - start_time:.2f} seconds!")
                        break

st.markdown("---")
st.caption("Sudoku Solver - Supporting variable grid sizes (6x6, 9x9, 12x12, etc.)")