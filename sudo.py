def cross(A, B):
    """Cross product of elements in A and elements in B."""
    return [a+b for a in A for b in B]

def initialize_structures(size):
    """Initialize all Sudoku structures with proper cell IDs"""
    if size < 1:
        raise ValueError(f"Invalid Sudoku size: {size}")

    # Generate row labels (A, B, C, etc.)
    rows = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:size]
    
    # Generate column labels (1, 2, 3, etc.) - MUST start from 1
    cols = [str(i+1) for i in range(size)]
    
    # Create all valid cell IDs
    squares = cross(rows, cols)
    
    # Calculate subgrid dimensions
    subgrid_rows = int(size**0.5)
    while size % subgrid_rows != 0:
        subgrid_rows -= 1
    subgrid_cols = size // subgrid_rows
    
    # Generate valid digits (1-9, A-Z for larger sizes)
    digits = [str(i+1) for i in range(min(size, 9))]
    if size > 9:
        digits += [chr(ord('A') + i) for i in range(size - 9)]
    
    # Build unitlist - FIXED implementation
    unitlist = []
    
    # Columns (A1-L1, A2-L2, etc.)
    for c in cols:
        unitlist.append([f"{r}{c}" for r in rows])
    
    # Rows (A1-A12, B1-B12, etc.)
    for r in rows:
        unitlist.append([f"{r}{c}" for c in cols])
    
    # Subgrids - fixed to handle multi-digit columns
    for i in range(0, size, subgrid_rows):
        for j in range(0, size, subgrid_cols):
            subgrid = []
            for x in range(subgrid_rows):
                for y in range(subgrid_cols):
                    row = rows[i+x]
                    col = cols[j+y]
                    subgrid.append(f"{row}{col}")
            unitlist.append(subgrid)
    
    units = {s: [u for u in unitlist if s in u] for s in squares}
    peers = {s: set().union(*units[s]) - {s} for s in squares}
    
    return rows, cols, squares, unitlist, units, peers, subgrid_rows, subgrid_cols, digits

def grid_values(grid, size):
    """Convert grid into a dict of {square: char} with proper size handling"""
    rows, cols, squares, *_ = initialize_structures(size)
    
    if isinstance(grid, dict):
        # Verify all keys use proper 1-based columns
        for key in grid.keys():
            if not (key[0] in rows and key[1:] in cols):
                raise ValueError(f"Invalid cell ID: {key}")
        return grid
    
    if isinstance(grid, str):
        lines = [line.strip() for line in grid.split('\n') if line.strip()]
        
        # Handle size indicator if present
        if lines and lines[0].isdigit() and int(lines[0]) == size:
            lines = lines[1:]  # Skip the size line
        
        # Combine all puzzle lines and filter out non-puzzle characters
        puzzle_chars = []
        for line in lines:
            for c in line:
                if c in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.':
                    puzzle_chars.append(c)
        
        if len(puzzle_chars) != size * size:
            raise ValueError(f"Expected {size*size} cells, got {len(puzzle_chars)}")
        
        # Create dictionary with proper cell IDs
        grid_dict = {}
        index = 0
        for row in rows:
            for col in cols:
                cell_id = f"{row}{col}"
                val = puzzle_chars[index]
                grid_dict[cell_id] = val if val != '0' else '.'
                index += 1
        return grid_dict
    
    raise ValueError("Unsupported grid format")

def parse_grid(grid, size):
    """Convert grid to a dict of possible values with proper units"""
    rows, cols, squares, unitlist, units, peers, _, _, digits = initialize_structures(size)
    values = {s: digits.copy() for s in squares}
    
    grid_dict = grid_values(grid, size)
    
    for s, d in grid_dict.items():
        if d != '.' and d in digits:
            if not assign(values, s, d, peers):
                return False
    return values

def assign(values, s, d, peers):
    """Eliminate all other values except d from values[s]"""
    other_values = [v for v in values[s] if v != d]
    for d2 in other_values:
        if not eliminate(values, s, d2, peers):
            return False
    return values

def eliminate(values, s, d, peers):
    """Eliminate d from values[s]"""
    if d not in values[s]:
        return values
    
    values[s].remove(d)
    
    if len(values[s]) == 0:
        return False
    elif len(values[s]) == 1:
        d2 = values[s][0]
        for s2 in peers[s]:
            if not eliminate(values, s2, d2, peers):
                return False
    return values

def solve(grid, size):
    """Solve a Sudoku puzzle and return dict of single values"""
    values = parse_grid(grid, size)
    if not values:
        return False
    
    solution = search(values, size)
    if not solution:
        return False
    
    # Convert to single values
    solved_dict = {}
    for s in solution:
        val = solution[s]
        if isinstance(val, list):
            if len(val) == 1:
                solved_dict[s] = val[0]
            else:
                return False  # Invalid solution
        else:
            solved_dict[s] = val
    
    return solved_dict

def search(values, size):
    """Depth-first search"""
    if values is False:
        return False
    
    _, _, squares, _, _, peers, _, _, _ = initialize_structures(size)
    
    if all(len(values[s]) == 1 for s in squares):
        return {s: values[s][0] for s in squares}  # Convert to single values
    
    n, s = min((len(values[s]), s) for s in squares if len(values[s]) > 1)
    
    for d in values[s]:
        new_values = {k: v.copy() for k, v in values.items()}
        result = search(assign(new_values, s, d, peers), size)
        if result:
            return result
    return False