import pandas as pd
import io
def grid_to_markdown(grid, cookbook, agent_pos=None, include_indices=False) -> str:
    width, height, n_kinds = grid.shape
    inv_index = cookbook.index.reverse_contents  # index -> item name
    table = []
    for y in range(height):  # row by row
        row = []
        for x in range(width):
            cell_items = [inv_index[k] for k in range(1, n_kinds) if grid[x, y, k] == 1]
            cell_repr = ",".join(cell_items) if cell_items else "."
            # Mark the agent start location
            if agent_pos and (x, y) == agent_pos:
                cell_repr = f"Agent({cell_repr})" if cell_repr != "." else "Agent"
            row.append(cell_repr)
        table.append(row)

    if include_indices:
        # Domain indexing: (0,0) at top-left, x increases right, y increases down.
        col_labels = [f"x={x}" for x in range(width)]
        row_labels = [f"y={y}" for y in range(height)]
        df = pd.DataFrame(table, columns=col_labels, index=row_labels)
        return df.to_markdown(index=True)

    df = pd.DataFrame(table)
    return df.to_markdown(index=False, headers=[])


def parse_markdown_grid(md: str) -> pd.DataFrame:
        lines = [line for line in md.strip().splitlines() if not line.startswith('|:')]
        data = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", engine="python")
        # drop empty columns due to markdown '|' separators at start/end
        data = data.dropna(axis=1, how="all")
        data = data.applymap(lambda x: str(x).strip())
        return data

# def verify_stone_break(grid_before: str, grid_after: str) -> float:
#     try:
#         df_before = parse_markdown_grid(grid_before)
#         df_after = parse_markdown_grid(grid_after)
#     except Exception as e:
#         print(f"Error parsing grids: {e}")
#         return 0.0

#     # Compare cell-by-cell for 'stone'
#     stones_before = (df_before == "stone")
#     stones_after = (df_after == "stone")

#     # A stone is "broken" if it existed before but no longer after
#     broken_stones = (stones_before & ~stones_after).sum().sum()

#     return 1.0 if broken_stones > 0 else 0.0





