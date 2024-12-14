small_map = [
    [1, 0, 0, 1, 0],
    [1, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0],
]

if __name__ == "__main__":
    from hurry_taxi.utils.show_map import show_grid_map
    show_grid_map(small_map)