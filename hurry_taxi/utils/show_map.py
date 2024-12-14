def show_grid_map(map):
    n = len(map)
    # Print the grid to the console
    print("\n" + "-" * (n* 2 + 1))
    for row in map:
        print("|" + " ".join([' ' if x==0 else 'C' for x in row]) + "|")
    print("-" * (n * 2 + 1))