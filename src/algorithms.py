from typing import List

from src.models import Bin, Item

def first_fit(items: List[Item], existing_bins: List[Bin] = None) -> List[Bin]:
    bins = existing_bins or []
    for item in items:
        fitted = False
        for bin in bins:
            if bin.can_fit(item):
                bin.add_item(item)
                fitted = True
                break
        if not fitted:
            new_bin = Bin(len(bins))
            new_bin.add_item(item)
            bins.append(new_bin)
    return bins


def next_fit(items: List[Item]) -> List[Bin]:
    # Handle the edge case where the input list is empty
    if not items:
        return []
    
    bins = [Bin(0)]
    current_bin = 0
    for item in items:
        if bins[current_bin].can_fit(item):
            bins[current_bin].add_item(item)
        else:
            new_bin = Bin(len(bins))
            new_bin.add_item(item)
            bins.append(new_bin)
            current_bin += 1
    return bins


def best_fit(items: List[Item]) -> List[Bin]:
    bins = []
    for item in items:
        best_bin = None
        best_remaining_capacity = float('inf')
        for bin in bins:
            if bin.can_fit(item) and bin.remaining_capacity < best_remaining_capacity:
                best_bin = bin
                best_remaining_capacity = bin.remaining_capacity
        if best_bin:
            best_bin.add_item(item)
        else:
            new_bin = Bin(len(bins))
            new_bin.add_item(item)
            bins.append(new_bin)
    return bins


def worst_fit(items: List[Item]) -> List[Bin]:
    bins = []
    for item in items:
        worst_bin = None
        worst_remaining_capacity = -1
        for bin in bins:
            if bin.can_fit(item) and bin.remaining_capacity > worst_remaining_capacity:
                worst_bin = bin
                worst_remaining_capacity = bin.remaining_capacity
        if worst_bin:
            worst_bin.add_item(item)
        else:
            new_bin = Bin(len(bins))
            new_bin.add_item(item)
            bins.append(new_bin)
    return bins


def sort_items_decreasing(items: List[Item]) -> List[Item]:
    return sorted(items, key=lambda x: x.size, reverse=True)

def first_fit_decreasing(items: List[Item]) -> List[Bin]:
    sorted_items = sort_items_decreasing(items)
    return first_fit(sorted_items)

def best_fit_decreasing(items: List[Item]) -> List[Bin]:
    sorted_items = sort_items_decreasing(items)
    return best_fit(sorted_items)

def worst_fit_decreasing(items: List[Item]) -> List[Bin]:
    sorted_items = sort_items_decreasing(items)
    return worst_fit(sorted_items)

def next_fit_decreasing(items: List[Item]) -> List[Bin]:
    sorted_items = sort_items_decreasing(items)
    return next_fit(sorted_items)

def time_aware_harmonic_algorithm(items: List[Item], k: int = 5) -> List[Bin]:
    # Sort items by departure time, then by size (descending)
    sorted_items = sorted(items, key=lambda x: (-x.size))

    bins = []
    placed_items = set()

    for item in sorted_items:
        if item in placed_items:
            continue

        # Try to place the item in an existing bin
        placed = False
        for bin in bins:
            if bin.can_fit(item):
                bin.add_item(item)
                placed_items.add(item)
                placed = True
                break

        if not placed:
            # If not placed, create a new bin
            new_bin = Bin(len(bins))
            new_bin.add_item(item)
            placed_items.add(item)
            bins.append(new_bin)

            # Try to fill the new bin with smaller items
            for potential_item in sorted_items:
                if potential_item not in placed_items and new_bin.can_fit(potential_item):
                    new_bin.add_item(potential_item)
                    placed_items.add(potential_item)

    return bins

def rolling_horizon_dp(items: List[Item], horizon: int = 10) -> List[Bin]:
    def solve_dp(current_items: List[Item]) -> List[Bin]:
        n = len(current_items)
        max_bins = n  # Worst case: one item per bin

        # dp[i][j] represents the minimum wasted space when packing items 0 to i using j bins
        dp = [[float('inf')] * (max_bins + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        # decision[i][j] stores the number of items packed in the last bin for the optimal solution
        decision = [[0] * (max_bins + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            for j in range(1, min(i, max_bins) + 1):
                for k in range(i):
                    items_in_last_bin = i - k
                    last_bin_content = sum(item.size for item in current_items[k:i])
                    if last_bin_content <= 1:  # Bin capacity constraint
                        wasted_space = 1 - last_bin_content
                        if dp[k][j-1] + wasted_space < dp[i][j]:
                            dp[i][j] = dp[k][j-1] + wasted_space
                            decision[i][j] = items_in_last_bin

        # Find the minimum number of bins needed
        min_bins = max_bins
        for j in range(max_bins + 1):
            if dp[n][j] < float('inf'):
                min_bins = j
                break

        # Reconstruct the solution
        bins = []
        i, j = n, min_bins
        while i > 0:
            items_in_last_bin = decision[i][j]
            new_bin = Bin(len(bins))
            for item in current_items[i - items_in_last_bin:i]:
                if new_bin.can_fit(item):
                    new_bin.add_item(item)
                else:
                    # If an item doesn't fit, start a new bin
                    bins.append(new_bin)
                    new_bin = Bin(len(bins))
                    new_bin.add_item(item)
            bins.append(new_bin)
            i -= items_in_last_bin
            j -= 1

        return list(reversed(bins))

    bins = []
    remaining_items = items.copy()

    while remaining_items:
        current_items = remaining_items[:horizon]
        current_solution = solve_dp(current_items)

        # Add all bins from the current solution to the overall solution
        bins.extend(current_solution)

        # Remove packed items from remaining_items
        for bin in current_solution:
            for item in bin.items:
                if item in remaining_items:
                    remaining_items.remove(item)

    return bins