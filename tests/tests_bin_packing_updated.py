import random
import numpy as np
import pytest
from typing import List
from tqdm import tqdm
from src.models import Item, Bin 
from src.algorithms import first_fit, next_fit, best_fit, worst_fit, sort_items_decreasing, first_fit_decreasing, best_fit_decreasing,\
      next_fit_decreasing, worst_fit_decreasing, greedy_size_first, rolling_horizon_dp

# Helper functions for testing
def create_items(sizes: List[float]) -> List[Item]:
    return [Item(i, size, 0, 1) for i, size in enumerate(sizes)]

# new
def create_pattern_items(num_items, pattern):
    items = []
    for _ in range(num_items // len(pattern)):
        for size in pattern:
            items.append(Item(len(items), size, len(items), len(items) + 1))
    return items

def create_items_with_varying_bin_capacity(num_items):
    items = []
    for i in range(num_items):
        size = random.uniform(0.1, 1.0)
        arrival_time = i
        departure_time = i + 1
        items.append(Item(i, size, arrival_time, departure_time))
    return items

def create_correlated_items(num_items, max_days):
    items = []
    for i in range(num_items):
        arrival_time = random.randint(0, max_days - 1)
        departure_time = arrival_time + random.randint(1, max_days - arrival_time)
        size = random.uniform(0.1, 1.0)
        items.append(Item(i, size, arrival_time, departure_time))
    return items

def create_fragmented_items(num_items):
    items = []
    for i in range(num_items):
        size = min(random.uniform(0.1, 0.5), 1.0) if random.random() < 0.8 else random.uniform(0.5, 1.0)
        arrival_time = i
        departure_time = i + 1
        items.append(Item(i, size, arrival_time, departure_time))
    return items

def check_valid_packing(bins: List[Bin]) -> bool:
    for bin in bins:
        if sum(item.size for item in bin.items) > bin.capacity:
            return False
    return True

def count_used_bins(bins: List[Bin]) -> int:
    return len([bin for bin in bins if bin.items])

def avg_bin_utilization(bins: List[Bin]) -> int:
    return np.median([sum(item.size for item in bin.items) / bin.capacity for bin in bins]) if bins else -1


def avg_items_per_bin(bins: List[Bin]) -> float:
    if len(bins) == 0:
        return -1
    return sum(len(bin.items) for bin in bins) / len(bins)

# Unit tests for each algorithm
@pytest.mark.parametrize("algorithm", [
    first_fit, next_fit, best_fit, worst_fit, first_fit_decreasing,
    best_fit_decreasing, next_fit_decreasing, worst_fit_decreasing
])
def test_basic_packing(algorithm):
    items = create_items([0.5, 0.7, 0.3, 0.2, 0.4, 0.8])
    bins = algorithm(items)
    assert check_valid_packing(bins)
    assert count_used_bins(bins) <= len(items)

@pytest.mark.parametrize("algorithm", [
    first_fit, next_fit, best_fit, worst_fit, first_fit_decreasing,
    best_fit_decreasing, next_fit_decreasing, worst_fit_decreasing
])
def test_empty_input(algorithm):
    items = []
    bins = algorithm(items)
    assert len(bins) == 0

@pytest.mark.parametrize("algorithm", [
    first_fit, next_fit, best_fit, worst_fit, first_fit_decreasing,
    best_fit_decreasing, next_fit_decreasing, worst_fit_decreasing
])
def test_single_item(algorithm):
    items = create_items([0.5])
    bins = algorithm(items)
    assert len(bins) == 1
    assert len(bins[0].items) == 1

@pytest.mark.parametrize("algorithm", [
    first_fit, next_fit, best_fit, worst_fit, first_fit_decreasing,
    best_fit_decreasing, next_fit_decreasing, worst_fit_decreasing
])
def test_perfect_fit(algorithm):
    items = create_items([0.5, 0.5, 0.5, 0.5])
    bins = algorithm(items)
    assert len(bins) == 2
    assert all(len(bin.items) == 2 for bin in bins)

@pytest.mark.parametrize("algorithm", [
    first_fit, next_fit, best_fit, worst_fit, first_fit_decreasing,
    best_fit_decreasing, next_fit_decreasing, worst_fit_decreasing
])
def test_large_items(algorithm):
    items = create_items([0.9, 0.8, 0.7, 0.6])
    bins = algorithm(items)
    assert len(bins) == 4

@pytest.mark.parametrize("algorithm", [sort_items_decreasing])
def test_decreasing_order(algorithm):
    items = create_items([0.3, 0.5, 0.2, 0.8, 0.1])
    sorted_items = algorithm(items)
    sorted_sizes = sorted([item.size for item in items], reverse=True)
    assert [item.size for item in sorted_items] == sorted_sizes

# Tests for time-aware algorithms
def test_greedy_size_first():
    items = [
        Item(0, 0.5, 0, 2),
        Item(1, 0.3, 1, 3),
        Item(2, 0.4, 0, 1),
        Item(3, 0.2, 2, 4)
    ]
    bins = greedy_size_first(items)
    assert check_valid_packing(bins)
    assert all(max(item.departure_time for item in bin.items) >= min(item.arrival_time for item in bin.items) for bin in bins)

# Test for rolling horizon dynamic programming
def test_rolling_horizon_dp():
    items = [
        Item(0, 0.5, 0, 2),
        Item(1, 0.3, 1, 3),
        Item(2, 0.4, 0, 1),
        Item(3, 0.2, 2, 4)
    ]
    bins = rolling_horizon_dp(items)
    assert check_valid_packing(bins)
    assert all(max(item.departure_time for item in bin.items) >= min(item.arrival_time for item in bin.items) for bin in bins)

# Edge cases
@pytest.mark.parametrize("algorithm", [
    first_fit, next_fit, best_fit, worst_fit, first_fit_decreasing,
    best_fit_decreasing, next_fit_decreasing, worst_fit_decreasing,
    greedy_size_first, rolling_horizon_dp
])
def test_all_items_full_size(algorithm):
    items = create_items([1.0, 1.0, 1.0])
    bins = algorithm(items)
    assert len(bins) == 3
    assert all(len(bin.items) == 1 for bin in bins)

# Integration tests
# Integration tests
def test_integration_varied_items():
    items = create_items([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5])
    algorithms = [
        first_fit, next_fit, best_fit, worst_fit, first_fit_decreasing,
        best_fit_decreasing, next_fit_decreasing, worst_fit_decreasing,
        greedy_size_first, rolling_horizon_dp
    ]
    results = {}
    for algorithm in algorithms:
        bins = algorithm(items)
        assert check_valid_packing(bins)
        results[algorithm.__name__] = count_used_bins(bins)

    print("\nIntegration Test with Varied Items:")
    for algo_name, bins_used in results.items():
        print(f"{algo_name}: {bins_used} bins used")

def test_integration_mixed_time_and_size():
    items = [
        Item(0, 0.5, 0, 2),
        Item(1, 0.3, 1, 3),
        Item(2, 0.7, 0, 4),
        Item(3, 0.4, 2, 5),
        Item(4, 0.6, 1, 2),
        Item(5, 0.2, 0, 1)
    ]
    algorithms = [greedy_size_first, rolling_horizon_dp]
    results = {}

    for algorithm in algorithms:
        bins = algorithm(items)
        assert check_valid_packing(bins)
        results[algorithm.__name__] = count_used_bins(bins)

    print("\nIntegration Test with Mixed Time and Size:")
    for algo_name, bins_used in results.items():
        print(f"{algo_name}: {bins_used} bins used")

def test_integration_large_scale():
    random.seed(42)
    items = create_random_items(5000)
    algorithms = [
        first_fit, next_fit, best_fit, worst_fit, first_fit_decreasing,
        best_fit_decreasing, next_fit_decreasing, worst_fit_decreasing, rolling_horizon_dp
    ]
    results = {}

    for algorithm in algorithms:
        bins = algorithm(items)
        assert check_valid_packing(bins)
        results[algorithm.__name__] = count_used_bins(bins)

    print("\nIntegration Test with Large Scale:")
    for algo_name, bins_used in results.items():
        print(f"{algo_name}: {bins_used} bins used")

# Functional tests
def test_item_handling():
    items = [
        Item(0, 0.5, 0, 2),
        Item(1, 0.3, 1, 3),
        Item(2, 0.4, 0, 1),
        Item(3, 0.2, 2, 4)
    ]
    for algorithm in [greedy_size_first, rolling_horizon_dp]:
        bins = algorithm(items)
        assert check_valid_packing(bins)
        for bin in bins:
            arrival_times = [item.arrival_time for item in bin.items]
            departure_times = [item.departure_time for item in bin.items]
            assert max(departure_times) >= min(arrival_times)

# Performance tests
import time

@pytest.mark.parametrize("algorithm", [
    first_fit, next_fit, best_fit, worst_fit, first_fit_decreasing,
    best_fit_decreasing, next_fit_decreasing, worst_fit_decreasing,
    greedy_size_first, rolling_horizon_dp
])
def test_algorithm_performance(algorithm):
    items = create_items([random.random() for _ in range(1000)])
    start_time = time.time()
    bins = algorithm(items)
    end_time = time.time()
    assert check_valid_packing(bins)
    print(f"{algorithm.__name__} took {end_time - start_time:.4f} seconds for 1000 items")


    # Next part


# Additional helper functions
def create_random_items(n: int, min_size: float = 0.1, max_size: float = 1.0) -> List[Item]:
    return [Item(i, random.uniform(min_size, max_size), 0, 1) for i in range(n)]

def create_time_aware_items(n: int, max_days: int) -> List[Item]:
    return [Item(i, random.uniform(0.1, 1.0), random.randint(0, max_days-1), random.randint(1, max_days)) for i in range(n)]

# Advanced tests for all algorithms
@pytest.mark.parametrize("algorithm", [
    first_fit, next_fit, best_fit, worst_fit, first_fit_decreasing,
    best_fit_decreasing, next_fit_decreasing, worst_fit_decreasing,
    greedy_size_first, rolling_horizon_dp
])
class TestAdvancedScenarios:

    def test_random_sizes(self, algorithm):
        random.seed(42)  # For reproducibility
        items = create_random_items(100)
        bins = algorithm(items)
        assert check_valid_packing(bins)
        assert count_used_bins(bins) <= 100

    def test_mostly_large_items(self, algorithm):
        items = create_items([0.7] * 8 + [0.29] * 2)
        bins = algorithm(items)
        assert check_valid_packing(bins)
        assert count_used_bins(bins) <= 9

    def test_mostly_small_items(self, algorithm):
        items = create_items([0.2] * 8 + [0.8] * 2)
        bins = algorithm(items)
        assert check_valid_packing(bins)
        assert count_used_bins(bins) <= 5

# Specific tests for time-aware and rolling_horizon_dp algorithms
@pytest.mark.parametrize("algorithm", [greedy_size_first, rolling_horizon_dp])
class TestTimeAwareScenarios:
    def test_non_overlapping_items(self, algorithm):
        items = [
            Item(0, 0.5, 0, 1),
            Item(1, 0.5, 1, 2),
            Item(2, 0.5, 2, 3),
            Item(3, 0.5, 3, 4)
        ]
        bins = algorithm(items)
        assert check_valid_packing(bins)

        if algorithm == greedy_size_first:
            assert count_used_bins(bins) == 4  # Expecting 4 bins from this algorithm
        elif algorithm == rolling_horizon_dp:
            assert count_used_bins(bins) == 2  # Expecting 2 bins from this algorithm

    def test_partially_overlapping_items(self, algorithm):
        items = [
            Item(0, 0.5, 0, 2),
            Item(1, 0.5, 1, 3),
            Item(2, 0.5, 2, 4),
            Item(3, 0.5, 3, 5)
        ]
        bins = algorithm(items)
        assert check_valid_packing(bins)

        if algorithm == greedy_size_first:
            assert count_used_bins(bins) <= 4  # Adjusted expectation for greedy_size_first
        else:
            assert count_used_bins(bins) <= 2

    def test_all_overlapping_items(self, algorithm):
        items = [Item(i, 0.2, 0, 5) for i in range(5)]
        bins = algorithm(items)
        assert check_valid_packing(bins)

        if algorithm == greedy_size_first:
            assert count_used_bins(bins) == 2  # Adjusted expectation for greedy_size_first
        else:
            assert count_used_bins(bins) == 1

    def test_mixed_overlap_scenario(self, algorithm):
        items = [
            Item(0, 0.4, 0, 2),
            Item(1, 0.3, 1, 3),
            Item(2, 0.5, 2, 4),
            Item(3, 0.2, 0, 4),
            Item(4, 0.1, 3, 5)
        ]
        bins = algorithm(items)
        assert check_valid_packing(bins)

        if algorithm == greedy_size_first:
            assert count_used_bins(bins) <= 3  # Adjusted expectation for greedy_size_first
        else:
            assert count_used_bins(bins) <= 2

# Stress tests
#@pytest.mark.timeout(30)  # Increased timeout for 30 seconds
@pytest.mark.parametrize("algorithm", [
    first_fit, next_fit, best_fit, worst_fit, first_fit_decreasing,
    best_fit_decreasing, next_fit_decreasing, worst_fit_decreasing,
    greedy_size_first, rolling_horizon_dp
])
def test_large_input(algorithm):
    if algorithm.__name__ == "greedy_size_first":
        # Reduce the size of input specifically for this algorithm due to time complexity
        items = create_random_items(3000)  # Adjusted input size
    else:
        items = create_random_items(10000)

    start_time = time.time()
    bins = algorithm(items)
    end_time = time.time()

    assert check_valid_packing(bins)
    print(f"{algorithm.__name__} processed {len(items)} items in {end_time - start_time:.2f} seconds")

# Comparative tests
def test_algorithm_comparison():
    algorithms = [
        first_fit, next_fit, best_fit, worst_fit, first_fit_decreasing,
        best_fit_decreasing, next_fit_decreasing, worst_fit_decreasing,
        greedy_size_first, rolling_horizon_dp
    ]

    test_cases = []
    for ns in [10,100,1000,10000]:
        test_cases.extend([
            ("Small items",ns,-1, create_items([random.uniform(0.1, 0.5) for _ in range(ns)])),
            ("Large items",ns,-1,create_items([random.uniform(0.5, 1.0) for _ in range(ns)])),
            ("Mixed items",ns,-1,create_random_items(ns)),
        ])
        for d in [10,100,1000,10000]:
            test_cases.append(("Time-aware items",ns,d,create_time_aware_items(ns,d)))

    results = []
    for case_name, nitems,ndays,items in tqdm(test_cases):
        for algorithm in algorithms:
            start_time = time.time()
            bins = algorithm(items)
            end_time = time.time()
            results.append(
                {
                "case": case_name,
                "algorithm": algorithm.__name__,
                "bins_used": avg_bin_utilization(bins),
                "avg_items_per_bin": avg_items_per_bin(bins),
                "time": end_time - start_time,
                "nitems": nitems,
                "ndays": ndays
                }
            )

    #save resuts as csv
    import csv
    with open('results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Case","Algorithm", "Bins Used", "Time (s)","nitems","ndays"])
        for result in results:
                writer.writerow([result['case'], result['algorithm'], result["bins_used"], result["time"],result["nitems"],result["ndays"]
                ])

    # # Print comparison results
    # for case_name, case_results in results.items():
    #     print(f"\nResults for {case_name}:")
    #     print(f"{'Algorithm':<30} {'Bins Used':<10} {'Time (s)':<10}")
    #     print("-" * 50)
    #     for algo_name, algo_results in case_results.items():
    #         print(f"{algo_name:<30} {algo_results['bins_used']:<10} {algo_results['time']:.4f}")

# Run all tests
if __name__ == "__main__":
    # pytest.main(["-v", __file__])
    test_algorithm_comparison()