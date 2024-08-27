import random
from collections import defaultdict
import time
from typing import Callable, List, Dict, Any, Set
from joblib import Parallel, delayed
from src.models import Bin, Item


def generate_random_items(num_items: int, max_days: int) -> List[Item]:
    items = []
    for i in range(num_items):
        size = round(random.uniform(0.01, 1.0), 2)
        arrival_time = random.randint(0, max_days - 2)
        departure_time = random.randint(arrival_time + 1, min(arrival_time + 3, max_days))
        items.append(Item(i, size, arrival_time, departure_time))
    return items

def _compare_algo(algo_name, algo_func, items, max_days):
    start_time = time.time()
    daily_state = simulate_dynamic_packing(items, max_days, algo_func, algo_name)
    end_time = time.time()

    algo_results = {
        "bins_used": [],
        "avg_utilization": [],
        "runtime": end_time - start_time
    }

    for day in range(max_days + 1):
        if day in daily_state:
            bins = daily_state[day]["bins"]
            algo_results["bins_used"].append(len(bins))
            if bins:
                avg_util = sum((1.0 - bin.remaining_capacity) for bin in bins) / len(bins)
                algo_results["avg_utilization"].append(avg_util)
            else:
                algo_results["avg_utilization"].append(0)
        else:
            algo_results["bins_used"].append(0)
            algo_results["avg_utilization"].append(0)

    return algo_name,algo_results

def compare_algorithms(items: List[Item], max_days: int, algorithms: Dict[str, Callable]):

    algos=algorithms.items()
    # multiprocessing to threading
    results=Parallel(n_jobs=-1,backend="threading"
                     )(delayed(_compare_algo)(algo_name,algo_func,items,max_days) for algo_name,algo_func in algos)
    #wait for all to finish
    results = {algo_name: algo_results for algo_name, algo_results in results}
    return results

def departed_items_percentage(bins):
    total_items = sum(len(bin.items) for bin in bins)
    non_departed_items = sum(1 for bin in bins for item in bin.items if bin.departure_day and item.departure_time > bin.departure_day)
    return (total_items-non_departed_items) / total_items if total_items > 0 else 0

def simulate_dynamic_packing(items: List[Item], max_days: int, packing_algorithm, algorithm_name: str) -> Dict[int, Any]:
    print(f'\nStarting simulation with {algorithm_name}')
    print('INSIDE SIMULATE DYNAMIC PACKING')
    bins: List[Bin] = []
    active_items: List[Item] = []
    departed_items: Set[Item] = set()  # Use a set to prevent duplicates
    bin_departure_day: defaultdict[int, List[int]] = defaultdict(list)
    next_bin_id = 0
    total_bins_used = 0

    daily_state = {}  # Dictionary to hold the state of bins for each day

    for day in range(max_days + 1):
        print(f"Day {day}:")

        new_items = [item for item in items if item.arrival_time == day]
        active_items.extend(new_items)

        print("  Existing/Arrived items:")
        if active_items:
            for item in active_items:
                status = "Existing" if item.arrival_time < day else "New"
                print(f"    {status} Item {item.id}: Size={item.size:.2f}, Arrival=Day {item.arrival_time}, Departure=Day {item.departure_time}")
        else:
            print("    No active items")

        departed = [item for item in active_items if item.departure_time == day]
        for item in departed:
            active_items.remove(item)
            departed_items.add(item)

        items_to_pack = [item for item in active_items if item.departure_time == day + 1]
        if items_to_pack:
            print("  Packing bins for tomorrow's departure:")
            other_items = [item for item in active_items if item.departure_time > day + 1]

            new_bins = packing_algorithm(items_to_pack)

            for bin in new_bins:
                for item in other_items[:]:
                    if bin.can_fit(item):
                        bin.add_item(item)
                        other_items.remove(item)
                        items_to_pack.append(item)

            for bin in new_bins:
                if bin.items:  # Only process bins that actually contain items
                    bin.bin_id = next_bin_id
                    next_bin_id += 1
                    bins.append(bin)
                    total_bins_used += 1
                    earliest_departure = min(item.departure_time for item in bin.items)
                    bin_departure_day[earliest_departure].append(bin.bin_id)
                    total_size = sum(item.size for item in bin.items)
                    print(f"    Bin {bin.bin_id}: Items={[item.id for item in bin.items]} -> Total Size of Bin = {total_size:.2f}")
                else:
                    print("    Warning: Empty bin created and discarded")

            active_items = [item for item in active_items if item not in items_to_pack]
        else:
            print("  No items to pack for tomorrow")

        if day in bin_departure_day:
            print("  Departed bins:")
            for bin_id in bin_departure_day[day]:
                departing_bin = next(bin for bin in bins if bin.bin_id == bin_id)
                total_size = sum(item.size for item in departing_bin.items)
                print(f"    Bin {bin_id}: Items={[item.id for item in departing_bin.items]} -> Total Size of Bin = {total_size:.2f}")
                for item in departing_bin.items:
                    if item not in departed_items:
                        departed_items.add(item)
                    else:
                        print(f"WARNING: Item {item.id} was already marked as departed!")
                active_items = [item for item in active_items if item not in departing_bin.items]

            bins = [bin for bin in bins if bin.bin_id not in bin_departure_day[day]]
        # else:
        #     print("  No bins departed today.")

        daily_state[day] = {
            "bins": [bin for bin in bins],            # Copy the list of bins
            "active_items": active_items[:],       # Copy the list of active items
            "departed_items": departed_items.copy()  # Copy the set of departed items
        }

        print(f"  Active items: {len(active_items)}")
        print(f"  Current bins: {len(bins)}")
        print(f"  Total bins used: {total_bins_used}")
        print(f"  Total departed items: {len(departed_items)}")
        print()

    print("Simulation Complete")
    print(f"Algorithm used: {algorithm_name}")
    print(f"Total bins used: {total_bins_used}")
    print(f"Total items processed: {len(items)}")
    print(f"Total departed items: {len(departed_items)}")
    if active_items:
        print(f"Warning: {len(active_items)} items were not departed by the end of the simulation.")

    # Sanity check
    if len(departed_items) > len(items):
        print("ERROR: More items departed than were processed!")
        extra_items = len(departed_items) - len(items)
        print(f"Extra departed items: {extra_items}")

    return daily_state
