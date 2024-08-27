from typing import List

class Item:
    def __init__(self, id: int, size: float,arrival_time:int,departure_time:int):
        self.id = id
        self.size = size
        self.arrival_time=arrival_time
        self.departure_time=departure_time


class Bin:
    def __init__(self, bin_id: int, capacity: float = 1.0):
        self.bin_id = bin_id
        self.capacity = capacity
        self.remaining_capacity = capacity
        self.items: List[Item] = []
        self.departure_day = None

    def can_fit(self, item: Item) -> bool:
        return self.remaining_capacity >= item.size

    def add_item(self, item: Item):
        if self.can_fit(item):
            self.items.append(item)
            self.remaining_capacity -= item.size
        else:
            raise ValueError("Item doesn't fit in the bin")

    def remove_item(self, item: Item):
        if item in self.items:
            self.items.remove(item)
            self.remaining_capacity += item.size
        else:
            raise ValueError("Item not found in the bin")