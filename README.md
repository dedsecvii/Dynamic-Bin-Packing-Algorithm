# **Dynamic Bin Packing Algorithm for Periodic Shipping Scenarios**

## **Introduction**

This project provides implementations of multiple algorithms to solve the **Bin Packing Problem** in Python. The goal is to minimize the number of bins needed to pack items of varying sizes, each of which must fit into a bin with a fixed capacity (typically 1 unit) with respesct to arrival and departure times. Different algorithms are used to find optimal or near-optimal solutions for this problem.

## **Algorithms Included**

The project implements the following algorithms:

1. **First Fit (FF)**
2. **Next Fit (NF)**
3. **Best Fit (BF)**
4. **Worst Fit (WF)**
5. **First Fit Decreasing (FFD)**
6. **Next Fit Decreasing (NFD)**
7. **Best Fit Decreasing (BFD)**
8. **Worst Fit Decreasing (WFD)**
9. **Greedy Size-First**
10. **Rolling Horizon Dynamic Programming (RHDP)**

## **Algorithms Overview**

### 1. **First Fit (FF)**

The **First Fit** algorithm places each item into the first bin that has enough space. If no bin can accommodate the item, a new bin is created.

#### **Steps**:
1. For each item, place it into the first bin with sufficient space.
2. If no bin can accommodate the item, start a new bin.

#### **Time Complexity**:  
- **O(n²)** in the worst case, where `n` is the number of items.

### 2. **Next Fit (NF)**

The **Next Fit** algorithm places each item into the current bin if it fits. If it doesn’t fit, a new bin is started for the next item.

#### **Steps**:
1. Place the item into the current bin if there is enough space.
2. If the item doesn’t fit, close the current bin and open a new one.

#### **Time Complexity**:  
- **O(n)**, where `n` is the number of items.

### 3. **Best Fit (BF)**

The **Best Fit** algorithm places each item into the bin that will leave the least remaining space after the item is added (i.e., the best fit). If no bin can accommodate the item, a new bin is started.

#### **Steps**:
1. For each item, place it into the bin with the least remaining space that can still fit the item.
2. If no bin can fit the item, create a new bin.

#### **Time Complexity**:  
- **O(n log n)** (due to searching for the best bin in a sorted list).

### 4. **Worst Fit (WF)**

The **Worst Fit** algorithm places each item into the bin that leaves the most remaining space after the item is added (i.e., the worst fit). If no bin can accommodate the item, a new bin is created.

#### **Steps**:
1. For each item, place it into the bin with the most remaining space.
2. If no bin can fit the item, create a new bin.

#### **Time Complexity**:  
- **O(n log n)** (due to searching for the worst bin).

### 5. **First Fit Decreasing (FFD)**

The **First Fit Decreasing** algorithm first sorts the items in descending order of size and then applies the First Fit algorithm.

#### **Steps**:
1. Sort the items in decreasing order of size.
2. Apply the First Fit algorithm.

#### **Time Complexity**:  
- **O(n log n)** for sorting the items, and **O(n²)** for the First Fit step.
- **Overall**: **O(n²)**.

### 6. **Next Fit Decreasing (NFD)**

The **Next Fit Decreasing** algorithm sorts items in descending order of size before applying the Next Fit algorithm.

#### **Steps**:
1. Sort items in descending order of size.
2. Apply the Next Fit algorithm.

#### **Time Complexity**:  
- **O(n log n)** for sorting and **O(n)** for packing.
- **Overall**: **O(n log n)**.

### 7. **Best Fit Decreasing (BFD)**

The **Best Fit Decreasing** algorithm first sorts the items in descending order and then applies the Best Fit algorithm.

#### **Steps**:
1. Sort items in descending order of size.
2. Apply the Best Fit algorithm.

#### **Time Complexity**:  
- **O(n log n)** for sorting and **O(n log n)** for packing.
- **Overall**: **O(n log n)**.

### 8. **Worst Fit Decreasing (WFD)**

The **Worst Fit Decreasing** algorithm first sorts items in descending order of size and then applies the Worst Fit algorithm.

#### **Steps**:
1. Sort items in descending order of size.
2. Apply the Worst Fit algorithm.

#### **Time Complexity**:  
- **O(n log n)** for sorting and **O(n log n)** for packing.
- **Overall**: **O(n log n)**.

### 9. **Greedy Size-First**

In this algorithm, the items are sorted in decreasing order based on their size. Each item is then placed into the first available bin that can accommodate it.

#### **Steps**:
1. Sort the items in decreasing order of size.
2. Use a greedy strategy to place each item in the first bin that can fit it.

#### **Time Complexity**:  
- **O(n log n)** for sorting and **O(n²)** for the greedy placement.
- **Overall**: **O(n²)**.

### 10. **Rolling Horizon Dynamic Programming (RHDP)**

The **Rolling Horizon Dynamic Programming (RHDP)** algorithm is a more advanced approach that uses dynamic programming to optimize packing over a fixed horizon of items. It reduces the number of bins by minimizing wasted space through backtracking and reconstructing the optimal packing solution.

#### **Steps**:
1. Sort the items in decreasing order of size.
2. Apply dynamic programming over chunks (horizons) of items, minimizing wasted space within each horizon.
3. Repeat until all items are packed.

#### **Time Complexity**:  
- **O(n * h²)**, where `n` is the number of items and `h` is the size of the horizon for each DP step.

## **How to Run the Project**

1. Clone the repository - https://github.com/dedsecvii/Dynamic-Bin-Packing-Algorithm.git.
2. Install required dependencies (if specified in `requirements.txt`).
3. Run the `main.py` file to run the project
4. In the terminal of `main.py` go to visualization directory - `cd visualization`
5. Run the command - `streamlit run graph_maker.py` to enable Streamlit-based visualizations and interface.