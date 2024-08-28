import csv
import math
import random
import subprocess
import threading
import time
import tkinter as tk
import webbrowser
from tkinter import ttk, filedialog
from typing import List
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tqdm import tqdm

from src.models import Bin, Item
from src.utils import compare_algorithms, departed_items_percentage, generate_random_items, simulate_dynamic_packing
from tests.tests_bin_packing_updated import create_correlated_items,avg_items_per_bin,avg_bin_utilization, create_fragmented_items, create_items, \
    create_items_with_varying_bin_capacity, create_random_items, create_time_aware_items, count_used_bins, create_pattern_items
from src.algorithms import first_fit, next_fit, best_fit, worst_fit, sort_items_decreasing, first_fit_decreasing, best_fit_decreasing,\
      next_fit_decreasing, worst_fit_decreasing, greedy_size_first, rolling_horizon_dp


class BinPackingGUI:
    def __init__(self, master):
        self.current_runtime = 1
        self.master = master
        master.title("Dynamic Bin Packing Algorithm")
        master.geometry("1200x800")

        self.items = generate_random_items(100, 100)
        self.algorithms = {
            "First Fit": first_fit,
            "Next Fit": next_fit,
            "Best Fit": best_fit,
            "Worst Fit": worst_fit,
            "First Fit Decreasing": first_fit_decreasing,
            "Next Fit Decreasing": next_fit_decreasing,
            "Best Fit Decreasing": best_fit_decreasing,
            "Worst Fit Decreasing": worst_fit_decreasing,
            "Greedy Size-First": greedy_size_first,
            "Rolling Horizon DP": rolling_horizon_dp,
        }

        self.daily_state = {}  # To hold the state of bins for each day

        # Sidebar
        self.sidebar = tk.Frame(master, width=200, bg='lightgray')
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)

        # Algorithm selection
        self.algo_label = tk.Label(self.sidebar, text="Select Algorithm:")
        self.algo_label.pack(pady=10)
        self.algo_dropdown = ttk.Combobox(self.sidebar, values=list(self.algorithms.keys()))
        self.algo_dropdown.pack(pady=5)
        self.algo_dropdown.current(0)
        #pad all buttons to same size
        padding=10
        self.run_button = tk.Button(self.sidebar, text=f"{'Run Algorithm':^25}", command=self.run_algorithm)
        self.run_button.pack(pady=10)

        # Data input
        self.upload_button = tk.Button(self.sidebar, text=f"{'Upload Data':^30}", command=self.upload_data)
        self.upload_button.pack(pady=5)
        self.generate_button = tk.Button(self.sidebar, text=f"{'Make Random Data':^20}", command=self.generate_data)
        self.generate_button.pack(pady=5)

        # Data output
        self.download_button = tk.Button(self.sidebar, text=f"{'Download Data':^28}", command=self.download_data)
        self.download_button.pack(pady=5)

        self.show_data_button = tk.Button(self.sidebar, text=f"{'Show Data':^30}", command=self.show_data)
        self.show_data_button.pack(pady=5)

        # Compare button
        self.compare_button = tk.Button(self.sidebar, text=f"{'Compare Algorithms':^25}", command=self.run_comparison)
        self.compare_button.pack(pady=20)

        #streamlit
        self.launch_streamlit_button = tk.Button(self.sidebar, text=f"{'Show Comparison Graphs':^20}", command=self.launch_streamlit_app)
        self.launch_streamlit_button.pack(pady=20)


        self.content = tk.Frame(master)
        self.content.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Top bar with settings and help icons
        self.top_bar = tk.Frame(self.content, bg='lightgray')
        self.top_bar.pack(fill=tk.X)
        
        timeline_max_days=100
        self.timeline = tk.Scale(self.top_bar, from_=0, to=timeline_max_days, orient=tk.HORIZONTAL, label="Timeline (Days)",
                                 command=self.update_visualization)
        self.timeline.pack(fill=tk.X, padx=10, pady=10, side=tk.BOTTOM)

        # Visualization area
        self.viz_frame = tk.Frame(self.content)
        self.viz_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)


        self.info_panel = tk.Frame(self.viz_frame, width=200, bg='lightgray')
        self.info_panel.pack(side=tk.RIGHT, fill=tk.Y)
        tk.Label(self.info_panel, text="Bin Information", bg='lightgray').pack(pady=10)
        self.item_info_labels = []
        for _ in range(100):
            label = tk.Label(self.info_panel, text="", bg='lightgray', anchor="w")
            label.pack()
            self.item_info_labels.append(label)

        self.canvas_frame = tk.Frame(self.viz_frame)
        self.canvas_frame.pack(expand=True, fill=tk.BOTH)

        self.h_scroll = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        self.v_scroll = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.viz_area = tk.Canvas(self.canvas_frame, bg='white',scrollregion=(0, 0, 2000, 2000),
                                  xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)
        self.viz_area.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        #
        self.h_scroll.config(command=self.viz_area.xview)
        self.v_scroll.config(command=self.viz_area.yview)

        # Performance metrics
        self.metrics_frame = tk.Frame(self.viz_area)
        self.metrics_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        self.bins_label = tk.Label(self.metrics_frame, text="Bins: 0")
        self.bins_label.pack(side=tk.LEFT, padx=10)
        self.util_label = tk.Label(self.metrics_frame, text="Utilization: 0%")
        self.util_label.pack(side=tk.LEFT, padx=10)
        self.avg_items_per_bin_label = tk.Label(self.metrics_frame, text="Avg Items per Bin: 0")
        self.avg_items_per_bin_label.pack(side=tk.LEFT, padx=10)
        self.Efficiency_label = tk.Label(self.metrics_frame, text="Efficiency: 0")
        self.Efficiency_label.pack(side=tk.LEFT, padx=10)
        self.time_label = tk.Label(self.metrics_frame, text="Runtime: 0s")
        self.time_label.pack(side=tk.LEFT, padx=10)
        self.question_window=None

        # Loading spinner
        self.loading_label = tk.Label(self.content, text="Processing...", font=("Arial", 16), fg="blue")
        self.loading_label.pack_forget()  # Hide initially

        # Plot canvas
        self.plot_canvas = None

    def launch_streamlit_app(self):
        # Command to run the Streamlit app
        streamlit_command = ["streamlit", "run", "graph_maker.py"]

        # Start the Streamlit app in a separate process
        subprocess.Popen(streamlit_command)

        # Open the app in the default web browser
        webbrowser.open("http://localhost:8501")

    def generate_data(self):

        alert_window = tk.Toplevel(self.master)
        alert_window.title("Random Data")
        alert_window.geometry("400x100")
        tk.Label(alert_window, text="Generating Data..", font=("Arial", 16)).pack(pady=20)

        #dont stall this thread do it in a seperate thead
        #self.items = generate_random_items(100000, 10)

        def _generate_data():
            self.generate_button.config(state=tk.DISABLED)
            self.show_data_button.config(state=tk.DISABLED)
            self.items = generate_random_items(5000, 10)
            alert_window.withdraw()
            alert_window.destroy()
            try:
                self.show_data("Generated Item Data")
            except Exception as e:
                print(e)
            finally:
                self.generate_button.config(state=tk.NORMAL)
                self.show_data_button.config(state=tk.NORMAL)

        threading.Thread(target=_generate_data).start()
        #call show data after thread is done
        #self.show_data()

    def run_algorithm(self):
        self.timer_start = time.time()
        # Show loading spinner
        self.loading_label.pack()
        self.master.update()

        algorithm_name = self.algo_dropdown.get()
        algorithm = self.algorithms[algorithm_name]
        self.run_button.config(state=tk.DISABLED)
        # Run algorithm in a separate thread
        self.runningalgo=threading.Thread(target=self.execute_algorithm, args=(algorithm,)).start()

    def execute_algorithm(self, algorithm):
        self.viz_area.delete("all")
        max_days = int(self.timeline.cget("to"))
        algorithm_name = self.algo_dropdown.get()
        self.daily_state = simulate_dynamic_packing(self.items, max_days, algorithm, algorithm_name)

        # Update the UI with results
        self.master.after(0, self.on_algorithm_complete)

    def on_algorithm_complete(self):
        self.run_button.config(state=tk.NORMAL)
        self.update_visualization(self.timeline.get())
        self.loading_label.pack_forget()  # Hide loading spinner
        self.timer_end = time.time()
        runtime = self.timer_end - self.timer_start
        max_arrival_time = max(item.arrival_time  for item in self.items)
        max_departure_time = max(item.departure_time  for item in self.items)
        self.timeline.config(to=max(max_arrival_time, max_departure_time))
        self.current_runtime=runtime
        self.update_metrics(self.daily_state[self.timeline.get()]["bins"], runtime)

    def update_visualization(self, day):
        day = int(day)
        if day in self.daily_state:
            state = self.daily_state[day]
            self.draw_bins(state["bins"])
            self.update_metrics(state["bins"],self.current_runtime)  # No runtime for state update

    def draw_bins(self, bins: List[Bin]):
        self.viz_area.delete("all")
        self.viz_area.config(scrollregion=(0, 0, 90*len(bins)+600, 2000))
        bin_width = 60
        bin_height = 200
        spacing = 20

        for i, bin in enumerate(bins):
            x0 = i * (bin_width + spacing) + spacing
            y0 = spacing
            x1 = x0 + bin_width
            y1 = y0 + bin_height

            self.viz_area.create_rectangle(x0, y0, x1, y1, outline="black", fill="lightgrey", tags=f"bin{bin.bin_id}")
            self.viz_area.create_text((x0 + x1) / 2, y1 + 10, text=f"Bin {bin.bin_id}", anchor=tk.N,
                                      tags=f"bin{bin.bin_id}")

            current_y = y1

            total_filled=sum(item.size for item in bin.items)
            percentage_filled=total_filled/bin.capacity
            fill_height=percentage_filled*bin_height
            self.viz_area.create_rectangle(x0, y1-fill_height, x1, y1, outline="black", fill="skyblue", tags=f"bin{bin.bin_id}")
            self.viz_area.create_text((x0 + x1) / 2, y1 - (bin_height//2), text=f"{percentage_filled:.2%}", anchor=tk.N,
                                      tags=f"bin{bin.bin_id}")
            self.viz_area.create_text((x0 + x1) / 2, y1 - (bin_height//2)-20, text=f"({len(bin.items)} items)", anchor=tk.N,
                                        tags=f"bin{bin.bin_id}")
            self.viz_area.tag_bind(f"bin{bin.bin_id}", "<Enter>", lambda e, i=bin.bin_id: self.show_bin_items(e, i))


    def show_item_info(self, item: Item):
        self.item_info_labels[0].config(text=f"Item ID: {item.id}")
        self.item_info_labels[1].config(text=f"Size: {item.size}")
        self.item_info_labels[2].config(text=f"Arrival: {item.arrival_time}")
        self.item_info_labels[3].config(text=f"Departure: {item.departure_time}")

    def show_bin_items(self, event,bin_id):
        # bin_id = int(self.viz_area.gettags(event.widget.find_closest(event.x, event.y))[0][3:])
        if bin_id is None or self.timeline.get() not in self.daily_state:
            return
        bin = next((b for b in self.daily_state[self.timeline.get()]["bins"] if b.bin_id == bin_id), None)
        max_bin_items_show=3
        if bin:
            #clear all labels
            for label in self.item_info_labels:
                label.config(text="")
            items_text = [
                f"Bin ID: {bin.bin_id}",
                f"Capacity: {int(bin.capacity)}",
                f"Remaining Capacity: {bin.remaining_capacity:.2f}",
                f"Items: {len([item.id for item in bin.items])}"
            ]
            #sort by top 10 items by size
            sorted_bin_items=sorted(bin.items,key=lambda x:x.size,reverse=True)
            if len(sorted_bin_items) > 0:
                header = "{:^10} | {:^10} | {:^10} | {:^10}".format("Item ID", "Size", "Arrival", "Departure")
                items_text.append(header)
                items_text.append("-" * len(header))

            for item in sorted_bin_items[:max_bin_items_show]:
                    item_info = "{:^10} | {:^10} | {:^10} | {:^10}".format(item.id, item.size, item.arrival_time,
                                                                        item.departure_time)
                    items_text.append(item_info)


            if len(sorted_bin_items)>max_bin_items_show:
                items_text.append("...show more")



            for label, text in zip(self.item_info_labels, items_text):
                if text=="...show more":
                    label.config(text=text,fg="blue")
                    label.bind("<Button-1>", lambda e: self.show_all_items_in_bin(bin_id))
                else:
                    label.config(text=text)

    def update_metrics(self, bins: List[Bin], runtime: float):
        current_bins = len(bins)
        all_bins=[bin for day in self.daily_state for bin in self.daily_state[day]["bins"]]
        total_bins=len(all_bins)
        if current_bins == 0:
            avg_utilization=0
        else:
            avg_utilization = sum((1.0 - bin.remaining_capacity) for bin in bins) / current_bins if current_bins > 0 else 0
        self.bins_label.config(text=f"Bins: {current_bins}/{total_bins}")
        self.util_label.config(text=f"Avg Utilization: {avg_utilization:.2%}")
        self.avg_items_per_bin_label.config(text=f"Avg Items per Bin: {round(avg_items_per_bin(bins) if current_bins > 0 else 0, 2)}")
        total_items= sum(len(bin.items) for bin in all_bins)
        avg_total_utilization=sum((1.0 - bin.remaining_capacity) for bin in all_bins) / total_bins if total_bins > 0 else 0
        self.Efficiency_label.config(text=f"Efficiency: {round(avg_total_utilization*departed_items_percentage(all_bins) / ((total_bins / total_items)*(1000000*runtime/(total_items*math.log(total_items,2))+0.00001)), 2)}")
        self.time_label.config(text=f"Runtime: {runtime:.2f}s")

    def upload_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        alert_window = tk.Toplevel(self.master)
        alert_window.title("Alert")
        alert_window.geometry("400x100")
        tk.Label(alert_window, text="Uploading Data..", font=("Arial", 16)).pack(pady=20)

        def _upload_data():
            items = []
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    try:
                        item_id, size, arrival_time, departure_time = map(float, row)
                        items.append(Item(int(item_id), size, int(arrival_time), int(departure_time)))
                    except ValueError:
                        continue

            self.items = items

            alert_window.withdraw()
            alert_window.destroy()
            try:
                self.show_data("Uploaded Item Data")
            except Exception as e:
                print(e)

        threading.Thread(target=_upload_data).start()

    def download_data(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])

        if not file_path:

            return

        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for item in self.items:
                writer.writerow([item.id, item.size, item.arrival_time, item.departure_time])

    def run_comparison(self,skip_deep_comparison=False):
        self.compare_button.config(state=tk.DISABLED)
        #show a question box to ask if user wants to do a deep comparison warn that it will take a long time
        #if yes do deep comparison
        #else do normal comparison
        if not skip_deep_comparison:
            self.question_window = tk.Toplevel(self.master)
            self.question_window.title("Deep Comparison")
            self.question_window.geometry("900x100")
            tk.Label(self.question_window, text="Do you want to perform a deep comparison?(Warning! this takes long time)", font=("Arial", 16)).pack(pady=20, padx=20)
            self.yes_button = tk.Button(self.question_window, text="Yes", command=self.deep_comparison)

            self.no_button = tk.Button(self.question_window, text="No", command=lambda: self.run_comparison(skip_deep_comparison=True))
            self.yes_button.pack(side=tk.LEFT, padx=10)
            self.no_button.pack(side=tk.RIGHT, padx=10)
            #destroy the question window after 5 seconds
            # self.question_window.after(10000, self.question_window.destroy)
            self.compare_button.config(state=tk.NORMAL)
            return

        if self.question_window:
            self.question_window.destroy()
        fig, ax = plt.subplots()
        def _run_comparison():
            try:
                self.plot_window = tk.Toplevel(self.master)
                self.plot_window.title("Comparison of Algorithms")
                self.plot_window.geometry("800x600")
                #add a label saying processing
                self.plot_window_label = tk.Label(self.plot_window, text="Processing...", font=("Arial", 16), fg="blue")
                self.plot_window_label.pack()
                results = compare_algorithms(self.items, 10, self.algorithms)
                self.plot_window_label.pack_forget()

                if self.plot_canvas:
                    self.plot_canvas.get_tk_widget().pack_forget()
                for algo_name, result in results.items():
                    ax.plot(range(11), result["bins_used"], label=algo_name)

                ax.set_xlabel("Days")
                ax.set_ylabel("Bins Used")
                ax.set_title("Comparison of Algorithms")
                ax.legend()

                self.plot_canvas = FigureCanvasTkAgg(fig, master=self.plot_window)
                self.plot_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                self.plot_canvas.draw()
            except:
                pass
            finally:
                self.compare_button.config(state=tk.NORMAL)

        #do above in a thread
        threading.Thread(target=_run_comparison).start()
        #_run_comparison()

    def deep_comparison(self):
            if self.question_window:
                self.question_window.destroy()
            algorithms = [
                first_fit, next_fit, best_fit, worst_fit, first_fit_decreasing,
                best_fit_decreasing, next_fit_decreasing, worst_fit_decreasing,
                greedy_size_first, rolling_horizon_dp ]

            test_cases = []
            # Add the new test cases with repeating patterns
            pattern_1 = [0.3, 0.7, 0.9]
            pattern_2 = [0.8, 0.2, 0.3]
            for ns in [10, 50, 100, 500, 1000, 5000, 10000]:
                test_cases.extend([
                    ("Small items",ns,-1, create_items([random.uniform(0.1, 0.5) for _ in range(ns)])),
                    ("Large items",ns,-1,create_items([random.uniform(0.5, 1.0) for _ in range(ns)])),
                    ("Mixed items", ns, -1, create_random_items(ns)),
                    ("Repeating Pattern 1", ns, -1, create_pattern_items(ns, pattern_1)),
                    ("Repeating Pattern 2", ns, -1, create_pattern_items(ns, pattern_2)),
                    ("Varying Bin Capacity", ns, -1, create_items_with_varying_bin_capacity(ns)),
                    ("Highly Fragmented Items", ns, -1, create_fragmented_items(ns)),
                ])
                for d in [10, 50, 100, 500, 1000, 5000, 10000]:
                    test_cases.append(("Time-aware items", ns, d, create_time_aware_items(ns, d)))
                    test_cases.append(("Correlated Arrival and Departure", ns, d, create_correlated_items(ns, d)))

            results = []
            for case_name, nitems, ndays, items in tqdm(test_cases):
                for algorithm in algorithms:
                    start_time = time.time()
                    bins = algorithm(items)
                    end_time = time.time()
                    results.append(
                        {
                            "case": case_name,
                            "algorithm": algorithm.__name__,
                            "bins_used": count_used_bins(bins),
                            "Bin Utilization": avg_bin_utilization(bins),
                            "Avg Items per Bin": avg_items_per_bin(bins),#add avg_days_per_bin
                            "Percent Items Departed": departed_items_percentage(bins),
                            "Bins Used per Item": count_used_bins(bins) / nitems,
                            "Efficiency": avg_bin_utilization(bins)* departed_items_percentage(bins) / (100000*((count_used_bins(bins) / nitems)*((end_time - start_time)/(nitems*math.log(nitems, 2)))+0.00001)),#(item_utilization * bin_utilization) / (normalized_bin_count * normalized_runtime)
                            "time per item": (end_time - start_time)/nitems,
                            "time": (end_time - start_time),
                            "nitems": nitems,
                            "ndays": ndays
                        }
                    )

            # save resuts as csv
            import csv
            filepath = './visualization/results.csv'
            with open(filepath, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["Case", "Algorithm", "Bins Used", "Time / Item (s)", "nitems", "ndays", "Bin Utilization", "Avg Items per Bin", "Percent Items Departed", "Bins Used per Item", "Efficiency","Time (s)"])
                for result in results:
                    writer.writerow(
                        [result['case'], result['algorithm'], result["bins_used"], result["time per item"], result["nitems"],
                         result["ndays"], result["Bin Utilization"], result["Avg Items per Bin"], result["Percent Items Departed"],
                            result["Bins Used per Item"], result["Efficiency"],result["time"]
                         ])

    def show_all_items_in_bin(self, bin_id):
        data_window = tk.Toplevel(self.master)
        data_window.title(f"Items in Bin {bin_id}")
        data_window.geometry("400x400")

        bin = next((b for b in self.daily_state[self.timeline.get()]["bins"] if b.bin_id == bin_id), None)
        if bin:
            for item in bin.items:
                text=f"Item ID: {item.id}, Size: {item.size}, Arrival: {item.arrival_time}, Departure: {item.departure_time}"
                padded_text=f"{text:<100}"
                tk.Label(data_window, text=padded_text).pack()

    def show_data(self,title="Item Data"):
        data_window = tk.Toplevel(self.master)
        data_window.title(title)
        data_window.geometry("400x400")

        text_widget = tk.Text(data_window, wrap="none")
        text_widget.pack(expand=True, fill=tk.BOTH)

        for item in self.items:
            text_widget.insert(tk.END,
                               f"ID: {item.id}, Size: {item.size}, Arrival: {item.arrival_time}, Departure: {item.departure_time}\n")

        for item in self.items:
            print(f"ID: {item.id}, Size: {item.size}, Arrival: {item.arrival_time}, Departure: {item.departure_time}")