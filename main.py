import tkinter as tk
from gui.frontend import BinPackingGUI

# Main loop to run the application
if __name__ == "__main__":
    root = tk.Tk()
    gui = BinPackingGUI(root)
    #gui.deep_comparison()
    root.mainloop()
