import logging
import tkinter as tk
from StarDriveRelayGUI import StarDriveRelayGUI

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting the GUI application")

    # Create the root window
    root = tk.Tk()

    # Start the Tkinter GUI
    app = StarDriveRelayGUI(root)  # Instantiate your custom GUI class, passing the root window as the master
    root.mainloop()  # This starts the main event loop for the GUI

if __name__ == "__main__":
    main()