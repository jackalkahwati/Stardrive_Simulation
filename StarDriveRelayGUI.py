import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from astropy import units as u
from scipy.optimize import minimize
import logging
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import json

from StarDriveRelayBase import StarDriveRelayBase, StarDriveRelaySimulation

class StarDriveRelayGUI:
    def __init__(self, master):
      super().__init__()
      self.master = master
      self.title("Star Drive Relay Simulation")
      self.geometry("800x1000")
      self.setup_logging()

      # Define the input_formats dictionary
      self.input_formats = {
          "Spacecraft Mass (kg):": "single",
          "Ring Diameters (m):": 3,
          "Ring Thickness (m):": 3,
          "Magnetic Fields (T):": 3,
          "Electric Fields (V/m):": "range",
          "Current Densities (A/m^2):": "range",
          "EM Losses (%)": "single",
          "Plasma Losses (%)": "single",
          "G-Force (Manned)": "single",
          "G-Force (Unmanned)": "single",
          "Initial Velocity (m/s)": "single",
          "Desired Delta-V (m/s)": "single",
          "Parameter Ranges": "range",
          "Pulse Frequency (Hz)": "single",
          "Pulse Duration (s)": "single",
          "Efficiency (%)": "single"
      }

      # Initialize the InputValidator instance
      self.input_validator = InputValidator(self, self.input_formats)

        # Setup input fields
        self.input_fields = {}
        for row, (label_text, default_value) in enumerate(input_data):
            label = ttk.Label(self.master, text=label_text)
            label.grid(row=row, column=0, sticky="e", padx=5, pady=5)

            entry = ttk.Entry(self.master)
            entry.insert(0, default_value)
            entry.grid(row=row, column=1, padx=5, pady=5)

            self.input_fields[label_text] = entry

        # Output area
        self.output_text = tk.Text(self.master, height=6, width=40)
        self.output_text.grid(row=len(input_data), column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        # Setup buttons
        self.setup_buttons(len(input_data))

        # Create plot area
        self.setup_plot_area(len(input_data))

    def setup_buttons(self, row_offset):
        self.run_simulation_button = ttk.Button(self.master, text="Run Simulation", command=self.run_simulation)
        self.run_simulation_button.grid(row=row_offset + 1, column=0, padx=5, pady=5)

        self.run_optimization_button = ttk.Button(self.master, text="Run Optimization", command=self.run_optimization)
        self.run_optimization_button.grid(row=row_offset + 1, column=1, padx=5, pady=5)

        self.analyze_button = ttk.Button(self.master, text="Analyze", command=self.analyze)
        self.analyze_button.grid(row=row_offset + 2, column=0, padx=5, pady=5)

        self.run_parameter_impact_analysis_button = ttk.Button(self.master, text="Run Parameter Impact Analysis", command=self.run_parameter_impact_analysis)
        self.run_parameter_impact_analysis_button.grid(row=row_offset + 2, column=1, padx=5, pady=5)

        self.export_results_button = ttk.Button(self.master, text="Export Results", command=self.export_results)
        self.export_results_button.grid(row=row_offset + 3, column=0, columnspan=2, padx=5, pady=5)

    def setup_plot_area(self, row_offset):
        self.plot_area = tk.Frame(self.master)
        self.plot_area.grid(row=row_offset + 4, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

    def run_simulation(self):
        try:
            # Collect input values from GUI fields
            inputs = {key: self.input_fields[key].get() for key in self.input_fields}
            # Convert inputs and run the simulation
            simulation_results = self.perform_simulation(inputs)
            # Display the results in the GUI
            self.display_results(simulation_results)
        except astropy.units.UnitConversionError as e:
            messagebox.showerror("Unit Error", f"Unit conversion error in simulation: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

    def perform_simulation(self, inputs):
        # Extract and convert input values, initialize the simulation object and run the simulation
        # Return a dictionary containing the simulation results
        return {"Placeholder": "Data"}

    def display_results(self, results):
        # Update the output text area with results from the simulation
        pass

    def clear_plot(self):
        # Clear previous plots if any
        pass

    def plot_data(self, x_data, y_data, title, xlabel, ylabel):
        # Plot the given data on the plot area
        pass

    def run_optimization(self):
        # Get input values from entry fields
        parameter_ranges = list(map(float, self.input_fields["Parameter Ranges"].get().split(',')))
        desired_delta_v_value = float(self.input_fields["Desired Delta-V (m/s)"].get() or "0.0")
        desired_delta_v = desired_delta_v_value * u.m / u.s  # Convert to u.Quantity with velocity units
        logging.debug(f"Desired Delta-V: {desired_delta_v}")

        efficiency = float(self.input_fields["Efficiency (%)"].get() or "0.0")
        max_iterations = int(self.input_fields["Maximum Iterations"].get() or "0")

        # Define optimization function
        def optimization_function(parameters):
            # Unpack parameters
            spacecraft_mass, ring_diameter, ring_thickness, magnetic_field, electric_field, current_density = parameters

            # Create simulation instance
            simulation = StarDriveRelaySimulation(
                num_rings=1,
                spacecraft_mass=spacecraft_mass,
                ring_diameters=[ring_diameter],
                ring_thicknesses=[ring_thickness],
                magnetic_fields=[magnetic_field],
                electric_fields=[electric_field],
                current_densities=[current_density],
                em_losses=0,
                plasma_losses=0,
                g_force_manned=0,
                g_force_unmanned=0,
                initial_velocity=0,
                pulse_frequency=0,
                pulse_duration=0
            )

            # Run simulation
            simulation_results = simulation.simulate(desired_delta_v, efficiency / 100, max_iterations=max_iterations)

            # Return the negative of the final velocity (to be maximized)
            if simulation_results["velocity"]:
                logging.debug(f"Simulation results for desired_delta_v={desired_delta_v}: {simulation_results['velocity'][-1]}")
                return -simulation_results["velocity"][-1]
            else:
                return 0.0

        # Run optimization
        initial_guess = [1000, 10, 0.1, 20, 1000, 10000]
        bounds = [
            (500, 2000),  # spacecraft_mass
            (5, 20),      # ring_diameter
            (0.05, 0.2),  # ring_thickness
            (10, 50),     # magnetic_field
            (500, 2000),  # electric_field
            (5000, 20000) # current_density
        ]
        result = minimize(optimization_function, x0=initial_guess, bounds=bounds, method='Nelder-Mead')

        # Update output text area
        output_text = "Optimization Results:\n"
        output_text += f"Optimized Spacecraft Mass: {result.x[0]:.2f} kg\n"
        output_text += f"Optimized Ring Diameter: {result.x[1]:.2f} m\n"
        output_text += f"Optimized Ring Thickness: {result.x[2]:.2f} m\n"
        output_text += f"Optimized Magnetic Field: {result.x[3]:.2f} T\n"
        output_text += f"Optimized Electric Field: {result.x[4]:.2f} V/m\n"
        output_text += f"Optimized Current Density: {result.x[5]:.2f} A/m^2\n"
        self.update_output(output_text)

    def analyze(self):
        # Get input values from entry fields
        spacecraft_mass = float(self.input_fields["Spacecraft Mass (kg)"].get() or "0.0")
        ring_diameters = list(map(float, self.input_fields["Ring Diameters (m)"].get().split(',')))
        ring_thicknesses = list(map(float, self.input_fields["Ring Thickness (m)"].get().split(',')))
        magnetic_fields = list(map(float, self.input_fields["Magnetic Fields (T)"].get().split(',')))
        electric_fields = list(map(float, self.input_fields["Electric Fields (V/m)"].get().split(',')))
        current_densities = list(map(float, self.input_fields["Current Densities (A/m^2)"].get().split(',')))
        em_losses = float(self.input_fields["EM Losses (%)"].get() or "0.0")
        plasma_losses = float(self.input_fields["Plasma Losses (%)"].get() or "0.0")
        g_force_manned = float(self.input_fields["G-Force (Manned)"].get() or "0.0")
        g_force_unmanned = float(self.input_fields["G-Force (Unmanned)"].get() or "0.0")
        initial_velocity = float(self.input_fields["Initial Velocity (m/s)"].get() or "0.0")
        desired_delta_v_value = float(self.input_fields["Desired Delta-V (m/s)"].get() or "0.0")
        desired_delta_v = desired_delta_v_value * u.m / u.s  # Convert to u.Quantity with units
        logging.debug(f"Desired Delta-V: {desired_delta_v}")

        pulse_frequency = float(self.input_fields["Pulse Frequency (Hz)"].get() or "0.0")
        pulse_duration = float(self.input_fields["Pulse Duration (s)"].get() or "0.0")
        efficiency = float(self.input_fields["Efficiency (%)"].get() or "0.0")
        max_iterations = int(self.input_fields["Maximum Iterations"].get() or "0")

        # Create simulation instance
        simulation = StarDriveRelaySimulation(
            num_rings=len(ring_diameters),
            spacecraft_mass=spacecraft_mass,
            ring_diameters=ring_diameters,
            ring_thicknesses=ring_thicknesses,
            magnetic_fields=magnetic_fields,
            electric_fields=electric_fields,
            current_densities=current_densities,
            em_losses=em_losses,
            plasma_losses=plasma_losses,
            g_force_manned=g_force_manned,
            g_force_unmanned=g_force_unmanned,
            initial_velocity=initial_velocity,
            pulse_frequency=pulse_frequency,
            pulse_duration=pulse_duration
        )

        # Run simulation
        simulation_results = simulation.simulate(desired_delta_v, efficiency / 100, max_iterations=max_iterations)
        logging.debug(f"Simulation results for desired_delta_v={desired_delta_v}: {simulation_results['velocity'][-1]}")

        # Perform analysis
        analysis_results = simulation.analyze(efficiency / 100)

        # Update output text area
        output_text = "Analysis Results:\n"
        for key, value in analysis_results.items():
            output_text += f"{key}: {value}\n"
        self.update_output(output_text)

    def run_parameter_impact_analysis(self):
        try:
            # Gather inputs with error checking
            num_rings = 1  # Example value, adjust as needed
            spacecraft_mass = self.input_validator.validate_input(self.spacecraft_mass_entry, "single")
            spacecraft_mass = self.input_validator.convert_to_units(self.spacecraft_mass_entry, spacecraft_mass, u.kg)
            ring_diameters = self.input_validator.validate_input(self.ring_diameters_entry, "list")
            ring_diameters = np.asarray(ring_diameters) * u.m  # Convert to quantity with units
            ring_thicknesses = self.input_validator.validate_input(self.ring_thickness_entry, "list")
            ring_thicknesses = np.asarray(ring_thicknesses) * u.m  # Convert to quantity with units
            magnetic_fields = self.input_validator.validate_input(self.magnetic_fields_entry, "list")
            magnetic_fields = self.input_validator.convert_to_units(self.magnetic_fields_entry, magnetic_fields, u.T)
            electric_fields = self.input_validator.validate_input(self.electric_fields_entry, "range")
            electric_fields = self.input_validator.convert_to_units(self.electric_fields_entry, electric_fields, u.V / u.m)
            current_densities = self.input_validator.validate_input(self.current_densities_entry, "range")
            current_densities = self.input_validator.convert_to_units(self.current_densities_entry, current_densities, u.A / u.m**2)
            em_losses = self.input_validator.validate_input(self.em_losses_entry, "single")
            plasma_losses = self.input_validator.validate_input(self.plasma_losses_entry, "single")
            g_force_manned = self.input_validator.validate_input(self.g_force_manned_entry, "single")
            g_force_unmanned = self.input_validator.validate_input(self.g_force_unmanned_entry, "single")
            initial_velocity = self.input_validator.validate_input(self.initial_velocity_entry, "single") * u.m / u.s
            pulse_frequency = self.input_validator.validate_input(self.pulse_frequency_entry, "single") * u.Hz
            pulse_duration = self.input_validator.validate_input(self.pulse_duration_entry, "single") * u.s
            efficiency = self.input_validator.validate_input(self.efficiency_entry, "single")  # Retrieve efficiency from the entry field
            parameter_ranges = self.input_validator.validate_input(self.parameter_ranges_entry, "range")
            parameter_ranges = self.input_validator.convert_to_units(self.parameter_ranges_entry, parameter_ranges, None)
            desired_delta_v = self.input_validator.validate_input(self.desired_delta_v_entry, "single") * u.m / u.s
    
            # Define parameter values for analysis
            spacecraft_mass_values = np.linspace(parameter_ranges[0], parameter_ranges[1], 5)
            ring_diameter_values = np.linspace(5, 20, 5)
            ring_thickness_values = np.linspace(0.05, 0.2, 5)
            magnetic_field_values = np.linspace(10, 50, 5)
            electric_field_values = np.linspace(500, 2000, 5)
            current_density_values = np.linspace(5000, 20000, 5)
    
            # Initialize result arrays
            spacecraft_mass_results = []
            ring_diameter_results = []
            ring_thickness_results = []
            magnetic_field_results = []
            electric_field_results = []
            current_density_results = []
    
            # Perform parameter impact analysis
            for spacecraft_mass in spacecraft_mass_values:
                simulation = StarDriveRelaySimulation(
                    num_rings=1,
                    spacecraft_mass=spacecraft_mass,
                    ring_diameters=[10],
                    ring_thicknesses=[0.1],
                    magnetic_fields=[20],
                    electric_fields=[1000],
                    current_densities=[10000],
                    em_losses=0,
                    plasma_losses=0,
                    g_force_manned=0,
                    g_force_unmanned=0,
                    initial_velocity=0,
                    pulse_frequency=0,
                    pulse_duration=0
                )
                simulation_results = simulation.simulate(desired_delta_v, efficiency / 100, max_iterations=max_iterations)
                if simulation_results["velocity"]:
                    spacecraft_mass_results.append(simulation_results["velocity"][-1])
                else:
                    spacecraft_mass_results.append(0.0 * u.m / u.s)
    
            for ring_diameter in ring_diameter_values:
                simulation = StarDriveRelaySimulation(
                    num_rings=1,
                    spacecraft_mass=1000,
                    ring_diameters=[ring_diameter],
                    ring_thicknesses=[0.1],
                    magnetic_fields=[20],
                    electric_fields=[1000],
                    current_densities=[10000],
                    em_losses=0,
                    plasma_losses=0,
                    g_force_manned=0,
                    g_force_unmanned=0,
                    initial_velocity=0,
                    pulse_frequency=0,
                    pulse_duration=0
                )
                simulation_results = simulation.simulate(desired_delta_v, efficiency / 100, max_iterations=max_iterations)
                if simulation_results["velocity"]:
                    ring_diameter_results.append(simulation_results["velocity"][-1])
                else:
                    ring_diameter_results.append(0.0 * u.m / u.s)
    
            for ring_thickness in ring_thickness_values:
                simulation = StarDriveRelaySimulation(
                    num_rings=1,
                    spacecraft_mass=1000,
                    ring_diameters=[10],
                    ring_thicknesses=[ring_thickness],
                    magnetic_fields=[20],
                    electric_fields=[1000],
                    current_densities=[10000],
                    em_losses=0,
                    plasma_losses=0,
                    g_force_manned=0,
                    g_force_unmanned=0,
                    initial_velocity=0,
                    pulse_frequency=0,
                    pulse_duration=0
                )
                simulation_results = simulation.simulate(desired_delta_v, efficiency / 100, max_iterations=max_iterations)
                if simulation_results["velocity"]:
                    ring_thickness_results.append(simulation_results["velocity"][-1])
                else:
                    ring_thickness_results.append(0.0 * u.m / u.s)
    
            # Perform similar analysis for other parameters...
    
            # Clear previous plots
            self.clear_plot()
    
            # Plot results
            self.plot_data(spacecraft_mass_values, spacecraft_mass_results, "Spacecraft Mass Impact", "Spacecraft Mass (kg)", "Final Velocity (m/s)")
            self.plot_data(ring_diameter_values, ring_diameter_results, "Ring Diameter Impact", "Ring Diameter (m)", "Final Velocity (m/s)")
            self.plot_data(ring_thickness_values, ring_thickness_results, "Ring Thickness Impact", "Ring Thickness (m)", "Final Velocity (m/s)")
            # Plot results for other parameters...
        except ValueError as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, str(e))
        
    def export_results(self):
        # Get input values from entry fields
        spacecraft_mass = float(self.input_fields["Spacecraft Mass (kg)"].get() or "0.0")
        ring_diameters = list(map(float, self.input_fields["Ring Diameters (m)"].get().split(',')))
        ring_thicknesses = list(map(float, self.input_fields["Ring Thickness (m)"].get().split(',')))
        magnetic_fields = list(map(float, self.input_fields["Magnetic Fields (T)"].get().split(',')))
        electric_fields = list(map(float, self.input_fields["Electric Fields (V/m)"].get().split(',')))
        current_densities = list(map(float, self.input_fields["Current Densities (A/m^2)"].get().split(',')))
        em_losses = float(self.input_fields["EM Losses (%)"].get() or "0.0")
        plasma_losses = float(self.input_fields["Plasma Losses (%)"].get() or "0.0")
        g_force_manned = float(self.input_fields["G-Force (Manned)"].get() or "0.0")
        g_force_unmanned = float(self.input_fields["G-Force (Unmanned)"].get() or "0.0")
        initial_velocity = float(self.input_fields["Initial Velocity (m/s)"].get() or "0.0")
        desired_delta_v_value = float(self.input_fields["Desired Delta-V (m/s)"].get() or "0.0")
        desired_delta_v = desired_delta_v_value * u.m / u.s  # Convert to u.Quantity with units
        logging.debug(f"Desired Delta-V: {desired_delta_v}")
    
        pulse_frequency = float(self.input_fields["Pulse Frequency (Hz)"].get() or "0.0")
        pulse_duration = float(self.input_fields["Pulse Duration (s)"].get() or "0.0")
        efficiency = float(self.input_fields["Efficiency (%)"].get() or "0.0")
        max_iterations = int(self.input_fields["Maximum Iterations"].get() or "0")
    
        # Create simulation instance
        simulation = StarDriveRelaySimulation(
            num_rings=len(ring_diameters),
            spacecraft_mass=spacecraft_mass,
            ring_diameters=ring_diameters,
            ring_thicknesses=ring_thicknesses,
            magnetic_fields=magnetic_fields,
            electric_fields=electric_fields,
            current_densities=current_densities,
            em_losses=em_losses,
            plasma_losses=plasma_losses,
            g_force_manned=g_force_manned,
            g_force_unmanned=g_force_unmanned,
            initial_velocity=initial_velocity,
            pulse_frequency=pulse_frequency,
            pulse_duration=pulse_duration
        )
    
        # Run simulation
        simulation_results = simulation.simulate(desired_delta_v, efficiency / 100, max_iterations=max_iterations)
    
        # Save results to file
        file_path = "simulation_results.json"
        with open(file_path, "w") as file:
            json.dump(simulation_results, file)
    
        # Update output text area
        output_text = f"Simulation results exported to {file_path}\n"
        self.update_output(output_text)
    
    def update_output(self, text):
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, text)
    
    def clear_plot(self):
        for widget in self.plot_area.winfo_children():
            widget.destroy()
    
    def plot_data(self, x_data, y_data, title, x_label, y_label):
        figure = plt.Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.plot(x_data, [value.value for value in y_data])
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    
        canvas = FigureCanvasTkAgg(figure, self.plot_area)
        canvas.draw()
        canvas.get_tk_widget().pack()
if __name__ == "__main__":
  root = tk.Tk()
  app = StarDriveRelayGUI(root)
  root.mainloop()