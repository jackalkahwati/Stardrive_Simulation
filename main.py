import tkinter as tk
from tkinter import ttk
import numpy as np
from star_drive_relay_part1 import StarDriveRelay
from star_drive_relay_part2 import StarDriveRelaySimulation
from star_drive_relay_part3 import StarDriveRelayOptimization

class StarDriveRelayGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Star Drive Relay Simulation and Optimization")
        self.geometry("800x600")

        # Create input frames
        input_frame = ttk.Frame(self)
        input_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.create_input_section(input_frame)
        self.create_range_section(input_frame)
        self.create_optimization_criteria_section(input_frame)

        # Create button frame
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=10, padx=10)

        simulate_button = ttk.Button(button_frame, text="Run Simulation", command=self.run_simulation)
        simulate_button.pack(side=tk.LEFT, padx=10)

        optimize_button = ttk.Button(button_frame, text="Run Optimization", command=self.run_optimization)
        optimize_button.pack(side=tk.LEFT, padx=10)

        # Create output area
        output_frame = ttk.Frame(self)
        output_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.output_text = tk.Text(output_frame, wrap=tk.WORD, width=80, height=20)
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # Create algorithm selection
        algorithm_frame = ttk.Frame(self)
        algorithm_frame.pack(pady=10, padx=10)

        algorithm_label = ttk.Label(algorithm_frame, text="Select Optimization Algorithm:")
        algorithm_label.pack(side=tk.LEFT)

        self.algorithm_var = tk.StringVar()
        self.algorithm_var.set("Genetic Algorithm")  # Set the default algorithm
        algorithm_dropdown = ttk.Combobox(algorithm_frame, textvariable=self.algorithm_var, values=["Genetic Algorithm", "Simulated Annealing", "Particle Swarm Optimization"])
        algorithm_dropdown.pack(side=tk.LEFT, padx=10)

    def create_input_section(self, parent):
        input_section = ttk.LabelFrame(parent, text="Input Parameters", padding=10)
        input_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.spacecraft_mass_entry = self.create_entry(input_section, "Spacecraft Mass (kg):", 1000.0, 500.0, 2000.0)
        self.ring_diameters_entries = self.create_multiple_entries(input_section, "Ring Diameters (m):", [10.0, 20.0, 30.0], 5.0, 50.0)
        self.magnetic_fields_entries = self.create_multiple_entries(input_section, "Magnetic Fields (T):", [20.0, 30.0, 40.0], 10.0, 50.0)
        self.electric_fields_entries = self.create_multiple_entries(input_section, "Electric Fields (V/m):", [1000.0, 2000.0, 3000.0], 500.0, 10000.0)
        self.current_densities_entries = self.create_multiple_entries(input_section, "Current Densities (A/m^2):", [1e6, 2e6, 3e6], 1e5, 1e8)
        self.em_losses_entry = self.create_entry(input_section, "EM Losses (%):", 5.0, 1.0, 10.0)
        self.plasma_losses_entry = self.create_entry(input_section, "Plasma Losses (%):", 10.0, 5.0, 20.0)
        self.g_force_manned_entry = self.create_entry(input_section, "G-Force (Manned):", 3.0, 1.0, 5.0)
        self.g_force_unmanned_entry = self.create_entry(input_section, "G-Force (Unmanned):", 6.0, 2.0, 10.0)
        self.initial_velocity_entry = self.create_entry(input_section, "Initial Velocity (m/s):", 1000.0, 500.0, 1500.0)
        self.desired_delta_v_entry = self.create_entry(input_section, "Desired Delta-V (m/s):", 100.0, 100.0, 10000.0)
        self.pulse_frequency_entry = self.create_entry(input_section, "Pulse Frequency (Hz):", 10.0, 5.0, 20.0)
        self.pulse_duration_entry = self.create_entry(input_section, "Pulse Duration (s):", 0.1, 0.05, 0.2)
        self.mhd_coefficient_entry = self.create_entry(input_section, "MHD Coefficient:", 0.01, 0.005, 0.05)
        self.efficiency_entry = self.create_entry(input_section, "Efficiency (%):", 80.0, 70.0, 95.0)

    def create_range_section(self, parent):
        range_section = ttk.LabelFrame(parent, text="Parameter Ranges", padding=10)
        range_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.spacecraft_mass_min_entry = self.create_entry(range_section, "Spacecraft Mass Min:", 500.0)
        self.spacecraft_mass_max_entry = self.create_entry(range_section, "Spacecraft Mass Max:", 2000.0)
        self.em_losses_min_entry = self.create_entry(range_section, "EM Losses Min:", 1.0, 0.0, 100.0)
        self.em_losses_max_entry = self.create_entry(range_section, "EM Losses Max:", 10.0, 0.0, 100.0)
        self.plasma_losses_min_entry = self.create_entry(range_section, "Plasma Losses Min:", 5.0, 0.0, 100.0)
        self.plasma_losses_max_entry = self.create_entry(range_section, "Plasma Losses Max:", 20.0, 0.0, 100.0)
        self.g_force_manned_min_entry = self.create_entry(range_section, "G-Force (Manned) Min:", 1.0)
        self.g_force_manned_max_entry = self.create_entry(range_section, "G-Force (Manned) Max:", 5.0)
        self.g_force_unmanned_min_entry = self.create_entry(range_section, "G-Force (Unmanned) Min:", 2.0)
        self.g_force_unmanned_max_entry = self.create_entry(range_section, "G-Force (Unmanned) Max:", 10.0)
        self.initial_velocity_min_entry = self.create_entry(range_section, "Initial Velocity Min:", 500.0)
        self.initial_velocity_max_entry = self.create_entry(range_section, "Initial Velocity Max:", 1500.0)
        self.pulse_frequency_min_entry = self.create_entry(range_section, "Pulse Frequency Min:", 5.0)
        self.pulse_frequency_max_entry = self.create_entry(range_section, "Pulse Frequency Max:", 20.0)
        self.pulse_duration_min_entry = self.create_entry(range_section, "Pulse Duration Min:", 0.05)
        self.pulse_duration_max_entry = self.create_entry(range_section, "Pulse Duration Max:", 0.2)
        self.mhd_coefficient_min_entry = self.create_entry(range_section, "MHD Coefficient Min:", 0.005)
        self.mhd_coefficient_max_entry = self.create_entry(range_section, "MHD Coefficient Max:", 0.05)
        self.efficiency_min_entry = self.create_entry(range_section, "Efficiency Min:", 70.0, 0.0, 100.0)
        self.efficiency_max_entry = self.create_entry(range_section, "Efficiency Max:", 95.0, 0.0, 100.0)

    def create_optimization_criteria_section(self, parent):
        criteria_section = ttk.LabelFrame(parent, text="Optimization Criteria", padding=10)
        criteria_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.delta_v_weight_entry = self.create_entry(criteria_section, "Delta-V Weight:", 1.0)
        self.num_passes_weight_entry = self.create_entry(criteria_section, "Num Passes Weight:", 1.0)
        self.power_requirement_weight_entry = self.create_entry(criteria_section, "Power Requirement Weight:", 1.0)
        self.thermal_load_weight_entry = self.create_entry(criteria_section, "Thermal Load Weight:", 1.0)
        self.efficiency_weight_entry = self.create_entry(criteria_section, "Efficiency Weight:", 1.0)
        self.stability_weight_entry = self.create_entry(criteria_section, "Stability Weight:", 1.0)

        self.max_iterations_entry = self.create_entry(criteria_section, "Max Iterations:", 100)

    def create_entry(self, parent, label_text, default_value, min_value=None, max_value=None):
        entry_frame = ttk.Frame(parent)
        entry_frame.pack(pady=5, fill=tk.X)

        label = ttk.Label(entry_frame, text=label_text, width=25, anchor=tk.W)
        label.pack(side=tk.LEFT)

        entry = ttk.Entry(entry_frame, width=10)
        entry.insert(0, str(default_value))
        if min_value is not None and max_value is not None:
            entry.config(validate="key", validatecommand=(self.register(self.validate_float), "%P", min_value, max_value))
        entry.pack(side=tk.LEFT, padx=5)

        return entry

    def create_multiple_entries(self, parent, label_text, default_values, min_value, max_value):
        entry_frame = ttk.Frame(parent)
        entry_frame.pack(pady=5, fill=tk.X)

        label = ttk.Label(entry_frame, text=label_text, width=25, anchor=tk.W)
        label.pack(side=tk.LEFT)

        entries = []
        for default_value in default_values:
            entry = ttk.Entry(entry_frame, width=10)
            entry.insert(0, str(default_value))
            entry.config(validate="key", validatecommand=(self.register(self.validate_float), "%P", min_value, max_value))
            entry.pack(side=tk.LEFT, padx=5)
            entries.append(entry)

        return entries

    def validate_float(self, value, min_value, max_value):
        if value == "":
            return True
        try:
            float_value = float(value)
            if min_value <= float_value <= max_value:
                return True
            else:
                return False
        except ValueError:
            return False

    def run_simulation(self):
        self.clear_output()

        # Get input values
        spacecraft_mass = float(self.spacecraft_mass_entry.get())
        ring_diameters = np.array([float(entry.get()) for entry in self.ring_diameters_entries])
        magnetic_fields = np.array([float(entry.get()) for entry in self.magnetic_fields_entries])
        electric_fields = np.array([float(entry.get()) for entry in self.electric_fields_entries])
        current_densities = np.array([float(entry.get()) for entry in self.current_densities_entries])
        em_losses = float(self.em_losses_entry.get())
        plasma_losses = float(self.plasma_losses_entry.get())
        g_force_manned = float(self.g_force_manned_entry.get())
        g_force_unmanned = float(self.g_force_unmanned_entry.get())
        initial_velocity = float(self.initial_velocity_entry.get())
        desired_delta_v = float(self.desired_delta_v_entry.get())
        pulse_frequency = float(self.pulse_frequency_entry.get())
        pulse_duration = float(self.pulse_duration_entry.get())
        mhd_coefficient = float(self.mhd_coefficient_entry.get())
        efficiency = float(self.efficiency_entry.get())

        # Create instances of the required classes
        star_drive_relay = StarDriveRelay(
            spacecraft_mass,
            ring_diameters,
            magnetic_fields,
            electric_fields,
            current_densities,
            em_losses,
            plasma_losses,
            g_force_manned,
            g_force_unmanned,
            initial_velocity,
            pulse_frequency,
            pulse_duration,
            mhd_coefficient,
        )

        simulation = StarDriveRelaySimulation(
            spacecraft_mass,
            ring_diameters,
            magnetic_fields,
            electric_fields,
            current_densities,
            em_losses,
            plasma_losses,
            g_force_manned,
            g_force_unmanned,
            initial_velocity,
            pulse_frequency,
            pulse_duration,
            mhd_coefficient,
        )

        # Run the simulation
        results = simulation.simulate(
            desired_delta_v,
            efficiency,
            spacecraft_mass,
            ring_diameters,
            magnetic_fields,
            electric_fields,
            current_densities,
            em_losses,
            plasma_losses,
            g_force_manned,
            g_force_unmanned,
            initial_velocity,
            pulse_frequency,
            pulse_duration,
            mhd_coefficient,
        )

        # Display the simulation results
        self.output_text.insert(tk.END, "Simulation Results:\n")
        for key, value in results.items():
            self.output_text.insert(tk.END, f"{key}: {value}\n")

    def run_optimization(self):
        self.clear_output()

        # Get input values
        spacecraft_mass = float(self.spacecraft_mass_entry.get())
        ring_diameters = np.array([float(entry.get()) for entry in self.ring_diameters_entries])
        magnetic_fields = np.array([float(entry.get()) for entry in self.magnetic_fields_entries])
        electric_fields = np.array([float(entry.get()) for entry in self.electric_fields_entries])
        current_densities = np.array([float(entry.get()) for entry in self.current_densities_entries])
        em_losses = float(self.em_losses_entry.get())
        plasma_losses = float(self.plasma_losses_entry.get())
        g_force_manned = float(self.g_force_manned_entry.get())
        g_force_unmanned = float(self.g_force_unmanned_entry.get())
        initial_velocity = float(self.initial_velocity_entry.get())
        desired_delta_v = float(self.desired_delta_v_entry.get())
        pulse_frequency = float(self.pulse_frequency_entry.get())
        pulse_duration = float(self.pulse_duration_entry.get())
        mhd_coefficient = float(self.mhd_coefficient_entry.get())
        efficiency = float(self.efficiency_entry.get())

        # Get optimization criteria weights
        delta_v_weight = float(self.delta_v_weight_entry.get())
        num_passes_weight = float(self.num_passes_weight_entry.get())
        power_requirement_weight = float(self.power_requirement_weight_entry.get())
        thermal_load_weight = float(self.thermal_load_weight_entry.get())
        efficiency_weight = float(self.efficiency_weight_entry.get())
        stability_weight = float(self.stability_weight_entry.get())

        # Get parameter ranges
        spacecraft_mass_min = float(self.spacecraft_mass_min_entry.get())
        spacecraft_mass_max = float(self.spacecraft_mass_max_entry.get())
        em_losses_min = float(self.em_losses_min_entry.get())
        em_losses_max = float(self.em_losses_max_entry.get())
        plasma_losses_min = float(self.plasma_losses_min_entry.get())
        plasma_losses_max = float(self.plasma_losses_max_entry.get())
        g_force_manned_min = float(self.g_force_manned_min_entry.get())
        g_force_manned_max = float(self.g_force_manned_max_entry.get())
        g_force_unmanned_min = float(self.g_force_unmanned_min_entry.get())
        g_force_unmanned_max = float(self.g_force_unmanned_max_entry.get())
        initial_velocity_min = float(self.initial_velocity_min_entry.get())
        initial_velocity_max = float(self.initial_velocity_max_entry.get())
        pulse_frequency_min = float(self.pulse_frequency_min_entry.get())
        pulse_frequency_max = float(self.pulse_frequency_max_entry.get())
        pulse_duration_min = float(self.pulse_duration_min_entry.get())
        pulse_duration_max = float(self.pulse_duration_max_entry.get())
        mhd_coefficient_min = float(self.mhd_coefficient_min_entry.get())
        mhd_coefficient_max = float(self.mhd_coefficient_max_entry.get())
        efficiency_min = float(self.efficiency_min_entry.get())
        efficiency_max = float(self.efficiency_max_entry.get())

        max_iterations = int(self.max_iterations_entry.get())

        # Create instances of the required classes
        optimization = StarDriveRelayOptimization(
            spacecraft_mass,
            ring_diameters,
            magnetic_fields,
            electric_fields,
            current_densities,
            em_losses,
            plasma_losses,
            g_force_manned,
            g_force_unmanned,
            initial_velocity,
            pulse_frequency,
            pulse_duration,
            mhd_coefficient,
        )

        # Define optimization criteria
        optimization_criteria = {
            "delta_v": delta_v_weight,
            "passes_manned": num_passes_weight,
            "passes_unmanned": num_passes_weight,
            "power_requirements_manned": power_requirement_weight,
            "power_requirements_unmanned": power_requirement_weight,
            "thermal_management_manned": thermal_load_weight,
            "thermal_management_unmanned": thermal_load_weight,
            "efficiency": efficiency_weight,
            "stability_manned": stability_weight,
            "stability_unmanned": stability_weight,
        }

        # Define parameter ranges
        parameter_ranges = [
            (spacecraft_mass_min, spacecraft_mass_max),
            (5, 50),  # Ring diameters range (m)
            (0.5, 10),  # Magnetic fields range (T)
            (500, 10000),  # Electric fields range (V/m)
            (1e5, 1e8),  # Current densities range (A/m^2)
            (em_losses_min, em_losses_max),
            (plasma_losses_min, plasma_losses_max),
            (g_force_manned_min, g_force_manned_max),
            (g_force_unmanned_min, g_force_unmanned_max),
            (initial_velocity_min, initial_velocity_max),
            (pulse_frequency_min, pulse_frequency_max),
            (pulse_duration_min, pulse_duration_max),
            (mhd_coefficient_min, mhd_coefficient_max),
        ]

        # Get the selected optimization algorithm
        algorithm_name = self.algorithm_var.get()

        # Perform optimization
        best_solution, best_score = optimization.select_algorithm(
            algorithm_name,
            desired_delta_v,
            efficiency,
            optimization_criteria,
            parameter_ranges,
            max_iterations,
        )

        # Display the optimization results
        self.output_text.insert(tk.END, "Optimization Results:\n")
        self.output_text.insert(tk.END, f"Best Solution: {best_solution}\n")
        self.output_text.insert(tk.END, f"Best Score: {best_score}\n")

    def clear_output(self):
        self.output_text.delete("1.0", tk.END)

if __name__ == "__main__":
    app = StarDriveRelayGUI()
    app.mainloop()