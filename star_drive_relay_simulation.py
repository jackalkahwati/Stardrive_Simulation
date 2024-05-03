import numpy as np
from astropy import units as u
from stardrive_utils import calculate_lorentz_force, calculate_plasma_interaction, calculate_thermal_effects

class StarDriveRelaySimulation:
    def __init__(self, params):
        self.params = params
        self.preprocess_params()

    def preprocess_params(self):
        """Convert any necessary input params, e.g., strings to arrays or other unit transformations."""
        self.params['ring_diameters'] = np.array(self.params['ring_diameters']) * u.m
        self.params['initial_velocity'] = self.params['initial_velocity'] * u.m / u.s
        self.params['spacecraft_mass'] = self.params['spacecraft_mass'] * u.kg
        self.params['time_step'] = self.params['time_step'] * u.s
        self.params['magnetic_fields'] = np.array(self.params['magnetic_fields']) * u.T
        self.params['electric_fields'] = np.array(self.params['electric_fields']) * u.V / u.m
        self.params['current_densities'] = np.array(self.params['current_densities']) * u.A / u.m**2
        self.params['ring_thicknesses'] = np.array(self.params['ring_thicknesses']) * u.m

    def simulate(self, desired_delta_v, max_iterations=1000):
        """Run the simulation."""
        # Ensure desired_delta_v has the correct units (velocity, m/s)
        desired_delta_v = desired_delta_v.to(u.m / u.s)

        velocity = self.params['initial_velocity']
        time_history = []
        velocity_history = []
        acceleration_history = []
        power_consumption_history = []
        temperature_history = []

        current_velocity = velocity
        current_acceleration = 0 * u.m / u.s**2
        current_power_consumption = 0 * u.W
        current_temperature = 300 * u.K  # Assuming an initial temperature of 300 K

        for iteration in range(max_iterations):
            total_force = 0 * u.N

            for ring_index in range(len(self.params['ring_diameters'])):
                ring_diameter = self.params['ring_diameters'][ring_index]
                ring_thickness = self.params['ring_thicknesses'][ring_index]
                magnetic_field = self.params['magnetic_fields'][ring_index]
                electric_field = self.params['electric_fields'][ring_index]
                current_density = self.params['current_densities'][ring_index]

                ring_force = calculate_lorentz_force(magnetic_field, current_velocity)
                plasma_force, debye_length = calculate_plasma_interaction(current_velocity, magnetic_field, electric_field)
                ring_force += plasma_force

                total_force += ring_force

            current_acceleration = total_force / self.params['spacecraft_mass']
            current_velocity += current_acceleration * self.params['time_step']

            power_dissipation = current_acceleration * total_force
            current_power_consumption = power_dissipation.to(u.W)

            current_temperature = calculate_thermal_effects(current_power_consumption, current_temperature)

            time_history.append((iteration + 1) * self.params['time_step'].value)
            velocity_history.append(current_velocity.value)
            acceleration_history.append(current_acceleration.value)
            power_consumption_history.append(current_power_consumption.value)
            temperature_history.append(current_temperature.value)

            # Break the loop if the desired_delta_v is reached
            if current_velocity >= desired_delta_v:
                break

        simulation_results = {
            'time': time_history * u.s,
            'velocity': velocity_history * u.m / u.s,
            'acceleration': acceleration_history * u.m / u.s**2,
            'power_consumption': power_consumption_history * u.W,
            'temperature': temperature_history * u.K
        }

        return simulation_results

    def analyze_simulation(self, simulation_results):
        """Analyze results post-simulation."""
        results = {}

        # Check if the velocity list is not empty
        if len(simulation_results['velocity']) > 0:
            # Calculate efficiency
            kinetic_energy = 0.5 * self.params['spacecraft_mass'] * simulation_results['velocity'][-1] ** 2
            total_energy_consumed = np.sum(simulation_results['power_consumption'] * self.params['time_step'])
            efficiency = kinetic_energy / total_energy_consumed
            results['efficiency'] = efficiency

            # Calculate maximum values
            results['max_velocity'] = np.max(simulation_results['velocity'])
            results['max_acceleration'] = np.max(simulation_results['acceleration'])
            results['max_power_consumption'] = np.max(simulation_results['power_consumption'])
            results['max_temperature'] = np.max(simulation_results['temperature'])
        else:
            # If the velocity list is empty, set default or error values
            results['efficiency'] = 0.0
            results['max_velocity'] = 0.0
            results['max_acceleration'] = 0.0
            results['max_power_consumption'] = 0.0
            results['max_temperature'] = 0.0

        return results