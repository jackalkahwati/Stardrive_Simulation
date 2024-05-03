import numpy as np
from astropy import units as u
import logging

logging.basicConfig(level=logging.DEBUG)

class StarDriveRelayBase:
    def __init__(self, num_rings, spacecraft_mass, ring_diameters, ring_thicknesses,
                 magnetic_fields, electric_fields, current_densities, em_losses,
                 plasma_losses, g_force_manned, g_force_unmanned, initial_velocity,
                 pulse_frequency, pulse_duration):
        self.num_rings = num_rings
        self.spacecraft_mass = spacecraft_mass * u.kg
        self.ring_diameters = np.array(ring_diameters) * u.m
        self.ring_thicknesses = np.array(ring_thicknesses) * u.m
        self.magnetic_fields = np.array(magnetic_fields) * u.T
        self.electric_fields = np.array(electric_fields) * (u.V / u.m)
        self.current_densities = np.array(current_densities) * (u.A / u.m**2)
        self.em_losses = em_losses
        self.plasma_losses = plasma_losses
        self.g_force_manned = g_force_manned * (u.m / u.s**2)
        self.g_force_unmanned = g_force_unmanned * (u.m / u.s**2)
        self.initial_velocity = initial_velocity * (u.m / u.s)
        self.pulse_frequency = pulse_frequency * u.Hz
        self.pulse_duration = pulse_duration * u.s

    def calculate_force(self, magnetic_field, electric_field, current_density, ring_diameter, ring_thickness):
        try:
            force = magnetic_field * electric_field * current_density * ring_diameter * ring_thickness
            return force
        except (AttributeError, u.UnitsError) as e:
            logging.error(f"Error occurred while calculating force: {str(e)}")
            raise

    def calculate_acceleration(self, force, spacecraft_mass):
        try:
            acceleration = force / spacecraft_mass
            return acceleration
        except (AttributeError, u.UnitsError) as e:
            logging.error(f"Error occurred while calculating acceleration: {str(e)}")
            raise

    def calculate_velocity(self, initial_velocity, acceleration, time_step):
        try:
            velocity = initial_velocity + acceleration * time_step
            return velocity
        except (AttributeError, u.UnitsError) as e:
            logging.error(f"Error occurred while calculating velocity: {str(e)}")
            raise

    def calculate_power_consumption(self, force, velocity):
        try:
            power_consumption = force * velocity
            return power_consumption
        except (AttributeError, u.UnitsError) as e:
            logging.error(f"Error occurred while calculating power consumption: {str(e)}")
            raise

    def calculate_temperature(self, power_consumption, em_losses, plasma_losses):
        try:
            temperature = power_consumption * (em_losses + plasma_losses)
            return temperature
        except (AttributeError, u.UnitsError) as e:
            logging.error(f"Error occurred while calculating temperature: {str(e)}")
            raise

class StarDriveRelaySimulation(StarDriveRelayBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.velocity_history = []  # Initialize the velocity_history attribute
        self.acceleration_history = []
        self.power_consumption_history = []
        self.temperature_history = []

    def simulate(self, desired_delta_v_m_per_s, efficiency, time_step=1.0, max_iterations=1000):
        time = 0 * u.s
        iteration = 0
        velocity = self.initial_velocity.to(u.m / u.s)  # Ensure velocity is correctly initialized in m/s
        time_step = time_step * u.s  # Define time_step in seconds
        desired_delta_v = desired_delta_v_m_per_s * u.m / u.s  # Explicitly define desired_delta_v in m/s

        logging.debug(f"Initial velocity: {velocity}, with unit: {velocity.unit}")
        logging.debug(f"Desired Delta V: {desired_delta_v}, with unit: {desired_delta_v.unit}")

        while iteration < max_iterations:
            try:
                velocity_m_per_s = velocity.to(u.m / u.s)
                desired_delta_v_m_per_s = desired_delta_v.to(u.m / u.s)
            except u.UnitConversionError as e:
                logging.error(f"Failed to convert units: {str(e)}")
                break

            if velocity_m_per_s >= desired_delta_v_m_per_s:
                break

            total_force = 0 * u.N
            for i in range(self.num_rings):
                try:
                    force = self.calculate_force(
                        self.magnetic_fields[i].to(u.T),
                        self.electric_fields[i].to(u.V / u.m),
                        self.current_densities[i].to(u.A / u.m**2),
                        self.ring_diameters[i].to(u.m),
                        self.ring_thicknesses[i].to(u.m)
                    )
                    total_force += force
                except (AttributeError, u.UnitsError) as e:
                    logging.error(f"Error occurred while calculating force for ring {i}: {str(e)}")
                    raise

            try:
                acceleration = self.calculate_acceleration(total_force, self.spacecraft_mass)
                velocity = self.calculate_velocity(velocity, acceleration, time_step)
                logging.debug(f"Velocity after iteration {iteration}: {velocity}, with unit: {velocity.unit}")
                self.velocity_history.append(velocity.value)  # Append velocity value to the history
                self.acceleration_history.append(acceleration.value)  # Append acceleration value to the history
            except (AttributeError, u.UnitsError) as e:
                logging.error(f"Error occurred while calculating acceleration or updating velocity: {str(e)}")
                raise

            try:
                power_consumption = self.calculate_power_consumption(total_force, velocity)
                temperature = self.calculate_temperature(power_consumption, self.em_losses, self.plasma_losses)
                self.power_consumption_history.append(power_consumption.value)  # Append power consumption value to the history
                self.temperature_history.append(temperature.value)  # Append temperature value to the history
            except (AttributeError, u.UnitsError) as e:
                logging.error(f"Error occurred while calculating power consumption or temperature: {str(e)}")
                raise

            time += time_step
            iteration += 1

        simulation_results = {
            "final_velocity": velocity,
            "iterations": iteration,
            "velocity": self.velocity_history,  # Add the velocity history to the results
            "acceleration": self.acceleration_history,  # Add the acceleration history to the results
            "power_consumption": self.power_consumption_history,  # Add the power consumption history to the results
            "temperature": self.temperature_history  # Add the temperature history to the results
        }

        return simulation_results

    def analyze(self, efficiency):
        velocity_history = np.array(self.velocity_history) * (u.m / u.s)  # Convert velocity history to quantity with units
        acceleration_history = np.array(self.acceleration_history) * (u.m / u.s**2)  # Convert acceleration history to quantity with units
        power_consumption_history = np.array(self.power_consumption_history) * u.W  # Convert power consumption history to quantity with units
        temperature_history = np.array(self.temperature_history) * u.K  # Convert temperature history to quantity with units

        if len(velocity_history) == 0:
            return {}  # Return an empty dictionary if there are no elements in the arrays

        max_velocity = np.max(velocity_history)
        max_acceleration = np.max(acceleration_history)
        total_power_consumption = np.sum(power_consumption_history)
        max_temperature = np.max(temperature_history)

        try:
            specific_impulse = (max_velocity / (9.81 * u.m / u.s**2)).decompose()
            thrust = max_acceleration * self.spacecraft_mass
            power_efficiency = ((max_velocity ** 2) / (2 * total_power_consumption)).decompose()
            energy_efficiency = ((0.5 * self.spacecraft_mass * max_velocity ** 2) / total_power_consumption).decompose()
            heat_generation_rate = total_power_consumption * (1 - efficiency)
        except (AttributeError, u.UnitsError) as e:
            logging.error(f"Error occurred during analysis calculations: {str(e)}")
            raise

        analysis_results = {
            "Maximum Velocity": f"{max_velocity:.2f}",
            "Maximum Acceleration": f"{max_acceleration:.2f}",
            "Total Power Consumption": f"{total_power_consumption:.2f}",
            "Maximum Temperature": f"{max_temperature:.2f}",
            "Specific Impulse": f"{specific_impulse:.2f}",
            "Thrust": f"{thrust:.2f}",
            "Power Efficiency": f"{power_efficiency:.2f}",
            "Energy Efficiency": f"{energy_efficiency:.2f}",
            "Heat Generation Rate": f"{heat_generation_rate:.2f}"
        }

        return analysis_results