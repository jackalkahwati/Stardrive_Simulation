import numpy as np
from astropy import units as u
from astropy import constants as const
from astropy.coordinates.matrix_utilities import rotation_matrix

class StarDriveRelay:
    def __init__(
        self,
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
    ):
        self.spacecraft_mass = spacecraft_mass * u.kg
        self.ring_diameters = ring_diameters * u.m
        self.magnetic_fields = magnetic_fields * u.T
        self.electric_fields = electric_fields * u.V / u.m
        self.current_densities = current_densities * u.A / u.m**2
        self.em_losses = em_losses / 100  # Convert percentage to ratio
        self.plasma_losses = plasma_losses / 100  # Convert percentage to ratio
        self.g_force_manned = g_force_manned
        self.g_force_unmanned = g_force_unmanned
        self.initial_velocity = initial_velocity * u.m / u.s
        self.g = const.g0  # Gravitational acceleration (m/s²)
        self.pulse_frequency = pulse_frequency * u.Hz
        self.pulse_duration = pulse_duration * u.s
        self.mhd_coefficient = mhd_coefficient
        self.zhanikbekov_threshold = 0.1  # Threshold for Zhanikbekov effect

    def calculate_magnetic_field(self, ring_index, position):
        ring_diameter = self.ring_diameters[ring_index]
        current_density = self.current_densities[ring_index]

        r = np.sqrt(position[0] ** 2 + position[1] ** 2)
        z = position[2]

        if r == 0 and z == 0:
            return np.zeros(3) * u.T  # Handle the case when r and z are both zero

        B_r = (
            const.mu0
            * current_density
            * ring_diameter**2
            * z
            / (2 * (r**2 + z**2) ** (3 / 2))
        )
        B_z = (
            const.mu0
            * current_density
            * ring_diameter**2
            / (2 * (r**2 + z**2) ** (1 / 2))
        )

        return np.array([B_r, 0, B_z]) * u.T  # Return a 3D array for the magnetic field

    def calculate_lorentz_force(self, magnetic_field, velocity):
        if isinstance(velocity, (int, float)):
            velocity = np.array([0, 0, velocity]) * u.m / u.s  # Convert scalar velocity to 3D array
        return const.e * np.cross(velocity, magnetic_field)

    def calculate_plasma_interaction(self, velocity, magnetic_field, electric_field):
        plasma_density = 1e12 * u.m**-3  # Plasma density (particles/m^3)
        plasma_temperature = 1e6 * u.K  # Plasma temperature (K)
        debye_length = np.sqrt(
            const.k_B * plasma_temperature / (plasma_density * const.e**2)
        )

        if isinstance(electric_field, (int, float)):
            electric_field = np.array([0, 0, electric_field]) * u.V / u.m  # Convert scalar electric field to 3D array

        if isinstance(velocity, (int, float)):
            velocity = np.array([0, 0, velocity]) * u.m / u.s  # Convert scalar velocity to 3D array

        E_eff = electric_field + np.cross(velocity, magnetic_field)
        F_plasma = const.e * plasma_density * E_eff

        collision_frequency = 1e8 * u.Hz  # Collision frequency (Hz)
        conductivity = (
            plasma_density
            * const.e**2
            / (const.m_e * collision_frequency)
        )
        resistivity = 1 / conductivity

        F_resistive = resistivity * plasma_density * velocity

        F_total = F_plasma + F_resistive

        return F_total, debye_length

    def calculate_thermal_effects(self, power_dissipation, temperature):
        emissivity = 0.8  # Emissivity of the spacecraft
        surface_area = 100 * u.m**2  # Surface area of the spacecraft (m^2)

        # Radiative heat transfer
        Q_rad = (
            emissivity
            * const.sigma_sb
            * surface_area
            * (temperature**4 - (300 * u.K)**4)
        )

        # Conductive heat transfer (assuming a simplified model)
        thermal_conductivity = (
            50 * u.W / (u.m * u.K)  # Thermal conductivity of the spacecraft material (W/m/K)
        )
        thickness = 0.05 * u.m  # Thickness of the spacecraft wall (m)
        delta_T = (
            temperature - 300 * u.K
        )  # Temperature difference between the spacecraft and the environment (K)
        Q_cond = thermal_conductivity * surface_area * delta_T / thickness

        # Total heat transfer
        Q_total = Q_rad + Q_cond + power_dissipation

        # Update spacecraft temperature (assuming a lumped capacitance model)
        heat_capacity = 1000 * u.J / u.K  # Heat capacity of the spacecraft (J/K)
        time_step = 1 / self.pulse_frequency  # Time step based on pulse frequency
        temperature_change = Q_total * time_step / heat_capacity
        new_temperature = temperature + temperature_change

        return new_temperature

    def calculate_lorentz_force_acceleration(self, ring_index, time):
        if self.is_pulsing(time):
            return (
                self.current_densities[ring_index]
                * self.magnetic_fields[ring_index]
                * self.ring_diameters[ring_index]
            )
        else:
            return 0 * u.m / u.s**2

    def is_pulsing(self, time):
        return (time % (1 / self.pulse_frequency)) < self.pulse_duration

    def calculate_electric_force_acceleration(self, ring_index):
        return (self.electric_fields[ring_index] * const.e) / const.m_e

    def calculate_total_acceleration(self, ring_index, time):
        a_L = self.calculate_lorentz_force_acceleration(ring_index, time)
        a_E = self.calculate_electric_force_acceleration(ring_index)
        a_total = a_L + a_E
        effective_acceleration = a_total * (1 - (self.em_losses + self.plasma_losses))
        return effective_acceleration

    def apply_g_force_constraints(self, acceleration, manned):
        if manned:
            return min(acceleration, self.g_force_manned * self.g)
        else:
            return min(acceleration, self.g_force_unmanned * self.g)

    def calculate_velocity(self, initial_velocity, acceleration, distance):
        return np.sqrt(initial_velocity**2 + 2 * acceleration * distance)

    def calculate_mhd_acceleration(self, velocity):
        return -self.mhd_coefficient * velocity**2

    def zhanikbekov_effect(self, orientation):
        if np.linalg.norm(orientation) > self.zhanikbekov_threshold:
            # Apply a random rotation to simulate the Zhanikbekov effect
            rotation_matrix = self.generate_random_rotation_matrix()
            orientation = np.dot(rotation_matrix, orientation)
        return orientation

    def generate_random_rotation_matrix(self):
        # Generate a random rotation matrix for the Zhanikbekov effect
        random_rotation = rotation_matrix(np.random.uniform(0, 2*np.pi), axis='z')
        return random_rotation.matrix