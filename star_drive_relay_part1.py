import numpy as np
from astropy import units as u
from astropy import constants as const

class StarDriveRelayBase:
    """
    Base class for Star Drive Relay simulation and optimization.

    This class provides core parameters and methods used in both simulation and 
    optimization, including calculations for magnetic fields, Lorentz forces, 
    plasma interactions, and thermal effects. 
    """

    def __init__(
        self,
        num_rings: int,
        spacecraft_mass: u.Quantity,
        ring_diameters: u.Quantity,
        ring_thicknesses: u.Quantity,
        magnetic_fields: u.Quantity,
        electric_fields: u.Quantity,
        current_densities: u.Quantity,
        em_losses: float = 5.0,
        plasma_losses: float = 10.0,
        g_force_manned: float = 3.0,
        g_force_unmanned: float = 20.0,
        initial_velocity: u.Quantity = 1000 * u.m / u.s,
        pulse_frequency: u.Quantity = 10 * u.Hz,
        pulse_duration: u.Quantity = 0.01 * u.s,
        mhd_coefficient: float = 0.01,
        parameter_ranges: tuple = (500.0, 2000.0),
        spacecraft_mass_max: float = 2000.0,
        ring_diameters_min: float = 5.0,
        ring_diameters_max: float = 30.0,
        **kwargs
    ):
        self.gui = kwargs.pop('gui', None)
        self.num_rings = num_rings
        self.spacecraft_mass = spacecraft_mass
        self.ring_diameters = ring_diameters
        self.ring_thicknesses = ring_thicknesses
        self.ring_radii = self.ring_diameters / 2
        self.magnetic_fields = magnetic_fields
        self.electric_fields = electric_fields
        self.current_densities = current_densities
        self.em_losses = em_losses / 100  # Convert percentage to ratio
        self.plasma_losses = plasma_losses / 100  # Convert percentage to ratio
        self.g_force_manned = g_force_manned
        self.g_force_unmanned = g_force_unmanned
        self.initial_velocity = initial_velocity
        self.g = const.g0  # Gravitational acceleration (m/s²)
        self.pulse_frequency = pulse_frequency
        self.pulse_duration = pulse_duration
        self.mhd_coefficient = mhd_coefficient
        self.zhanikbekov_threshold = 0.1  # Threshold for Zhanikbekov effect
        self.parameter_ranges = parameter_ranges
        self.spacecraft_mass_max = spacecraft_mass_max
        self.ring_diameters_min = ring_diameters_min
        self.ring_diameters_max = ring_diameters_max

        self.calculate_ring_cross_sectional_areas()

    def calculate_ring_cross_sectional_areas(self):
        self.ring_cross_sectional_areas = np.pi * (
            self.ring_radii ** 2
            - (self.ring_radii - self.ring_thicknesses) ** 2
        )

    def calculate_magnetic_field(
        self, ring_index: int, position: np.ndarray
    ) -> u.Quantity:
        """
        Calculate the magnetic field at a given position for a specific ring.

        Args:
            ring_index (int): Index of the ring.
            position (np.ndarray): Position at which to calculate the magnetic field (x, y, z).

        Returns:
            u.Quantity: Magnetic field vector in Teslas.
        """
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

    def calculate_lorentz_force(self, magnetic_field: u.Quantity, velocity: u.Quantity) -> u.Quantity:
        """Calculate the Lorentz force acting on the spacecraft."""
        try:
            magnetic_field = np.atleast_1d(magnetic_field)  # Ensure at least 1D array
            velocity = np.atleast_1d(velocity)           # Ensure at least 1D array
            return const.e.si * np.cross(velocity, magnetic_field, axis=0).squeeze()
        except Exception as e:
            print(f"Error in calculate_lorentz_force: {e}")
            return np.zeros(3) * u.N 

    def calculate_plasma_interaction(
        self, velocity: u.Quantity, magnetic_field: u.Quantity, electric_field: u.Quantity
    ) -> tuple:
        """
        Calculate the plasma interaction forces and effects.

        Args:
            velocity (u.Quantity): Velocity vector of the spacecraft in meters per second.
            magnetic_field (u.Quantity): Magnetic field vector in Teslas.
            electric_field (u.Quantity): Electric field vector in Volts per meter.

        Returns:
            tuple: A tuple containing:
                - F_total (u.Quantity): Total force vector due to plasma interaction in Newtons.
                - debye_length (u.Quantity): Debye length in meters.
        """
        plasma_density = 1e12 * u.m**-3  # Plasma density (particles/m^3)
        plasma_temperature = 1e6 * u.K  # Plasma temperature (K)
        debye_length = np.sqrt(
            const.k_B.si * plasma_temperature / (plasma_density * const.e.si**2)
        )

        if isinstance(electric_field, u.Quantity) and electric_field.ndim == 1:
            electric_field = electric_field[:, np.newaxis]

        if isinstance(magnetic_field, u.Quantity) and magnetic_field.ndim == 1:
            magnetic_field = magnetic_field[:, np.newaxis]

        # Calculate the plasma interaction forces
        F_electric = plasma_density * const.e.si * electric_field
        F_magnetic = plasma_density * const.e.si * np.cross(velocity, magnetic_field, axis=0)
        F_total = F_electric + F_magnetic

        return F_total, debye_length

    def calculate_thermal_effects(
        self, power_dissipation: u.Quantity, temperature: u.Quantity
    ) -> u.Quantity:
        """
        Calculate the thermal effects on the spacecraft.

        Args:
            power_dissipation (u.Quantity): Power dissipation in Watts.
            temperature (u.Quantity): Current temperature of the spacecraft in Kelvin.

        Returns:
            u.Quantity: New temperature of the spacecraft in Kelvin after thermal effects.
        """
        # Assume a simple thermal model with constant specific heat capacity
        specific_heat_capacity = 500 * u.J / (u.kg * u.K)  # Example value
        mass = self.spacecraft_mass.to(u.kg)
        heat_capacity = mass * specific_heat_capacity

        # Calculate the new temperature based on power dissipation and heat capacity
        delta_temperature = (power_dissipation * self.pulse_duration) / heat_capacity
        new_temperature = temperature + delta_temperature

        return new_temperature