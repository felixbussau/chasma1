# -*- coding: utf-8 -*-
"""
Created on 05.02.2025
 
@author: i_valais
"""

import numpy as np
# import matplotlib.pyplot as plt

import Work.scripts.implementations.material as mat
import Work.scripts.implementations.lattice_struct as lat
import Work.scripts.implementations.sheet_implement as sheet


class RevolutionGeom:
    """
    Class to define the spherical or cylindrical with spherical ends tank.

    Args:
        name (str): Name of the tank.
        tank_radius (float or int): Radius of the outer side of the tank (in mm).
        tot_length (float or int): Total length of the tank between the two outer sides of the
                tank wall (in mm).
        sheets (list): List containing the Sheet objects that consist the tank (from outermost
                to innermost).

    TODO:
        - plot the tank (can wait).
        - check if there is a lattice as first or last layer and raise an error. (partly fixed)
        - prevent the stacking of two lattice layers in the row. See class Sheet.
    """

    def __init__(self, name=None, tank_radius=None, tot_length=None, sheets=None):
        """
        Initialisation of the class RevolutionGeom.
        """
        self.name = name
        self.tank_radius = tank_radius # radius of the outer surface of the tank
        self.tot_length = tot_length
        self.sheets = sheets # list of objects gotten by other class (Sheet)

        if 2 * self.tank_radius > self.tot_length:
            raise ValueError("The diameter of the tank cannot be larger than the total length.")

        if self.sheets[0].lattice is not None or self.sheets[-1].lattice is not None:
            raise ValueError("The innermost or outermost layers cannot be lattices.")  # not a ValueError (check the error types)

        self.tot_thickness = self.calc_thickness()

    def calc_thickness(self):
        """
        Function to calculate the total thickness of the tank.
        """
        total_thickness = 0
        for key in self.sheets:
            total_thickness += key.thickness

        return total_thickness

    def calc_vol(self):
        """
        Function to calculate the volume of the inner side of the tank. The possible
        fillets are not considered (at the moment).

        Returns:
            tank_vol (float): Total inner volume of the tank (in mm^3).
        """
        vol_sphere = 4 / 3 * np.pi * (self.tank_radius - self.tot_thickness) ** 3
        vol_cyl = (self.tot_length - 2 * self.tank_radius) * np.pi * (self.tank_radius - self.tot_thickness) ** 2

        tank_vol = vol_sphere + vol_cyl

        return tank_vol

    def calc_surf(self):
        """
        Function to calculate from the outer to the inner surface area of the specified position.
        The possible fillets are not considered (at the moment).

        Returns:
            surf_area (list): List containing the area of each surface of the tank (from
                    outermost to innermost) (in mm^2).
        """
        surf_area = list(np.zeros(len(self.sheets) + 1))
        sub_thickness = 0

        for i, key in enumerate(self.sheets):
            surf_sphere = 4 * np.pi * (self.tank_radius - sub_thickness) ** 2
            surf_cyl = 2 * np.pi * (self.tank_radius - sub_thickness) * (self.tot_length - 2 * self.tank_radius)

            surf_area[i] = surf_cyl + surf_sphere

            sub_thickness += key.thickness

        surf_sphere = 4 * np.pi * (self.tank_radius - self.tot_thickness) ** 2
        surf_cyl = 2 * np.pi * (self.tank_radius - self.tot_thickness) * (self.tot_length - 2 * self.tank_radius)

        surf_area[-1] = surf_cyl + surf_sphere

        return surf_area

    def surf_to_vol(self):
        """
        Function to calculate the surface area to volume ratio of the tank.

        Returns:
            sa_v_ratio (float): Ratio of the outer surface area to inner volume (in mm^-1).
        """
        surf_area = self.calc_surf()[0]
        vol = self.calc_vol()

        sa_v_ratio = surf_area / vol

        return sa_v_ratio

    def calc_mass(self):
        """
        Function to calculate the mass of the tank.

        Returns:
            tank_mass (float): Mass of the defined tank (in ton).
        """
        tank_mass = 0
        radius_out = self.tank_radius
        radius_in = self.tank_radius
        for i, key in enumerate(self.sheets):
            radius_in -= key.thickness

            vol_sphere_out = 4 / 3 * np.pi * radius_out ** 3
            vol_cyl_out = (self.tot_length - 2 * self.tank_radius) * np.pi * radius_out ** 2

            vol_out = vol_sphere_out + vol_cyl_out

            vol_sphere_in = 4 / 3 * np.pi * radius_in ** 3
            vol_cyl_in = (self.tot_length - 2 * self.tank_radius) * np.pi * radius_in ** 2

            vol_in = vol_sphere_in + vol_cyl_in

            vol_sheet = vol_out - vol_in
            if key.lattice is None:
                tank_mass += vol_sheet * key.material.rho
            else:
                tank_mass += vol_sheet * key.lattice.calc_unit_cell_density() * key.lattice.material.rho

            radius_out -= key.thickness

        return tank_mass


    def tank_plot(self):
        """
        Function to plot the tank's outer and inner geometry.
        """
#         # Outer Side
#         x1 = np.linspace(0, self.h_left)
#         y1_up = np.sqrt(self.left_curvature_radius ** 2 - (x1 - self.left_curvature_radius) ** 2)
#         y1_down = -y1_up
#
#         x3 = np.linspace(self.tot_length - self.h_right, self.tot_length)
#         y3_up = np.sqrt(self.right_curvature_radius ** 2 - (x3 - self.tot_length + self.right_curvature_radius) ** 2)
#         y3_down = -y3_up
#
#         x = np.append(x1, x3)
#         y_up = np.append(y1_up, y3_up)
#         y_down = np.append(y1_down, y3_down)
#
#         # Inner Side
#         inner_x1 = np.linspace(self.thickness, self.h_left)
#         inner_y1_up = np.sqrt(
#             (self.left_curvature_radius - self.thickness) ** 2 - (inner_x1 - self.left_curvature_radius) ** 2)
#         inner_y1_down = -inner_y1_up
#
#         inner_x3 = np.linspace(self.tot_length - self.h_right, self.tot_length - self.thickness)
#         inner_y3_up = np.sqrt((self.right_curvature_radius - self.thickness) ** 2 - (
#                 inner_x3 - self.tot_length + self.right_curvature_radius) ** 2)
#         inner_y3_down = -inner_y3_up
#
#         inner_x = np.append(inner_x1, inner_x3)
#         inner_y_up = np.append(inner_y1_up, inner_y3_up)
#         inner_y_down = np.append(inner_y1_down, inner_y3_down)
#
#         # Plotting
#         fig, ax = plt.subplots()
#         plt.title('Tank Geometry')
#         plt.plot(x, y_up, 'b')
#         plt.plot(x, y_down, 'b')
#         plt.plot(inner_x, inner_y_up, 'b', linestyle='--')
#         plt.plot(inner_x, inner_y_down, 'b', linestyle='--')
#         plt.xlabel("x (mm)")
#         plt.ylabel("y (mm)")
#         ax.set_aspect('equal')
#         plt.grid()
#         plt.show()

if __name__ == '__main__':
    isolation_mat = mat.Material(name='aero', rho=5.4e-12, e_mod=204000, nu=0.3, s_yield=380, s_ult=690, e_break=0.5,
                        therm_cond=0.04)
    p511 = mat.Material(name='p511', rho=7.8e-9, e_mod=204000, nu=0.3, s_yield=380, s_ult=690, e_break=0.5,
                        therm_cond=15)

    bcc = lat.UnitCell(name='bcc unit cell', uni_type='bcc', material=p511, l_long_cell=15, l_circ_cell=15, h_cell=7.9, d_rod=0.5)

    iso_layer = sheet.Sheet(name='iso_layer', material=isolation_mat, thickness=40, lattice=None)
    metal_layer = sheet.Sheet(name='metal_layer', material=p511, thickness=0.87, lattice=None)
    lat_layer = sheet.Sheet(name='metal_layer', material=p511, thickness=7.9, lattice=bcc)
    metal_layer_in = sheet.Sheet(name='metal_layer', material=p511, thickness=2, lattice=None)

    tank_sheets = [metal_layer, lat_layer, metal_layer, lat_layer, metal_layer, lat_layer, metal_layer]

    tank = RevolutionGeom(name='tank', tank_radius=500, tot_length=1182, sheets=tank_sheets)

    print("Total thickness of the tank: " + str(tank.tot_thickness) + " mm.")
    print("Inner volume of tank: " + str(tank.calc_vol() / 1e9) + " m^3.")
    print("Outer surface area of tank: " + str(tank.calc_surf()[0] / 1e6) + " m^2.")
    print("Inner surface area of tank: " + str(tank.calc_surf()[-1] / 1e6) + " m^2.")
    print("Ratio of outer surface to inner volume area: " + str(tank.surf_to_vol() * 1e3) + " m^-1.")
    print("Mass of the entire tank: " + str(tank.calc_mass() * 1e3) + " kg.")

