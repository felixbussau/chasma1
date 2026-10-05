# -*- coding: utf-8 -*-
"""
Created on 10.02.2025
 
@author: i_valais
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import Work.scripts.implementations.material as mat
import Work.scripts.implementations.strut_implementation as st



class UnitCell:
    """
    Class to define the geometry of the unit cell of the lattice structure. The definition
    of the unit cell type is done by defining the start and end coordinates of every strut,
    composing the struts and then creating a list of the struts to be used. The first coordinate
    is the one in longitudinal direction, the second in circumferential direction and the third
    in the radial direction.

    Args:
        name (str): Name of the unit cell.
        uni_type (str): Type of the unit cell.
        l_long_cell (float): Length of the base of the unit cell in the longitudinal direction of the tank.
        l_circ_cell (float): Length of the base of the unit cell in the circumferential direction of the tank.
        h_cell (float): Height of the unit cell in the radial direction of the tank.
        d_rod (float): Diameter of the rods of the unit cell.
    """

    def __init__(self, name, uni_type, material, d_rod, l_long_cell, l_circ_cell, h_cell=None):
        """
        Initialisation of the class UnitCell.
        """
        self.name = name
        self.uni_type = uni_type
        self.material = material
        self.l_long_cell = l_long_cell
        self.l_circ_cell = l_circ_cell
        self.h_cell = h_cell
        self.d_rod = d_rod

        if self.uni_type == 'bcc':
            self.build_bcc()

            self.c_factor = 0.908

        elif self.uni_type == 'octet':
            p_0_0_0 = np.array((0, 0, 0))
            p_0_0_1 = np.array((0, 0, 1))
            p_0_1_0 = np.array((0, 1, 0))
            p_0_1_1 = np.array((0, 1, 1))
            p_1_0_0 = np.array((1, 0, 0))
            p_1_0_1 = np.array((1, 0, 1))
            p_1_1_0 = np.array((1, 1, 0))
            p_1_1_1 = np.array((1, 1, 1))
            p_05_0_05 = np.array((0.5, 0, 0.5))
            p_05_1_05 = np.array((0.5, 1, 0.5))
            p_0_05_05 = np.array((0, 0.5, 0.5))
            p_1_05_05 = np.array((1, 0.5, 0.5))
            p_05_05_0 = np.array((0.5, 0.5, 0))
            p_05_05_1 = np.array((0.5, 0.5, 1))

            strut1_start = p_0_0_0
            strut1_end = p_1_0_1
            strut1 = np.array((strut1_start, strut1_end))
            strut2_start = p_1_0_0
            strut2_end = p_0_0_1
            strut2 = np.array((strut2_start, strut2_end))
            strut3_start = p_1_0_0
            strut3_end = p_1_1_1
            strut3 = np.array((strut3_start, strut3_end))
            strut4_start = p_1_1_0
            strut4_end = p_1_0_1
            strut4 = np.array((strut4_start, strut4_end))
            strut5_start = p_0_1_0
            strut5_end = p_1_1_1
            strut5 = np.array((strut5_start, strut5_end))
            strut6_start = p_1_1_0
            strut6_end = p_0_1_1
            strut6 = np.array((strut6_start, strut6_end))
            strut7_start = p_0_0_0
            strut7_end = p_0_1_1
            strut7 = np.array((strut7_start, strut7_end))
            strut8_start = p_0_1_0
            strut8_end = p_0_0_1
            strut8 = np.array((strut8_start, strut8_end))
            strut9_start = p_05_05_0
            strut9_end = p_0_05_05
            strut9 = np.array((strut9_start, strut9_end))
            strut10_start = p_05_05_0
            strut10_end = p_1_05_05
            strut10 = np.array((strut10_start, strut10_end))
            strut11_start = p_05_05_0
            strut11_end = p_05_0_05
            strut11 = np.array((strut11_start, strut11_end))
            strut12_start = p_05_05_0
            strut12_end = p_05_1_05
            strut12 = np.array((strut12_start, strut12_end))
            strut13_start = p_05_05_1
            strut13_end = p_0_05_05
            strut13 = np.array((strut13_start, strut13_end))
            strut14_start = p_05_05_1
            strut14_end = p_1_05_05
            strut14 = np.array((strut14_start, strut14_end))
            strut15_start = p_05_05_1
            strut15_end = p_05_0_05
            strut15 = np.array((strut15_start, strut15_end))
            strut16_start = p_05_05_1
            strut16_end = p_05_1_05
            strut16 = np.array((strut16_start, strut16_end))
            strut17_start = p_0_05_05
            strut17_end = p_05_0_05
            strut17 = np.array((strut17_start, strut17_end))
            strut18_start = p_05_0_05
            strut18_end = p_1_05_05
            strut18 = np.array((strut18_start, strut18_end))
            strut19_start = p_1_05_05
            strut19_end = p_05_1_05
            strut19 = np.array((strut19_start, strut19_end))
            strut20_start = p_05_1_05
            strut20_end = p_0_05_05
            strut20 = np.array((strut20_start, strut20_end))

            self.struts = [strut1, strut2, strut3, strut4, strut5, strut6, strut7, strut8, strut9, strut10, strut11,
                           strut12, strut13, strut14, strut15, strut16, strut17, strut18, strut19, strut20]

            self.c_factor = 0.728



        else:
            raise NotImplementedError("The type of unit cell that you typed has not been calculated at the moment.")

    def build_bcc(self):
        """
        Function to build a BCC unit cell by defining the nodes of the unit cell, and the
        struts that connect them to each other.
        """
        # ---- Define Nodes ----
        # 8 cube corners
        unit_corners = np.array(list(product([0, 1], repeat=3)))
        corners = unit_corners * np.array([self.l_circ_cell, self.l_long_cell, self.h_cell])

        # center node
        center = np.array([[self.l_circ_cell / 2, self.l_long_cell / 2, self.h_cell / 2]])

        self.nodes = np.vstack((corners, center))

        center_index = len(self.nodes) - 1

        # ---- Extract Strut Start/End Points ----
        self.struts = []

        for i in range(8):
            strut = st.Strut(self.nodes[i], self.nodes[center_index], self.d_rod)
            strut.compute_geometry()
            self.struts.append(strut)


    def calc_total_uc_length(self):
        """
        Function that calculates the total length of the struts into the unit cell.

        Returns:
            uc_rod_length(float): Total length of the struts in the unit cell. (mm)
        """
        uc_rod_length = 0
        for key in self.struts:
            uc_rod_length += key.length

        return uc_rod_length

    #
    # def calc_tot_cond_area(self):
    #     """
    #     Calculates the total conduction area of the unit cells.
    #
    #     Returns:
    #
    #     TODO:
    #         - not very clear definition of face struts, but works for octet truss and fcc
    #     """
    #     uc_cond_area = 0
    #     for i, key in enumerate(self.struts):
    #         coord_dif = key[1] - key[0]
    #         if coord_dif == [1, 1, 0] or coord_dif == [1, 0, 1] or coord_dif == [0, 1, 1]:
    #             uc_cond_area += np.pi * self.d_rod ** 2 / 8
    #         else:

    def calc_unit_cell_density(self):
        """
        Function to calculate the relative density of the unit cell, by subtracting 3
            times the sphere (with radius d_rod/2) at the intersection of the 3 cylinders

        Returns:
            unit_cell_dens (float): Relative density of the unit cell (-)

        TODO:
            - not very accurate approach for thick struts
        """
        rho_0 = self.l_long_cell * self.l_circ_cell * self.h_cell
        rho_ls = np.pi * self.d_rod ** 2 / 4 * self.calc_total_uc_length() - 3 * 4 / 3 * np.pi * (self.d_rod / 2) ** 3

        unit_cell_dens = rho_ls / rho_0

        return unit_cell_dens

    def calc_uc_density_spi(self):
        """
        Function to calculate the density of a unit cell according to SPI's paper.
        Source: Z:\06_Publications\SPI_Publications\thermo-03-00034.pdf
        This is the most accurate method until now.

        TODO:
            - find out why it works better with the c_factor (considered on SPI's thesis)
            - implement the constants for other types of unit cell as well
        """
        beta_angle = np.atan(self.h_cell / np.sqrt(self.l_long_cell ** 2 + self.l_circ_cell ** 2))
        gamma_angle = np.atan(self.h_cell / self.l_circ_cell)
        f1 = 0
        f2 = 0
        f3 = 0
        f4 = 0
        coeff = 1
        delta_bc = 0
        delta_fc = 0
        delta_z = 0

        if self.uni_type == 'bcc':
            f1 = 0
            f2 = 0
            f3 = 2.993
            f4 = 3.34
            coeff = 1
            delta_bc = 4

        elif self.uni_type == 'octet':
            f1 = 3.061
            f2 = 1.954
            coeff = 2.5
            delta_fc = 10

        vol_intersection = coeff * 16 / 3 * (self.d_rod / 2) ** 3 * (
                f1 / np.sin(np.pi - 2 * gamma_angle) + f2 / np.sin(np.pi / 2 - gamma_angle) +
                f3 / np.sin(np.pi - 2 * beta_angle) + f4 / np.sin(np.pi / 2 - beta_angle))

        rho_0 = self.l_long_cell * self.l_circ_cell * self.h_cell
        rho_ls = np.pi * self.d_rod ** 2 / 4 * self.h_cell * (delta_bc / np.sin(beta_angle) + delta_fc / np.sin(
            gamma_angle) + delta_z) - vol_intersection

        unit_cell_dens = rho_ls / rho_0

        return unit_cell_dens

    def calc_e_mod_1(self):
        """
        Function to calculate the Elastic Modulus of the lattice structure in the 1-direction,
        according to the paper of JBÜ (Z:\06_Publications\JBU_Publications\2022_MAMS_published\
        [Bühring et al.] - Elastic axial stiffness properties of lattice structures Analytical
        approach and experimental validation for bcc and f2ccz unit cells).

        Returns:
            uc_e_mod_1(float): elastic modulus in hoop direction. (MPa)
        """
        omega = np.arctan(self.h_cell / np.sqrt(self.l_circ_cell ** 2 + self.l_long_cell ** 2))

        uc_e_mod_1 = 2 * np.sqrt(2) * np.pi * self.material.e_mod * (self.d_rod / 2) ** 2 / (self.h_cell / 2) ** 2 * (1 + 12 * (self.d_rod / 2) ** 2 / (self.h_cell / 2) ** 2 * (np.sin(omega) ** 2 + 1) * np.tan(omega) ** 2) * np.sin(omega) ** 2 * np.cos(omega)

        return uc_e_mod_1

    def calc_e_mod_2(self):
        """
        Function to calculate the Elastic Modulus of the lattice structure in the 2-direction,
        according to the paper of JBÜ (Z:\06_Publications\JBU_Publications\2022_MAMS_published\
        [Bühring et al.] - Elastic axial stiffness properties of lattice structures Analytical
        approach and experimental validation for bcc and f2ccz unit cells).

        Returns:
            uc_e_mod_2(float): elastic modulus in longizudinal direction. (MPa)
        """
        omega = np.arctan(self.h_cell / np.sqrt(self.l_circ_cell ** 2 + self.l_long_cell ** 2))

        uc_e_mod_2 = 2 * np.sqrt(2) * np.pi * self.material.e_mod * (self.d_rod / 2) ** 2 / (self.h_cell / 2) ** 2 * (1 + 12 * (self.d_rod / 2) ** 2 / (self.h_cell / 2) ** 2 * (np.sin(omega) ** 2 + 1) * np.tan(omega) ** 2) * np.sin(omega) ** 2 * np.cos(omega)

        return uc_e_mod_2

    def calc_e_mod_3(self):
        """
        Function to calculate the Elastic Modulus of the lattice structure in the 3-direction,
        according to the paper of JBÜ (Z:\06_Publications\JBU_Publications\2022_MAMS_published\
        [Bühring et al.] - Elastic axial stiffness properties of lattice structures Analytical
        approach and experimental validation for bcc and f2ccz unit cells).

        Returns:
            uc_e_mod_3(float): elastic modulus in radial direction. (MPa)
        """
        omega = np.arctan(self.h_cell / np.sqrt(self.l_circ_cell ** 2 + self.l_long_cell ** 2))

        uc_e_mod_3 = 8 * np.pi * self.material.e_mod * (self.d_rod / 2) ** 2 / (self.h_cell / 2) ** 2 * (1 + 12 * (self.d_rod / 2) ** 2 / (self.h_cell / 2) ** 2 * np.cos(omega) ** 2) * np.sin(omega) ** 3 * np.tan(omega) ** 2

        return uc_e_mod_3

    def calc_overhang_angle(self, print_dir=3):
        """
        Function to calculate the overhang angle (from the vertical) of the lattice during the printing (overhang
            angles over 45 degrees introduce instablity and low printing quality).

        Args:
            print_dir (int): indicates the direction of printing, 1 for x, 2 for y or 3 for z.
                Note: x direction is the longitudinal one, y the circumferential and z the
                radial (with direction to the outer side of the tank).

        Returns:
            overhang_angle (float): overhang angle in degrees.
        """

        if print_dir == 1:
            print_dim_x = self.l_circ_cell
            print_dim_y = self.h_cell
            print_dim_z = self.l_long_cell
        elif print_dir == 2:
            print_dim_x = self.h_cell
            print_dim_y = self.l_long_cell
            print_dim_z = self.l_circ_cell
        elif print_dir == 3:
            print_dim_x = self.l_long_cell
            print_dim_y = self.l_circ_cell
            print_dim_z = self.h_cell
        else:
            raise ValueError("print_dir value can only be 1 for x, 2 for y or 3 for z.")

        overhang_angle = np.rad2deg(np.acos(print_dim_z / np.sqrt(print_dim_x ** 2 + print_dim_y ** 2 + print_dim_z ** 2)))

        return overhang_angle

    def calc_eff_therm_cond(self):
        """
        Function to calculate the effective thermal conductivity of a unit cell. At the moment suitable only for BCC.

        Returns:
             eff_th_cond(float): effective thermal conductivity of the lattice structure in W/(mK).

        TODO:
            Make it general for multiple unit cell types.
        """
        z_levels = []
        for struts in self.struts:
            if struts.p1[2] not in z_levels:
                z_levels.append(struts.p1[2])
            if struts.p2[2] not in z_levels:
                z_levels.append(struts.p2[2])

        # q_dt_par is the heat flow over temperature difference
        str_per_level = [[] for _ in range(len(z_levels) - 1)]

        for i in range(len(z_levels) - 1):
            for strut in self.struts:
                if (strut.p1[2] == z_levels[i] and strut.p2[2] == z_levels[i + 1]) or \
                        (strut.p2[2] == z_levels[i] and strut.p1[2] == z_levels[i + 1]):
                    str_per_level[i].append(strut)

        dT_all = 150
        dT = []
        q_par = []
        q_all_inv = 0
        for i, level in enumerate(str_per_level):
            dT.append(dT_all / len(z_levels))
            q_par.append(0)
            for j, str in enumerate(level):
                q_par[i] += self.material.therm_cond * np.pi * (str.d_rod / 2) ** 2 * np.cos(str.alpha) * dT_all / (z_levels[i + 1] - z_levels[i])

            q_all_inv += 1 / np.abs(q_par[i])

        q_all = 1 / q_all_inv

        eff_th_cond = self.h_cell * q_all / (self.l_circ_cell * self.l_long_cell * dT_all)

        return eff_th_cond


#
# if __name__ == '__main__':
#     p511 = mat.Material(name='p511', rho=7.8e-9, e_mod=204700, nu=0.297, s_yield=380, s_ult=690, e_break=0.5,
#                         therm_cond=15)
#     bcc = UnitCell(name='bcc unit cell', uni_type='bcc', material=p511, l_long_cell=6, l_circ_cell=6, h_cell=6.5,
#                    d_rod=2)
#
#     print("The ratio of used volume of the unit cell to the total volume of the unit cell is: " + str(
#         bcc.calc_unit_cell_density()))
#     print(
#         "The ratio of used volume of the unit cell to the total volume of the unit cell according to SPI's paper is: " + str(
#             bcc.calc_uc_density_spi()))
#     print(
#         "The ratio of used volume of the unit cell to the total volume of the unit cell according to cube subtraction: " + str(
#             bcc.calc_uc_density_cube()))
#     print('The equivalent elastic modulus of the lattice is ' + str(bcc.calc_uc_elasticity()) + ' MPa.')

if __name__ == '__main__':
    p511 = mat.Material(name='p511', rho=7.8e-9, e_mod=204700, nu=0.297, s_yield=380, s_ult=690, e_break=0.5,
                        therm_cond=15)
    bcc = UnitCell(name='bcc unit cell', uni_type='bcc', material=p511, l_long_cell=10, l_circ_cell=10,
                     h_cell=10,
                     d_rod=0.5)
    print("Relative Density: " + str(bcc.calc_unit_cell_density()))
    # print(
    #     "The ratio of used volume of the unit cell to the total volume of the unit cell according to SPI's paper is: " + str(
    #         bcc.calc_uc_density_spi()))
    print("The overhang angle in degrees when printing in the radial direction is: " + str(bcc.calc_overhang_angle(print_dir=2)))
    print("The equivalent elastic modulus of the UC in 1-direction is: " + str(bcc.calc_e_mod_1()))
    print("The equivalent elastic modulus of the UC in 2-direction is: " + str(bcc.calc_e_mod_2()))
    print("The equivalent elastic modulus of the UC in 3-direction is: " + str(bcc.calc_e_mod_3()))
    print("The effective thermal conductivity is: " + str(bcc.calc_eff_therm_cond()))
    print(90-bcc.calc_overhang_angle(print_dir=2))
