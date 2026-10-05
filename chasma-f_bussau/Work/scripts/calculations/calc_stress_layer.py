# -*- coding: utf-8 -*-
"""
Created on 03.12.2025
 
@author: i_valais
"""
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

from Work.scripts.implementations import revolv_struct as geom
from Work.scripts.implementations import material as mat
from Work.scripts.implementations import lattice_struct as lat
from Work.scripts.implementations import sheet_implement as sheet


class StressTank:

    def __init__(self, tank, press_in, press_out):
        """
         Function to initialise the class StressTank.

        :param tank (obj): Tank object.
        :param press_in (float): Pressure on the inside of the tank (MPa).
        :param press_out (float): Pressure on the outside of the tank (MPa).
        """

        self.tank = tank
        self.press_in = press_in
        self.press_out = press_out

    def calc_rad_disp(self, p_in, p_out, r_in, r_out, r, e_mod, nu):
        """
        Function to calculate the displacement of a point a radius r, of a
        tank with inner and outer radius, when applying inner and outer pressure.

        :param p_in (float):
        :param p_out (float):
        :param r_in (float):
        :param r_out (float):
        :param r (float):
        :param e_mod (float):
        :param nu (float):

        :return disp (float):
        """

        k = (p_out * r_out ** 2 - p_in * r_in ** 2) / (r_out ** 2 - r_in ** 2)
        c = (p_out - p_in) * r_out ** 2 * r_in ** 2 / (r_out ** 2 - r_in ** 2)

        disp = (-k * (1 - nu) - c * (1 + nu)) / (e_mod * r)

        return disp

    def calc_stress(self, p_in, p_out, r_in, r_out, r):
        """
        Function to calculate the displacement of a point a radius r, of a
        tank with inner and outer radius, when applying inner and outer pressure.

        :param p_in (float):
        :param p_out (float):
        :param r_in (float):
        :param r_out (float):
        :param r (float):

        :return disp (float):
        """

        k = (p_out * r_out ** 2 - p_in * r_in ** 2) / (r_out ** 2 - r_in ** 2)
        c = (p_out - p_in) * r_out ** 2 * r_in ** 2 / (r_out ** 2 - r_in ** 2)

        stress_long = -k
        stress_circ = -k - c / r ** 2
        stress_rad = -k + c / r ** 2

        stress = [stress_long, stress_circ, stress_rad]

        return stress

    def calc_strain(self, p_in, p_out, r_in, r_out, r, e_mod_long, e_mod_circ, e_mod_rad, nu):
        """
        Function to calculate the displacement of a point a radius r, of a
        tank with inner and outer radius, when applying inner and outer pressure.

        :param p_in (float):
        :param p_out (float):
        :param r_in (float):
        :param r_out (float):
        :param r (float):
        :param e_mod (float):
        :param nu (float):

        :return disp (float):
        """

        k = (p_out * r_out ** 2 - p_in * r_in ** 2) / (r_out ** 2 - r_in ** 2)
        c = (p_out - p_in) * r_out ** 2 * r_in ** 2 / (r_out ** 2 - r_in ** 2)

        strain_long = 1 / e_mod_long * k * (2 * nu - 1)
        strain_circ = 1 / e_mod_circ * (k * (2 * nu - 1) - c / r ** 2 * (1 + nu))
        strain_rad = 1 / e_mod_rad * (k * (2 * nu - 1) + c / r ** 2 * (1 + nu))

        strain = [strain_long, strain_circ, strain_rad]

        return strain

    def press_eqs(self, vars):
        """

        :param vars:
        :return:
        """
        r_prof = [self.tank.tank_radius]
        r_count = self.tank.tank_radius
        for i, key in enumerate(self.tank.sheets):
            r_count -= key.thickness
            r_prof.append(r_count)
        press_prof = vars
        eqs = []

        for i, key in enumerate(self.tank.sheets):

            if i == len(self.tank.sheets) or i == len(self.tank.sheets) - 1:
                break

            else:
                if i == len(self.tank.sheets) - 2:
                    inner_p_in = self.press_in
                else:
                    inner_p_in = press_prof[i + 1]
                outer_p_in = press_prof[i]

                if i == 0:
                    outer_p_out = self.press_out
                else:
                    outer_p_out = press_prof[i - 1]
                inner_p_out = press_prof[i]
                outer_r_in = r_prof[i + 1]
                inner_r_in = r_prof[i + 2]
                outer_r_out = r_prof[i]
                inner_r_out = r_prof[i + 1]
                r = r_prof[i + 1]
                if key.lattice is None:
                    outer_e_mod = key.material.e_mod
                else:
                    outer_e_mod = key.lattice.calc_e_mod_3()
                if self.tank.sheets[i + 1].lattice is None:
                    inner_e_mod = self.tank.sheets[i + 1].material.e_mod
                else:
                    inner_e_mod = self.tank.sheets[i + 1].lattice.calc_e_mod_3()
                outer_nu = key.material.nu
                inner_nu = self.tank.sheets[i + 1].material.nu

                disp_out = self.calc_rad_disp(outer_p_in, outer_p_out, outer_r_in, outer_r_out, r, outer_e_mod,
                                              outer_nu)
                disp_in = self.calc_rad_disp(inner_p_in, inner_p_out, inner_r_in, inner_r_out, r, inner_e_mod, inner_nu)

                eqs.append(disp_out - disp_in)

        return eqs

    def press_solve(self, initial_guess=None):
        """
        Function to solve the function press_eqs.

        Args:
            initial_guess (list): List of the initial guess of pressure profile throughout
                    the layers. The length of it should be the length of the sheets minus 2.

        Returns:
            solution (list): List of the calculated pressure on every intermediate surface of the wall (MPa).
        """
        if initial_guess is None:
            initial_guess = list(np.zeros(len(self.tank.sheets) - 1))
            count_thickness = 0

            for i, key in enumerate(self.tank.sheets):
                if i == len(self.tank.sheets) - 1:
                    break
                else:
                    count_thickness += key.thickness
                    initial_guess[i] = (self.tank.tot_thickness - count_thickness) / self.tank.tot_thickness * (
                            self.press_out - self.press_in) + self.press_in

        solution = fsolve(lambda vars: self.press_eqs(vars), initial_guess)

        return solution

    def calc_results(self):
        """
        Function to calculate the results of the results of the analytical model.

        Returns:
             solution (list): List consisted 7 different lists, which are the following:
                - r_list: radius list throughout the thickness of the tank for its incrementation.
                - sigma_long: stresses on longitudinal direction of the cylindrical tank
                    along its thickness (from inside to outside) (21 data per layer).
                - sigma_long: stresses on circumferential direction of the cylindrical tank
                    along its thickness (from inside to outside) (21 data per layer).
                - sigma_rad: stresses on radial direction of the cylindrical tank
                    along its thickness (from inside to outside) (21 data per layer).
                - epsilon_long: strains on longitudinal direction of the cylindrical tank
                    along its thickness (from inside to outside) (21 data per layer).
                - epsilon_circ: strains on circumferential direction of the cylindrical tank
                    along its thickness (from inside to outside) (21 data per layer).
                - epsilon_rad: strains on radial direction of the cylindrical tank
                    along its thickness (from inside to outside) (21 data per layer).
                - disp_rad: displacements on radial direction of the cylindrical tank
                    along its thickness (from inside to outside) (21 data per layer).
        """
        r_in = self.tank.tank_radius - self.tank.tot_thickness
        r_surf_list = [r_in]
        p_list = [self.press_in]
        r_list = []
        for i, key in enumerate(reversed(self.tank.sheets)):
            r_surf_list.append(r_surf_list[i] + key.thickness)
            if i == len(self.tank.sheets) - 1:
                p_list.append(self.press_out)
            else:
                p_list.append(float(self.press_solve()[len(self.tank.sheets) - 2 - i]))
            r_list = r_list + list(np.linspace(r_surf_list[i], r_surf_list[i + 1], 21))

        r_list = [float(x) for x in r_list]

        sigma_long = []
        sigma_circ = []
        sigma_rad = []
        epsilon_long = []
        epsilon_circ = []
        epsilon_rad = []
        disp_rad = []

        for i, key in enumerate(reversed(self.tank.sheets)):
            r_in = r_surf_list[i]
            r_out = r_surf_list[i + 1]
            p_in = p_list[i]
            p_out = p_list[i + 1]
            if key.lattice is None:
                e_mod_1 = key.material.e_mod
                e_mod_2 = key.material.e_mod
                e_mod_3 = key.material.e_mod
            else:
                e_mod_1 = key.lattice.calc_e_mod_1()
                e_mod_2 = key.lattice.calc_e_mod_2()
                e_mod_3 = key.lattice.calc_e_mod_3()
            nu = key.material.nu

            for count in range(21 * i, 21 * (i + 1)):
                r = r_list[count]
                sigma_long.append(self.calc_stress(p_in, p_out, r_in, r_out, r)[0])
                sigma_circ.append(self.calc_stress(p_in, p_out, r_in, r_out, r)[1])
                sigma_rad.append(self.calc_stress(p_in, p_out, r_in, r_out, r)[2])
                epsilon_long.append(self.calc_strain(p_in, p_out, r_in, r_out, r, e_mod_2, e_mod_1, e_mod_3, nu)[0])
                epsilon_circ.append(self.calc_strain(p_in, p_out, r_in, r_out, r, e_mod_2, e_mod_1, e_mod_3, nu)[1])
                epsilon_rad.append(self.calc_strain(p_in, p_out, r_in, r_out, r, e_mod_2, e_mod_1, e_mod_3, nu)[2])
                disp_rad.append(self.calc_rad_disp(p_in, p_out, r_in, r_out, r, e_mod_3, nu))

            sigma_long = [float(x) for x in sigma_long]
            sigma_circ = [float(x) for x in sigma_circ]
            sigma_rad = [float(x) for x in sigma_rad]
            epsilon_long = [float(x) for x in epsilon_long]
            epsilon_circ = [float(x) for x in epsilon_circ]
            epsilon_rad = [float(x) for x in epsilon_rad]
            disp_rad = [float(x) for x in disp_rad]


        solution = [r_list, sigma_long, sigma_circ, sigma_rad, epsilon_long, epsilon_circ, epsilon_rad, disp_rad]

        return solution


    def profile_plot(self):
        """
        Function to plot the stress and strain profile of the cylindrical part of the tank.

        :return:
        """
        r_list = self.calc_results()[0]
        sigma_long = self.calc_results()[1]
        sigma_circ = self.calc_results()[2]
        sigma_rad = self.calc_results()[3]
        epsilon_long = self.calc_results()[4]
        epsilon_circ = self.calc_results()[5]
        epsilon_rad = self.calc_results()[6]

        r_in = self.tank.tank_radius - self.tank.tot_thickness
        r_surf_list = [r_in]
        for i, key in enumerate(reversed(self.tank.sheets)):
            r_surf_list.append(r_surf_list[i] + key.thickness)

        plt.figure(figsize=(15, 8))
        plt.suptitle('Stresses and Strain Distribution throughout the Thickness of the Cylindrical Part of the Tank')

        plt.subplot(231)
        plt.xlabel('Longitudinal Stress (MPa)')
        plt.ylabel('Radius (mm)')
        plt.grid()
        for i in r_surf_list:
            plt.axhline(y=i, color='black', linestyle='--', linewidth=1)
        plt.plot(sigma_long, r_list)

        plt.subplot(232)
        plt.xlabel('Circumferential Stress (MPa)')
        plt.ylabel('Radius (mm)')
        plt.grid()
        for i in r_surf_list:
            plt.axhline(y=i, color='black', linestyle='--', linewidth=1)
        plt.plot(sigma_circ, r_list)

        plt.subplot(233)
        plt.xlabel('Radial Stress (MPa)')
        plt.ylabel('Radius (mm)')
        plt.grid()
        for i in r_surf_list:
            plt.axhline(y=i, color='black', linestyle='--', linewidth=1)
        plt.plot(sigma_rad, r_list)

        plt.subplot(234)
        plt.xlabel('Longitudinal Strain')
        plt.ylabel('Radius (mm)')
        plt.grid()
        for i in r_surf_list:
            plt.axhline(y=i, color='black', linestyle='--', linewidth=1)
        plt.plot(epsilon_long, r_list)

        plt.subplot(235)
        plt.xlabel('Circumferential Strain')
        plt.ylabel('Radius (mm)')
        plt.grid()
        for i in r_surf_list:
            plt.axhline(y=i, color='black', linestyle='--', linewidth=1)
        plt.plot(epsilon_circ, r_list)

        plt.subplot(236)
        plt.xlabel('Radial Strain')
        plt.ylabel('Radius (mm)')
        plt.grid()
        for i in r_surf_list:
            plt.axhline(y=i, color='black', linestyle='--', linewidth=1)
        plt.plot(epsilon_rad, r_list)

        plt.show()

        # r_prof = [self.tank.tank_radius]
        # r_count = self.tank.tank_radius
        # for i, key in enumerate(self.tank.sheets):
        #     r_count -= key.thickness
        #     r_prof.append(r_count)
        #
        # for i, key in enumerate(self.tank.sheets):
        #     if i == 0:
        #         p_out = self.press_out
        #     elif i == len(self.tank.sheets) - 1:
        #         p_in = self.press_in
        #     else:
        #         p_in = self.press_solve()[i]
        #         p_out = self.press_solve()[i - 1]
        #         r_in = r_prof[i + 1]
        #         r_out = r_prof[i]
        #
        #     r_run = np.linspace(r_prof[i], r_prof[i + 1], 20)
        #
        #     for r in r_run:
        #         stress_long = self.calc_stress(p_in, p_out, r_in, r_out, r)[0]
        #         stress_circ = self.calc_stress(p_in, p_out, r_in, r_out, r)[1]
        #         stress_rad = self.calc_stress(p_in, p_out, r_in, r_out, r)[2]


if __name__ == '__main__':
    # Material Definition
    isolation_mat = mat.Material(name='aero', rho=5.4e-12, e_mod=204000., nu=0.3, s_yield=380., s_ult=690., e_break=0.5,
                                 therm_cond=0.04)
    p511 = mat.Material(name='p511', rho=7.8e-9, e_mod=204000., nu=0.3, s_yield=380., s_ult=690., e_break=0.5,
                        therm_cond=15., rad_emiss=0.4)

    # Lattice Definition
    bcc = lat.UnitCell(name='bcc unit cell', uni_type='bcc', material=p511, l_long_cell=10., l_circ_cell=10.,
                       h_cell=10.,
                       d_rod=0.5)
    octet = lat.UnitCell(name='octet unit cell', uni_type='octet', material=p511, l_long_cell=10., l_circ_cell=10.,
                         h_cell=15.,
                         d_rod=1.)

    # Sheet Definition
    iso_layer = sheet.Sheet(name='Isolation Layer', material=isolation_mat, thickness=15., lattice=None)
    metal_layer = sheet.Sheet(name='Not-Inner Cover Layer', material=p511, thickness=1., lattice=None)
    lat_layer = sheet.Sheet(name='BCC Lattice Layer', material=p511, thickness=10., lattice=bcc)
    metal_layer_in = sheet.Sheet(name='Inner Cover Layer', material=p511, thickness=2., lattice=None)

    # Sheet Order
    tank_sheets = [metal_layer, lat_layer, metal_layer, lat_layer, metal_layer]

    # Tank Geometry
    tank = geom.RevolutionGeom(name='tank', tank_radius=100., tot_length=400., sheets=tank_sheets)

    # Temperature Calculation
    model = StressTank(tank=tank, press_in=0.5, press_out=0.1)

    press = model.press_solve()
    print(model.calc_results()[5])
    model.profile_plot()
    for i, p in enumerate(press):
        print(f"P[{i}] = {p:.4f} MPa")
