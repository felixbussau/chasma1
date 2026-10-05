# -*- coding: utf-8 -*-
"""
Created on 12.01.2026
 
@author: i_valais
"""


import numpy as np
from scipy.optimize import fsolve
import time

from Work.scripts.implementations import revolv_struct as geom
from Work.scripts.implementations import material as mat
from Work.scripts.implementations import lattice_struct as lat
from Work.scripts.implementations import sheet_implement as sheet

from Work.scripts.implementations import implement_fluid as fl


class HeatTransfer:
    """
    Class to calculate the temperature distribution and the heat transfer through the wall.

    TODO:
        - during the calculation of the lattice conduction only the bcc lattice type is taken into account. Make it more general.
        - make plot of temperature distribution.
        - make plot of heat transfers (probably not important).
        - calculate effective thermal conduction coefficient (check "to do" in the function).
        - add the possibility of contact resistance in the case of Formschluss.
        - convection on the inside only takes into account the liquid and not the vapor H2.
    """

    def __init__(self, tank, fluid_out, fluid_in):
        """
        Function to initialise class HeatTransfer.

        Args:
            tank (obj): Tank object.
            fluid_out (obj): Fluid object at the outer side of the tank.
            fluid_in (obj): Fluid object at the inner side of the tank.
        """
        self.tank = tank
        self.fluid_out = fluid_out
        self.fluid_in = fluid_in

        g = 9810  # mm/s^2
        self.stefan_boltzmann = 5.6696e-11
        self.area = self.tank.calc_surf()

        # initialisation of environment affected regions (convection)
        self.temp_env = self.fluid_out.temperature
        self.temp_surf_out = 0.8 * self.temp_env
        char_length_out = 2 * self.tank.tank_radius

        # initialisation of content affected regions (convection)
        self.temp_h2 = self.fluid_in.temperature
        self.temp_surf_in = 1.2 * self.temp_h2
        char_length_in = 2 * self.tank.tank_radius - self.tank.tot_thickness

        # outer fluid thermal conductivity calculation
        beta_out = 1 / self.temp_env
        grashof_out = g * char_length_out ** 3 * beta_out * (
                self.temp_env - self.temp_surf_out) / self.fluid_out.nu ** 2
        nusselt_out = 0.56 * (self.fluid_out.prandtl ** 2 * grashof_out / (0.864 + self.fluid_out.prandtl)) ** 0.25 + 2
        self.h_fl_out = nusselt_out * self.fluid_out.therm_cond / char_length_out

        # inner fluid thermal conductivity calculation
        beta_in = 1 / self.temp_h2
        grashof_in = g * char_length_in ** 3 * beta_in * (
                self.temp_surf_in - self.temp_h2) / self.fluid_in.nu ** 2
        nusselt_in = 0.56 * (self.fluid_in.prandtl ** 2 * grashof_in / (0.864 + self.fluid_in.prandtl)) ** 0.25 + 2
        self.h_fl_in = nusselt_in * self.fluid_in.therm_cond / char_length_in

        # Thermal conduction area and length of the lattice structures
        for i, key in enumerate(self.tank.sheets):
            if key.lattice is not None:
                k_p511 = key.material.therm_cond
                d_strut = key.lattice.d_rod
                l_long_cell = key.lattice.l_long_cell
                l_circ_cell = key.lattice.l_circ_cell
                h_cell = key.lattice.h_cell
                alpha = np.arcsin(h_cell / np.sqrt(l_long_cell ** 2 + l_circ_cell ** 2 + h_cell ** 2))
                beta = np.arctan(l_circ_cell / l_long_cell)
                key.area_tc_lat = d_strut ** 2 / (2 * np.sin(alpha)) * (
                            np.arctan(1 / (np.sin(alpha) * np.tan(beta))) - np.arctan(- np.tan(beta) / np.sin(alpha)))
                key.k_eff = k_p511 * key.area_tc_lat / (l_long_cell * l_circ_cell)





    def thermal_eqs(self, vars):
        """
        Function to construct the thermal equilibrium equations on every surface.

        Returns:
            eqs (list): List of the thermal equilibrium equations on every surface.
        """
        temp_prof = vars
        eqs = []

        for i in range(len(self.tank.sheets) + 2):
            if i == 0:

                g = 9810
                char_length_out = 2 * self.tank.tank_radius

                beta_out = 1 / self.temp_env
                grashof_out = g * char_length_out ** 3 * beta_out * (
                        self.temp_env - self.temp_surf_out) / self.fluid_out.nu ** 2
                nusselt_out = 0.56 * (
                        self.fluid_out.prandtl ** 2 * grashof_out / (0.864 + self.fluid_out.prandtl)) ** 0.25 + 2
                h_fl_out = nusselt_out * self.fluid_out.therm_cond / char_length_out

                r_conv_air = 1 / (h_fl_out * self.area[0])

                eqs.append(self.temp_env - temp_prof[0] - r_conv_air * temp_prof[-1])

            elif i == len(self.tank.sheets) + 1:

                g = 9810
                char_length_in = 2 * self.tank.tank_radius - self.tank.tot_thickness

                beta_in = 1 / self.temp_h2
                grashof_in = g * char_length_in ** 3 * beta_in * (
                        self.temp_surf_in - self.temp_h2) / self.fluid_in.nu ** 2
                nusselt_in = 0.56 * (
                        self.fluid_in.prandtl ** 2 * grashof_in / (0.864 + self.fluid_in.prandtl)) ** 0.25 + 2
                h_fl_in = nusselt_in * self.fluid_in.therm_cond / char_length_in

                r_conv_h2 = 1 / (h_fl_in * self.area[-1])

                eqs.append(temp_prof[-2] - self.temp_h2 - r_conv_h2 * temp_prof[-1])

            else:
                if self.tank.sheets[i - 1].lattice is None:

                    r_cond = self.tank.sheets[i - 1].thickness / (
                                self.tank.sheets[i - 1].material.therm_cond * self.area[i - 1])

                    eqs.append(temp_prof[i - 1] - temp_prof[i] - r_cond * temp_prof[-1])

                else:

                    eps_out = self.tank.sheets[i - 2].material.rad_emiss
                    eps_in = self.tank.sheets[i].material.rad_emiss
                    eps_eff = 1 / (1 / eps_out + 1 / eps_in - 1)
                    h_rad = self.stefan_boltzmann * eps_eff * (temp_prof[i] + temp_prof[i - 1]) * (
                                temp_prof[i] ** 2 + temp_prof[i - 1] ** 2)
                    r_rad = 1 / (h_rad * self.area[i - 1])

                    alpha = np.arcsin(self.tank.sheets[i - 1].lattice.h_cell / np.sqrt(
                        self.tank.sheets[i - 1].lattice.l_long_cell ** 2 + self.tank.sheets[
                            i - 1].lattice.l_circ_cell ** 2 + self.tank.sheets[i - 1].lattice.h_cell ** 2))
                    beta = np.arctan(
                        self.tank.sheets[i - 1].lattice.l_circ_cell / self.tank.sheets[i - 1].lattice.l_long_cell)
                    area_tc_lat = self.tank.sheets[i - 1].lattice.d_rod ** 2 / (2 * np.sin(alpha)) * (
                                np.arctan(1 / (np.sin(alpha) * np.tan(beta))) - np.arctan(
                            - np.tan(beta) / np.sin(alpha)))
                    k_eff = self.tank.sheets[i - 1].material.therm_cond * area_tc_lat / (
                                self.tank.sheets[i - 1].lattice.l_long_cell * self.tank.sheets[
                            i - 1].lattice.l_circ_cell)
                    r_cond_eff = self.tank.sheets[i - 1].lattice.h_cell / (k_eff * self.area[i - 1])

                    r_tot = 1 / (1 / r_rad + 1 / r_cond_eff)

                    eqs.append(temp_prof[i - 1] - temp_prof[i] - r_tot * temp_prof[-1])

        return eqs

    def temp_solve(self, initial_guess=None):
        """
        Function to solve the function thermal_eqs.

        Args:
            initial_guess (list): List of the initial guess of temperature profile throughout
                    the layers. The length of it should be the length of the sheets plus 2.

        Returns:
            solution (list): List of the calculated temperatures on every surface of the wall (K).
        """
        if initial_guess is None:
            initial_guess = list(np.zeros(len(self.tank.sheets) + 1))
            count_thickness = 0

            for i, key in enumerate(self.tank.sheets):
                initial_guess[i] = (self.tank.tot_thickness - count_thickness) / self.tank.tot_thickness * (
                        self.temp_surf_out - self.temp_surf_in) + self.temp_surf_in
                count_thickness += key.thickness

            q_tot_guess = 10000  # mW
            initial_guess.append(q_tot_guess)

        solution = fsolve(lambda vars: self.thermal_eqs(vars), initial_guess)

        return solution

    def calc_therm_cond_eff(self):
        """
        Function to calculate the effective thermal conductivity of the calculated wall.
        The area used in the calculation is the inner one, while it gives the conservative
        solution.

        Returns:
            therm_cond_eff (float):

        TODO:
            - decide if you 're going to use the temperatures of the inner and outer wall,
                        or the ones of the environment and the content of the tank.
        """
        temp_out = self.temp_solve()[0]
        temp_in = self.temp_solve()[-2]

        therm_cond_eff = (self.tank.tot_thickness / self.area[-1]) * self.temp_solve()[-1] / (temp_out - temp_in)

        return therm_cond_eff

    def calc_boil_off_rate(self, gas, liquid):
        """
        Function to calculate the mass boil-off-rate of the content of the tank.

        Returns:
            boil_off (float): Boil-off-rate of the hydrogen (ton/s).

        """
        heat_transfer = self.temp_solve()[-1]
        h_fg = gas.enthalpy - liquid.enthalpy

        boil_off_rate = heat_transfer / h_fg

        return boil_off_rate


if __name__ == '__main__':

    start = time.time()

    # Material Definition
    isolation_mat = mat.Material(name='aero', rho=5.4e-12, e_mod=204000., nu=0.3, s_yield=380., s_ult=690., e_break=0.5,
                                 therm_cond=0.04)
    p511 = mat.Material(name='p511', rho=7.8e-9, e_mod=204000., nu=0.3, s_yield=380., s_ult=690., e_break=0.5,
                        therm_cond=15., rad_emiss=0.4)
    grey_v4 = mat.Material(name='grey_v4', rho=1.15e-9, e_mod=2800., nu=0.3, s_yield=35., s_ult=65., e_break=0.06,
                           therm_cond=0.35, rad_emiss=0.4)

    # Fluid Definition
    air_20 = fl.Fluid(name='air_20', nu=15.35, therm_cond=2.569e-2, prandtl=0.7148, temperature=293)
    lh2_10bar = fl.Fluid(name='lh2_20K', name_lib='LH2_sat', pressure=1.)
    gh2_10bar = fl.Fluid(name='gh2_20K', name_lib='GH2_sat', pressure=1.)

    # Lattice Definition
    bcc = lat.UnitCell(name='bcc unit cell', uni_type='bcc', material=grey_v4, l_long_cell=10., l_circ_cell=10.,
                       h_cell=10.,
                       d_rod=2.)
    octet = lat.UnitCell(name='octet unit cell', uni_type='octet', material=grey_v4, l_long_cell=10., l_circ_cell=10.,
                         h_cell=15.,
                         d_rod=1.)

    # Sheet Definition
    iso_layer = sheet.Sheet(name='Isolation Layer', material=isolation_mat, thickness=15., lattice=None)
    metal_layer = sheet.Sheet(name='Not-Inner Cover Layer', material=grey_v4, thickness=1.5, lattice=None)
    lat_layer = sheet.Sheet(name='BCC Lattice Layer', material=grey_v4, thickness=10., lattice=bcc)
    metal_layer_in = sheet.Sheet(name='Inner Cover Layer', material=grey_v4, thickness=2., lattice=None)

    # Sheet Order
    tank_sheets = [metal_layer, lat_layer, metal_layer]

    # Tank Geometry
    tank = geom.RevolutionGeom(name='tank', tank_radius=100., tot_length=400., sheets=tank_sheets)

    # Temperature Calculation
    model = HeatTransfer(tank=tank, fluid_in=lh2_10bar, fluid_out=air_20)

    var_calc = model.temp_solve()
    q_tot = var_calc[-1]
    temps = var_calc[:-1]
    end = time.time()
    for i, T in enumerate(temps):
        print(f"T[{i}] = {T:.2f} K")

    # Heat Transfer Calculation
    print('The total heat transfer is: ' + str(q_tot / 1000) + ' Watt.')
    print('Effective thermal conductivity of the wall is: ' + str(model.calc_therm_cond_eff()) + ' W/mK.')
    print('The boil-off_rate is ' + str(model.calc_boil_off_rate(gas=gh2_10bar, liquid=lh2_10bar) * 1e6) + ' g/s.')
    print('Mass of the tank: ' + str(tank.calc_mass() * 1000) + ' kg.')
    print(f"Execution time: {end - start:.6f} seconds")
