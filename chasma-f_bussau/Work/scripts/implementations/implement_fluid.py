# -*- coding: utf-8 -*-
"""
Created on 27.11.2025
 
@author: i_valais
"""

import numpy as np
import pandas as pd


class Fluid:
    """
    In this class a Fluid profile is implemented (and some basic parameters that
    will later help with the calculations are calculated).

    Args:
        - name (str): Name of the fluid
        - name_lib (str): Name of the fluid as implemented in the library:
            'LH2_sat': liquid hydrogen at the saturated state
            'GH2_sat': gas hydrogen at the saturated state
        - temperature (float or int): Fluid temperature (in K).
        - pressure (float or int): Fluid pressure (in MPa).
        - nu (float or int): Dynamic viscosity of the fluid (in MPa*s).
        - therm_cond (float or int): Thermal conductivity of the fluid (in mW/(mm*K) or W/(m*K)).
        - prandtl (float or int): Prandtl number of the fluid.
    """

    def __init__(self, name, name_lib=None, temperature=None,
                 pressure=None, nu=None,
                 therm_cond=None, prandtl=None):
        """
        Function to initiate Fluid class.

        """
        self.name = name
        self.name_lib = name_lib
        self.nu = nu
        self.therm_cond = therm_cond
        self.prandtl = prandtl
        self.temperature = temperature
        self.pressure = pressure
        file_exist = False

        if self.pressure is None and self.temperature is None:
            raise NotImplementedError("Either Temperature or Pressure should be given for the fluid.")

        if self.name_lib is None:
            if self.nu is None or self.therm_cond is None or self.prandtl is None:
                raise NotImplementedError(
                    "Kinematic viscosity, thermal conductivity and Prandtl number should be imported too for fluids that do not belong to the library.")
        elif self.name_lib == 'LH2_sat':
            file_exist = True
            self.file_path = 'C:/Users/i_valais/PycharmProjects/personalproject/input_files/H2_prop_1_13_bar_liquid.csv'
        elif self.name_lib == 'GH2_sat':
            file_exist = True
            self.file_path = 'C:/Users/i_valais/PycharmProjects/personalproject/input_files/H2_prop_1_13_bar_vapor.csv'

        if file_exist is True:
            with open(self.file_path, "r") as f:
                text_csv = f.read()

            text_csv = text_csv.replace(",", ".")
            text_csv = text_csv.replace("^M", "\n")
            text_csv = text_csv.replace("\r", "\n")

            from io import StringIO

            self.data_csv = np.genfromtxt(StringIO(text_csv), delimiter=";")
            self.get_properties()
        else:
            pass

    def get_closest_row(self, data, pressure):
        idx = np.nanargmin(np.abs(data[:, 1] - pressure))  # pressure = column 1
        return data[idx]

    def get_properties(self):
        """
        Function that gets the properties of the implemented fluid from the .csv file.
        """

        row = self.get_closest_row(self.data_csv, self.pressure)
        if self.name_lib == 'LH2_sat':
            self.temperature = row[0]  # K
            self.density = row[2]  # kg/m^3
            self.volume = row[3]  # m^3/kg
            self.int_energy = row[4]  # kJ/kg
            self.enthalpy = row[5]  # kJ/kg
            self.entropy = row[6]  # kJ/kg
            self.cv = row[7]  # J/g*K
            self.cp = row[8]  # J/g*K
            self.sound_spd = row[9]  # m/s
            self.jt = row[10]  # K/MPa
            self.visc = row[11]  # Pa*s
            self.therm_cond = row[12]  # W/mK
            self.surf_tens = row[13]  # N/m

        elif self.name_lib == 'GH2_sat':
            self.temperature = row[0]  # K
            self.density = row[2]  # kg/m^3
            self.volume = row[3]  # m^3/kg
            self.int_energy = row[4]  # kJ/kg
            self.enthalpy = row[5]  # kJ/kg
            self.entropy = row[6]  # kJ/kg
            self.cv = row[7]  # J/g*K
            self.cp = row[8]  # J/g*K
            self.sound_spd = row[9]  # m/s
            self.jt = row[10]  # K/MPa
            self.visc = row[11]  # Pa*s
            self.therm_cond = row[12]  # W/mK
        else:
            raise NotImplementedError("Implemented Fluid is not on the library. Try again with a valid one.")

        # Unit Conversion from existing to ton-mm-s.
        self.density = self.density * 1e-12  # ton/mm3
        self.int_energy = self.int_energy * 1e9  # Nmm/ton
        self.enthalpy = self.enthalpy * 1e9  # Nmm/ton
        self.entropy = self.entropy * 1e9  # Nmm/ton
        self.cp = self.cp * 1e9  # mm2/(s2K)
        self.visc = self.visc * 1e-6  # MPa*s

        # Extra Calculations
        self.nu = self.visc / self.density
        self.prandtl = self.visc * self.cp / self.therm_cond


if __name__ == '__main__':
    lh2 = Fluid(name='lh2', name_lib='LH2_sat', pressure=0.255)
    print(lh2.prandtl)
