# -*- coding: utf-8 -*-
"""
Created on 10.02.2025
 
@author: i_valais
"""

import numpy as np


class Material:
    """
    Class that creates a material.

    Args:
        name (str): Name of the material.
        rho (float): Density of the material.
        e_mod (float): E Modulus of the material.
        nu (float): Poisson's Ratio of the material.
        s_yield (float): Yield Strength of the material.
        s_ult (float): Ultimate Strength of the material.
        e_break (float): Elongation at break.
        therm_cond (float): Coefficient of thermal conductivity.

    TODO:
        - Add add_therm_cond function. Other thermal properties can also be necessary.
    """

    def __init__(self, name, e_mod, nu, rho=None, s_yield=None, s_ult=None, e_break=None, therm_cond=None, rad_emiss=None):
        """
        Function to initialize the class Material.
        """
        self.name = name
        self.e_mod = e_mod
        self.nu = nu
        self.rho = rho
        self.s_yield = s_yield
        self.s_ult = s_ult
        self.e_break = e_break
        self.therm_cond = therm_cond
        self.rad_emiss = rad_emiss

    def calc_am_max_stress(self, austenitic, am_factor, r_p10t=None, r_p02t=None, r_m20=None, r_mt=None):
        """
        Function to calculate the maximum nominative stress of the additive
        manufactured material. This applies at austenitic steels, with an
        elongation at break larger than 0.35 (DIN EN 13445-3). For the additive
        manufactured material there are more requirements (DIN/TS 17026). The
        R_p1.0 is for 20 Celsius degrees or cryogenic temperatures always larger
        as R_p0.2, therefore if the R_p1.0 is not provided, the R_p0.2 is used as
        yield strength to calculate the am_max_stress (DIN EN 10028-7).

        Args:
            austenitic (bool): Flag to determine austenitic or not steel.
            am_factor (float): Factor applied to the maximum stress of the
                material according to the additive manufacturing procedure
                (0.5 - 0.85).
            r_p02t (float): Yield strength at 0.2% strain at the temperature of heat treatment.
            r_p10t (float): Yield strength at 1.0% strain at the temperature of heat treatment.
            r_mt (float): Ultimate strength at the temperature of heat treatment.
            r_m20 (float): Ultimate strength at 20 degrees Celsius.

        Returns:
            am_max_stress (float): Maximum stress of the additive manufactured
                material.
        """
        if r_p02t is None:
            r_p02t = self.s_yield

        if r_p10t is None:
            r_p10t = self.s_yield

        if r_m20 is None:
            r_m20 = self.s_ult

        if r_mt is None:
            r_mt = self.s_ult

        if am_factor < 0.5 or am_factor > 0.85:
            raise ValueError(
                "The additive manufacturing factor for the calculation of the maximum nominative strength must be between 0.5 and 0.85.")

        if austenitic == True:
            if self.e_break >= 0.35:
                f = np.max([(r_p10t / 1.5), np.min([r_p10t / 1.2, r_mt / 3])])
            elif self.e_break >= 0.3:
                f = r_p10t / 1.5
            else:
                raise ValueError(
                    "No function is given for the maximum nominal strength in the case of austenitic steels with elongation at break under 0.3.")
        else:
            f = np.min([r_p02t / 1.5, r_m20 / 2.4])

        nom_strength = am_factor * f

        return nom_strength


if __name__ == '__main__':
    st_316l = Material(name='316L', rho=7.85e-9, e_mod=210000, nu=0.3, s_yield=320, s_ult=570, e_break=0.52)
    frr = st_316l.calc_am_max_stress(austenitic=0, am_factor=0.85)
    print('The maximum nominative stress is: ' + str(frr) + ' MPa.')
