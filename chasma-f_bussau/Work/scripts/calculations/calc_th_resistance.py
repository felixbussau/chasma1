# -*- coding: utf-8 -*-
"""
Created on 13.01.2026
 
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


class ThermalResistance:
    """
    In this class the thermal resistance in the case of a single convection,
    conduction or radiation will be calculated, as well as the combination
    of thermal resistances in parallel or in series.
    """

    def __init__(self):
        """
        Function to initialise class ThermalResistance. Until now there is no need for implementations
        """
        pass


    def res_conv(self, surf_area, temp_fl, temp_surf, char_length, fluid):
        """
        Function to calculate the thermal resistance due to convection.

        Args:
            surf_area(float): Area of surface of convection in mm^2.
            temp_fl(float): Temperature of the fluid (infinity) in K.
            temp_surf(float): Temperature on the surface of convection in K.
            char_length(float): Characteristic length of the geometry in mm.
            fluid(obj): Fluid object.

        Returns:
            r_conv(float): Thermal resistance due to convection in K/(m*W).
        """
        g = 9810
        beta = 1 / temp_fl
        if temp_fl > temp_surf:
            grashof = g * char_length ** 3 * beta * (
                    temp_fl - temp_surf) / fluid.nu ** 2
        else:
            grashof = g * char_length ** 3 * beta * (
                    temp_surf - temp_fl) / fluid.nu ** 2
        nusselt_out = 0.56 * (fluid.prandtl ** 2 * grashof / (0.864 + fluid.prandtl)) ** 0.25 + 2
        h_fl = nusselt_out * fluid.therm_cond / char_length

        r_conv = 1 / (h_fl * surf_area)

        return r_conv

    def res_cond(self, area_max, k_coeff, thickness):
        """
        Function to calculate the thermal resistance due to conduction.

        Args:
            area_max(float): Maximum area out of the two surfaces in mm^2. (Conservative approach)
            k_coeff(float): Thermal conductivity coefficient in W/(mK).
            thickness(float): Thickness, through which the conduction takes place in mm.

        Returns:
            r_cond(float): Thermal resistance due to conduction in K/(m*W).
        """
        r_cond = thickness / (k_coeff * area_max)

        return r_cond

    def res_rad(self, area_max, temp_1, temp_2, epsilon_1, epsilon_2):
        """
       Function to calculate the thermal resistance due to radiation.

        Args:
            area_max(float): Maximum area out of the two surfaces in mm^2. (Conservative approach)
            temp_1(float): Temperature of the outer surface in K.
            temp_2(float): Temperature of the inner surface in K.
            epsilon_1(float): Radiation coefficient of the outer surface.
            epsilon_2(float): Radiation coefficient of the inner surface.

        Returns:
            r_rad(float): Thermal resistance due to radiation in K/(m*W).
        """
        sigma = 5.6696e-11
        epsilon_eff = 1 / (1 / epsilon_1 + 1 / epsilon_2 - 1)

        h_rad = sigma * epsilon_eff * (temp_1 + temp_2) * (temp_1 ** 2 + temp_2 ** 2)

        r_rad = 1 / (h_rad * area_max)

        return r_rad

    def res_parallel(self, res_list):
        """
       Function to calculate the equivalent thermal resistance of a parallel connection.

        Args:
            res_list(list): List of thermal resistances that are parallel to each other.

        Returns:
            res_par(float): Equivalent thermal resistance of a parallel connection in K/(m*W).
        """
        res_inv = 0
        for resistance in res_list:
            res_inv += 1 / (resistance)

        res_par = 1 / res_inv

        return res_par

    def res_series(self, res_list):
        """
       Function to calculate the equivalent thermal resistance of a connection in series.

        Args:
            res_list(list): List of thermal resistances that are in series to each other.

        Returns:
            res_par(float): Equivalent thermal resistance of a connection in series in K/(m*W).
        """
        res_ser = sum(res_list)

        return res_ser


if __name__ == '__main__':

    start = time.time()

    # Material Definition
    isolation_mat = mat.Material(name='aero', rho=5.4e-12, e_mod=204000., nu=0.3, s_yield=380., s_ult=690., e_break=0.5,
                                 therm_cond=0.04)
    p511 = mat.Material(name='p511', rho=7.8e-9, e_mod=204000., nu=0.3, s_yield=380., s_ult=690., e_break=0.5,
                        therm_cond=15., rad_emiss=0.4)
    grey_v4 = mat.Material(name='grey_v4', rho=1.15e-9, e_mod=2800., nu=0.3, s_yield=35., s_ult=65., e_break=0.06,
                           therm_cond=0.35, rad_emiss=0.99)

    # Fluid Definition
    air_20 = fl.Fluid(name='air_20', nu=15.35, therm_cond=2.569e-2, prandtl=0.7148, temperature=293)
    lh2_10bar = fl.Fluid(name='lh2_20K', name_lib='LH2_sat', pressure=1.)
    gh2_10bar = fl.Fluid(name='gh2_20K', name_lib='GH2_sat', pressure=1.)

    # Lattice Definition
    bcc = lat.UnitCell(name='bcc unit cell', uni_type='bcc', material=p511, l_long_cell=10., l_circ_cell=10.,
                       h_cell=10.,
                       d_rod=0.35)
    octet = lat.UnitCell(name='octet unit cell', uni_type='octet', material=grey_v4, l_long_cell=10., l_circ_cell=10.,
                         h_cell=15.,
                         d_rod=1.)

    # Sheet Definition
    iso_layer = sheet.Sheet(name='Isolation Layer', material=isolation_mat, thickness=15., lattice=None)
    metal_layer = sheet.Sheet(name='Not-Inner Cover Layer', material=p511, thickness=1., lattice=None)
    lat_layer = sheet.Sheet(name='BCC Lattice Layer', material=p511, thickness=10., lattice=bcc)
    metal_layer_in = sheet.Sheet(name='Inner Cover Layer', material=p511, thickness=2., lattice=None)

    # Sheet Order
    tank_sheets = [metal_layer, lat_layer, metal_layer]

    # Tank Geometry
    tank = geom.RevolutionGeom(name='tank', tank_radius=100., tot_length=400., sheets=tank_sheets)

    # Lattice effective thermal conductivity
    alpha = np.arcsin(bcc.h_cell / np.sqrt(bcc.l_long_cell ** 2 + bcc.l_circ_cell ** 2 + bcc.h_cell ** 2))
    beta = np.arctan(bcc.l_circ_cell / bcc.l_long_cell)

    bcc_eff_therm_cond = bcc.d_rod ** 2 / (2 * np.sin(alpha)) * (np.arctan(1 / (np.sin(alpha) * np.tan(beta))) - np.arctan(-np.tan(beta) / np.sin(alpha)))


    #Resistance calculations
    calc = ThermalResistance()
    temp_profile = [188.51, 188.47, 33.48, 33.44]
    res_conv_air = calc.res_conv(surf_area=tank.calc_surf()[0], temp_fl=293.15, temp_surf=temp_profile[0], char_length=2 * tank.tank_radius, fluid=air_20)
    res_cond_sheet0 = calc.res_cond(area_max=tank.calc_surf()[0], k_coeff=p511.therm_cond, thickness=tank.sheets[0].thickness)
    res_cond_sheet1 = calc.res_cond(area_max=tank.calc_surf()[2], k_coeff=p511.therm_cond, thickness=tank.sheets[2].thickness)
    res_cond_lat0 = calc.res_cond(area_max=tank.calc_surf()[1], k_coeff=bcc_eff_therm_cond, thickness=tank.sheets[1].lattice.h_cell)
    res_rad = calc.res_rad(area_max=tank.calc_surf()[1], temp_1=temp_profile[1], temp_2=temp_profile[2], epsilon_1=p511.rad_emiss, epsilon_2=p511.rad_emiss)
    res_conv_lh2 = calc.res_conv(surf_area=tank.calc_surf()[3], temp_fl=26., temp_surf=temp_profile[-1], char_length=2 * (tank.tank_radius - tank.tot_thickness), fluid=lh2_10bar)

    res_par = calc.res_parallel([res_rad, res_cond_lat0])
    res_tot = calc.res_series(res_list=[res_conv_air, res_cond_sheet0, res_par, res_cond_sheet1, res_conv_lh2])

    q = (air_20.temperature - lh2_10bar.temperature) / res_tot / 1000


    end = time.time()

    print(res_cond_lat0)
    print(res_rad)
    print(res_tot)
    print(q)





