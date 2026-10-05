# -*- coding: utf-8 -*-
"""
Created on 13.02.2026
 
@author: i_valais
"""

import numpy as np
from scipy.optimize import fsolve
import time

from Work.scripts.implementations import revolv_struct as geom
from Work.scripts.implementations import material as mat
from Work.scripts.implementations import lattice_struct as lat
from Work.scripts.implementations import sheet_implement as sheet
from Work.scripts.calculations import calc_th_resistance as th_res

from Work.scripts.implementations import implement_fluid as fl


class ThermalNetwork:
    """
    Class to iterate for the calculation of the heat transfer through the wall. This function
    uses the electric circuit analog to calculate the heat transfer and the temperatures at
    each node.
    """

    def __init__(self):
        """
        Function to initialise the class ThermalNetwork.
        """
        self.n_nodes = 0
        self.connections = []
        self.T_fixed = {}

    # --------------------------------------------------
    def add_node(self, T_fixed=None):
        """
        Function to add a node on the circuit analog.

        Args:
            T_fixed(float): Fixed temperatures in K at the nodes where the temperature is known.

        Returns:
            idx(int): index of the node added.
        """
        idx = self.n_nodes
        self.n_nodes += 1

        if T_fixed is not None:
            self.T_fixed[idx] = T_fixed

        return idx

    # --------------------------------------------------
    def add_connection(self, i, j, R_function):
        """
        Function to add a connection between to nodes with a specific thermal resistance in between.

        Args:
            i(int): index of the first node of the connection.
            j(int): index of the second node of the connection.
            R_function(obj): Thermal resistance given the temperatures of node i and j.
        """
        self.connections.append((i, j, R_function))

    # --------------------------------------------------
    def _collapse_parallel(self, T):
        """
        Function that groups the parallel connections and computes the equivalent thermal resistance.
        """
        from collections import defaultdict

        grouped = defaultdict(list)

        for i, j, Rf in self.connections:
            key = tuple(sorted((i, j)))
            grouped[key].append(Rf)

        R_equiv = {}

        for key, Rfuncs in grouped.items():

            G_sum = 0.0
            i, j = key

            for Rf in Rfuncs:
                R = Rf(T[i], T[j])
                G_sum += 1.0 / R

            R_equiv[key] = 1.0 / G_sum

        return R_equiv

    def solve(self, tol=1e-6, max_iter=100):
        """
        Function to solve iteratively the system of equations from the circuit analog.

        Args:
            tol(float): tolerance for conversion.
            max_iter(int): Maximum number of interation before raising the error that the iteration isn't converging.
        :param tol:
        :param max_iter:

        Returns:
            T(list): list with surface temperatures from the all outer to the all inner in K.
            Q(float): heat transferred through the wall in mW.
        """

        n = self.n_nodes

        # Initial guess: linear interpolation
        T = np.linspace(
            self.T_fixed[min(self.T_fixed.keys())],
            self.T_fixed[max(self.T_fixed.keys())],
            n
        )

        for _ in range(max_iter):

            R_equiv = self._collapse_parallel(T)

            # total resistance (chain assumption)
            R_total = sum(R_equiv.values())

            T_in = self.T_fixed[min(self.T_fixed.keys())]
            T_out = self.T_fixed[max(self.T_fixed.keys())]

            Q = (T_in - T_out) / R_total

            T_new = np.zeros(n)
            T_new[0] = T_in

            for i in range(n - 1):
                R = R_equiv[(i, i + 1)]
                T_new[i + 1] = T_new[i] - Q * R

            if np.max(np.abs(T_new - T)) < tol:
                self.T = T_new
                self.Q = Q
                return T_new, Q

            T = T_new

        raise RuntimeError("Network did not converge")


    def calc_tank(self, tank, fl_out, fl_in):
        """
        Function to calculate the heat transfer in the case of a tank.

        Args:
            tank(obj): Tank object.

        Returns:
            T(list): list with surface temperatures from the all outer to the all inner in K.
            Q(float): heat transferred through the wall in mW.
        """
        num_node = len(tank.sheets) + 3

        node = []

        node.append(self.add_node(T_fixed=fl_out.temperature))

        for i in range(1, num_node - 1):
            node.append(self.add_node())

        node.append(self.add_node(T_fixed=fl_in.temperature))

        calc = th_res.ThermalResistance()

        self.add_connection(node[0], node[1], lambda Ti, Tj: calc.res_conv(surf_area=tank.calc_surf()[0],
                                                                           temp_fl=Ti,
                                                                           temp_surf=Tj,
                                                                           char_length=2 * tank.tank_radius,
                                                                           fluid=fl_out))

        self.add_connection(node[-2], node[-1], lambda Ti, Tj: calc.res_conv(surf_area=tank.calc_surf()[-1],
                                                                             temp_fl=Tj,
                                                                             temp_surf=Ti,
                                                                             char_length=2 * (
                                                                                     tank.tank_radius - tank.tot_thickness),
                                                                             fluid=fl_in))

        for i, key in enumerate(tank.sheets):
            if key.lattice is None:
                self.add_connection(node[i + 1], node[i + 2], lambda Ti, Tj, i=i, key=key: calc.res_cond(
                    area_max=np.mean(np.array([tank.calc_surf()[i], tank.calc_surf()[i+1]])),
                    k_coeff=key.material.therm_cond,
                    thickness=key.thickness))

            else:
                self.add_connection(node[i + 1], node[i + 2], lambda Ti, Tj, i=i, key=key: calc.res_cond(
                    area_max=np.mean(np.array([tank.calc_surf()[i], tank.calc_surf()[i+1]])),
                    k_coeff=key.lattice.calc_eff_therm_cond(),
                    thickness=key.thickness))

                self.add_connection(node[i + 1], node[i + 2], lambda Ti, Tj, i=i, key=key: calc.res_rad(
                    area_max=np.mean(np.array([tank.calc_surf()[i], tank.calc_surf()[i+1]])),
                    temp_1=Ti,
                    temp_2=Tj,
                    epsilon_1=tank.sheets[i - 1].material.rad_emiss,
                    epsilon_2=tank.sheets[i + 1].material.rad_emiss))

        tank_solution = self.solve()

        return tank_solution

    def calc_bo_rate(self, gas, liquid):
        """
        Function to calculate the boil-off rate of the LH2 in the tank.

        Args:
            gas(obj): H2 gas fluid object.
            liquid(obj): H2 liquid fluid object.

        Returns:
            bo_mass(float): Boil-off mass rate in ton/s.
        """
        heat_transfer = self.solve()[-1]
        h_fg = gas.enthalpy - liquid.enthalpy

        bo_mass = heat_transfer / h_fg

        return bo_mass


if __name__ == '__main__':
    start = time.time()

    # Material Definition
    isolation_mat = mat.Material(name='aero', rho=5.4e-12, e_mod=204000., nu=0.3, s_yield=380., s_ult=690., e_break=0.5,
                                 therm_cond=0.04)
    p511 = mat.Material(name='p511', rho=7.8e-9, e_mod=204000., nu=0.3, s_yield=380., s_ult=690., e_break=0.5,
                        therm_cond=15., rad_emiss=0.000001)
    grey_v4 = mat.Material(name='grey_v4', rho=1.15e-9, e_mod=2800., nu=0.3, s_yield=35., s_ult=65., e_break=0.06,
                           therm_cond=0.28, rad_emiss=0.4)

    # Fluid Definition
    air_20 = fl.Fluid(name='air_20', nu=15.35, therm_cond=2.569e-2, prandtl=0.7148, temperature=293)
    lh2_10bar = fl.Fluid(name='lh2_20K', name_lib='LH2_sat', pressure=0.2)
    gh2_10bar = fl.Fluid(name='gh2_20K', name_lib='GH2_sat', pressure=0.2)

    # Lattice Definition
    bcc = lat.UnitCell(name='bcc unit cell', uni_type='bcc', material=p511, l_long_cell=10., l_circ_cell=10.,
                       h_cell=15.,
                       d_rod=.5)
    octet = lat.UnitCell(name='octet unit cell', uni_type='octet', material=grey_v4, l_long_cell=10., l_circ_cell=10.,
                         h_cell=10.,
                         d_rod=1.)

    # Sheet Definition
    iso_layer = sheet.Sheet(name='Isolation Layer', material=isolation_mat, thickness=15., lattice=None)
    metal_layer = sheet.Sheet(name='Not-Inner Cover Layer', material=p511, thickness=0.8, lattice=None)
    lat_layer = sheet.Sheet(name='BCC Lattice Layer', material=bcc.material, thickness=bcc.h_cell, lattice=bcc)
    metal_layer_in = sheet.Sheet(name='Inner Cover Layer', material=p511, thickness=0.8, lattice=None)

    # Sheet Order
    tank_sheets = [metal_layer, lat_layer, metal_layer, lat_layer, metal_layer]

    # Tank Geometry
    tank = geom.RevolutionGeom(name='tank', tank_radius=399., tot_length=800., sheets=tank_sheets)

    # Temperature Calculation
    net = ThermalNetwork()

    T_solution, Q = net.calc_tank(tank=tank, fl_out=air_20, fl_in=lh2_10bar)
    bo_rate_2 = net.calc_bo_rate(gas=gh2_10bar, liquid=lh2_10bar)
    vol_tank = tank.calc_vol() * 1e-9
    mass_lh2 = tank.calc_vol() * lh2_10bar.density * 1e3
    bo_rate_per_day = bo_rate_2 / mass_lh2 * 1e3 * 86400 * 100

    # Heat Transfer Calculation
    print("UC Thermal Conductivity [mW/mK]: ", lat_layer.lattice.calc_eff_therm_cond() * 1000)
    print("Temperatures [K]:", T_solution)
    print("Heat flow [W]:", Q / 1000)
    print("Tank Volume (m^3): " + str(vol_tank))
    print("LH2 Mass (kg): " + str(mass_lh2))
    print("Boil-off Rate [g/s]:", bo_rate_2 * 1e6)
    print("Boil-off Rate [%/day]: " + str(bo_rate_per_day))
