# -*- coding: utf-8 -*-
"""
Created on 25.02.2026
 
@author: i_valais
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import time

from Work.scripts.implementations import revolv_struct as geom
from Work.scripts.implementations import material as mat
from Work.scripts.implementations import lattice_struct as lat
from Work.scripts.implementations import sheet_implement as sheet
from Work.scripts.calculations import calc_th_resistance as th_res

from Work.scripts.implementations import implement_fluid as fl

class Structural:
    """
    In this class the stresses and strains of the individual layers of
    the cylindrical part of the wall are calculated and plotted. Also
    the maximum stresses of the given struts of the lattice structure
    is also calculated.
    """

    def __init__(self, tank, press_in, press_out):
        """
        Function to initialise the class Structural.

        Args:
        tank (obj): Tank object.
        press_in (float): Pressure on the inside of the tank (MPa).
        press_out (float): Pressure on the outside of the tank (MPa).
        """
        self.tank = tank
        self.press_in = press_in
        self.press_out = press_out


    def plane_strain_system(self):
        """
        Function to set up and solve the system of equations of the tank under the assumption
        of generalised plane strain and internal and/or external pressure with axial load
        at the ends, according to the paper of Kardomateas, 2001 (Elasticity Solutions for
        Sandwich Orthotropic Cylindrical Shells Under External/Internal Pressure or Axial
        Force). The paper is solving the two problems (external/internal pressure and axial
        load) independently, and the two solutions are combined.

        Returns:
            c (list): list 2 times the quantity of the tank layers plus one. If n is the
            number of the layers, he first n components of the list contain the C_2 constants
            (defined in the paper) for all the layers in order from outer to inner layer,
            the second n components contain respectively the C_3 constants and the last
            component is the constant axial strain epsilon_0 defined in the paper.
        """

        self.a_11 = []
        self.a_22 = []
        self.a_33 = []
        self.a_12 = []
        self.a_13 = []
        self.a_23 = []
        self.beta_11 = []
        self.beta_22 = []
        self.beta_12 = []
        self.k = []
        self.ksi = []
        for i,key in enumerate(self.tank.sheets):
            if key.lattice is None:
                self.a_11.append(1 / key.material.e_mod)
                self.a_22.append(1 / key.material.e_mod)
                self.a_33.append(1 / key.material.e_mod)
                self.a_12.append(- key.material.nu / key.material.e_mod)
                self.a_13.append(- key.material.nu / key.material.e_mod)
                self.a_23.append(- key.material.nu / key.material.e_mod)

            else:
                self.a_11.append(1 / key.lattice.calc_e_mod_3())
                self.a_22.append(1 / key.lattice.calc_e_mod_1())
                self.a_33.append(1 / key.lattice.calc_e_mod_2())
                self.a_12.append(- key.material.nu / key.lattice.calc_e_mod_1())
                self.a_13.append(- key.material.nu / key.lattice.calc_e_mod_2())
                self.a_23.append(- key.material.nu / key.lattice.calc_e_mod_2())

            self.beta_11.append(self.a_11[i] - self.a_13[i] ** 2 / self.a_33[i])
            self.beta_22.append(self.a_22[i] - self.a_23[i] ** 2 / self.a_33[i])
            self.beta_12.append(self.a_12[i] - self.a_13[i] * self.a_23[i] / self.a_33[i])
            self.k.append(np.sqrt(self.beta_11[i] / self.beta_22[i]))

            if self.beta_11[i] == self.beta_22[i]:
                self.ksi.append(0)
            else:
                self.ksi.append((self.a_13[i] - self.a_23[i]) / (self.beta_22[i] - self.beta_11[i]))

        r = [self.tank.tank_radius]
        for i, key in enumerate(self.tank.sheets):
            r.append(r[i] - key.thickness)

        l = len(self.tank.sheets)

        # radial stress at the outer and inner side of the tank is equal to the outer and inner pressure
        a_matrix = np.zeros((2*l+1, 2*l+1))
        b_vec = np.zeros(2*l+1)

        # Outer Pressure equation
        a_matrix[0,0] = r[0] ** (self.k[0] - 1)
        a_matrix[0,l] = r[0] ** (- self.k[0] - 1)
        a_matrix[0,-1] = self.ksi[0] / self.a_33[0]
        b_vec[0] = -self.press_out

        # Inner Pressure equation
        a_matrix[-2,l-1] = r[-1] ** (self.k[-1] - 1)
        a_matrix[-2,-2] = r[-1] ** (- self.k[-1] - 1)
        a_matrix[-2,-1] = self.ksi[-1] / self.a_33[-1]
        b_vec[-2] = -self.press_in

        # Axial Force BC Equation
        p_ax = self.press_in * np.pi * r[-1] ** 2
        g1_factor = 0
        for i, sheet in enumerate(self.tank.sheets):

            # iterating for producing the g2 component of the paper
            a_matrix[-1, i] = (self.a_13[i] + self.a_23[i] * self.k[i]) / (self.a_33[i] * (self.k[i] + 1)) * (r[i] ** (self.k[i] + 1) - r[i+1] ** (self.k[i] + 1))

            # iterating for producing the g3 component of the paper
            if self.k[i] != 1:
                a_matrix[-1, l+i] = (self.a_13[i] - self.a_23[i] * self.k[i]) / (self.a_33[i] * (-self.k[i] + 1)) * (r[i] ** (-self.k[i] + 1) - r[i+1] ** (-self.k[i] + 1))
            else:
                a_matrix[-1, l+i] = 0

            # iterating for creating the factor of epsilon_0 of the g1 factor of the paper
            g1_factor += (1 - (self.a_13[i] + self.a_23[i]) / self.a_33[i] * self.ksi[i]) * (r[i] ** 2 - r[i+1]**2) / (2 * self.a_33[i])

        a_matrix[-1, -1] = g1_factor
        b_vec[-1] = -p_ax / (2 * np.pi)

        # Radial stress and radial deformation interaction conditions between the layers
        for i, key in enumerate(self.tank.sheets):
            if i == len(self.tank.sheets) - 1:
                break

            # Radial Stress equation
            a_matrix[2*i+1, i] = r[i+1] ** (self.k[i] - 1)
            a_matrix[2*i+1, l+i] = r[i+1] ** (- self.k[i] - 1)
            a_matrix[2*i+1, i+1] = -r[i+1] ** (self.k[i + 1] - 1)
            a_matrix[2*i+1, l+i+1] = -r[i+1] ** (- self.k[i + 1] - 1)
            a_matrix[2*i+1, -1] = self.ksi[i] / self.a_33[i] - self.ksi[i+1] / self.a_33[i+1]

            # Radial Deformation equations
            a_matrix[2*i+2, i] = ((self.beta_11[i] + self.k[i] * self.beta_12[i]) / self.k[i]) * r[i+1] ** self.k[i]
            a_matrix[2*i+2, l+i] = -((self.beta_11[i] - self.k[i] * self.beta_12[i]) / self.k[i]) * r[i+1] ** (-self.k[i])
            a_matrix[2*i+2, i+1] = -((self.beta_11[i + 1] + self.k[i + 1] * self.beta_12[i + 1]) / self.k[i + 1]) * r[i+1] ** self.k[i + 1]
            a_matrix[2*i+2, l+i+1] = ((self.beta_11[i + 1] - self.k[i + 1] * self.beta_12[i + 1]) / self.k[i + 1]) * r[i+1] ** (-self.k[i + 1])
            a_matrix[2*i+2, -1] = (1 / self.a_33[i]) * (self.a_13[i] + self.ksi[i] * (self.beta_11[i] + self.beta_12[i])) * r[i+1] - (1 / self.a_33[i+1]) * (self.a_13[i+1] + self.ksi[i+1] * (self.beta_11[i+1] + self.beta_12[i+1])) * r[i+1]

        c = np.linalg.solve(a_matrix, b_vec)

        return c


    def calc_plane_strain(self):
        """
        Function to calculate the the stresses, the strains and the radial displacement on
        three points of each layer, namely the inner, the middle and the outer surface.
        The stresses and strains are calculated in the radial, the circumferential and the
        longitudinal direction for each of the aforementioned points.

        Returns:
            - r_plot(list): list that contains (with that order) the outer, middle and
                inner surface radii of each layer, from the outermost to the inner most layer.
            - sigma_rr_prof(list): list that contains the radial stress of every point in
                r_plot list. (MPa)
            - sigma_tt_prof(list): list that contains the hoop stress of every point in
                r_plot list. (MPa)
            - sigma_zz_prof(list): list that contains the longitudinal stress of every point in
                r_plot list. (MPa)
            - epsilon_rr_prof(list): list that contains the radial strain of every point in
                r_plot list. (-)
            - epsilon_tt_prof(list): list that contains the hoop strain of every point in
                r_plot list. (-)
            - epsilon_zz_prof(list): list that contains the longitudinal strain of every point in
                r_plot list. (-)
            - disp_rr_prof(list): list that contains the radial displacements of every point in
                r_plot list. (mm)
        """
        c = self.plane_strain_system()
        l = len(tank.sheets)
        c_2 = c[0: l]
        c_3 = c[l: 2*l]
        eps_0 = c[-1]

        sigma_rr_prof = []
        sigma_tt_prof = []
        sigma_zz_prof = []
        epsilon_rr_prof = []
        epsilon_tt_prof = []
        epsilon_zz_prof = []
        disp_rr_prof = []

        r_list = [self.tank.tank_radius]
        for i, key in enumerate(self.tank.sheets):
            r_list.append(r_list[i] - key.thickness)

        r_plot = []

        for i, key in enumerate(self.tank.sheets):
            for r in np.linspace(r_list[i], r_list[i + 1], 3):
                sigma_rr_prof.append(eps_0 / self.a_33[i] * self.ksi[i] + c_2[i] * r ** (self.k[i] - 1) + c_3[i] * r ** (-self.k[i] - 1))
                sigma_tt_prof.append(eps_0 / self.a_33[i] * self.ksi[i] + c_2[i] * self.k[i] * r ** (self.k[i] - 1) - c_3[i] * self.k[i] * r ** (-self.k[i] - 1))
                sigma_zz_prof.append(eps_0 / self.a_33[i] * (1 - (self.a_13[i] + self.a_23[i]) / self.a_33[i] * self.ksi[i]) - c_2[i] * ((self.a_13[i] + self.a_23[i] * self.k[i]) / self.a_33[i]) * r ** (self.k[i] - 1) - c_3[i] * ((self.a_13[i] - self.a_23[i] * self.k[i]) / self.a_33[i]) * r ** (-self.k[i] - 1))
                disp_rr_prof.append(eps_0 / self.a_33[i] * (self.a_13[i] + self.ksi[i] * (self.beta_11[i] + self.beta_12[i])) * r + c_2[i] * ((self.beta_11[i] + self.k[i] * self.beta_12[i]) / self.k[i]) * r ** self.k[i] - c_3[i] * ((self.beta_11[i] - self.k[i] * self.beta_12[i]) / self.k[i]) * r ** (-self.k[i]))
                epsilon = np.array([[self.a_11[i], self.a_12[i], self.a_13[i]], [self.a_12[i], self.a_22[i], self.a_23[i]], [self.a_13[i], self.a_23[i], self.a_33[i]]]) @ np.array([[sigma_rr_prof[-1]], [sigma_tt_prof[-1]], [sigma_zz_prof[-1]]])

                epsilon_rr_prof.append(epsilon[0])
                epsilon_tt_prof.append(epsilon[1])
                epsilon_zz_prof.append(epsilon[2])
                r_plot.append(r)

        return [r_plot, sigma_rr_prof, sigma_tt_prof, sigma_zz_prof, epsilon_rr_prof, epsilon_tt_prof, epsilon_zz_prof, disp_rr_prof]

    def plot_plane_strain(self):
        """
        Function to plot the stresses, strains and the radial displacement calculated at the
        calc_plane_strain function, as a function of the radius.
        """
        r_plot = self.calc_plane_strain()[0]
        sigma_rr_prof = self.calc_plane_strain()[1]
        sigma_tt_prof = self.calc_plane_strain()[2]
        sigma_zz_prof = self.calc_plane_strain()[3]
        epsilon_rr_prof = self.calc_plane_strain()[4]
        epsilon_tt_prof = self.calc_plane_strain()[5]
        epsilon_zz_prof = self.calc_plane_strain()[6]
        disp_rr_prof = self.calc_plane_strain()[7]

        plt.figure()
        plt.suptitle('Stresses and Strain Distribution throughout the Thickness of the Cylindrical Part of the Tank')

        plt.subplot(331)
        plt.xlabel('Radial Stress (MPa)')
        plt.ylabel('Radius (mm)')
        plt.grid()
        for i in r_plot:
            plt.axhline(y=i, color='black', linestyle='--', linewidth=1)
        plt.plot(sigma_rr_prof, r_plot)

        plt.subplot(332)
        plt.xlabel('Hoop Stress (MPa)')
        plt.ylabel('Radius (mm)')
        plt.grid()
        for i in r_plot:
            plt.axhline(y=i, color='black', linestyle='--', linewidth=1)
        plt.plot(sigma_tt_prof, r_plot)

        plt.subplot(333)
        plt.xlabel('Longitudinal Stress (MPa)')
        plt.ylabel('Radius (mm)')
        plt.grid()
        for i in r_plot:
            plt.axhline(y=i, color='black', linestyle='--', linewidth=1)
        plt.plot(sigma_zz_prof, r_plot)

        plt.subplot(334)
        plt.xlabel('Radial Strain (-)')
        plt.ylabel('Radius (mm)')
        plt.grid()
        for i in r_plot:
            plt.axhline(y=i, color='black', linestyle='--', linewidth=1)
        plt.plot(epsilon_rr_prof, r_plot)

        plt.subplot(335)
        plt.xlabel('Hoop Strain (-)')
        plt.ylabel('Radius (mm)')
        plt.grid()
        for i in r_plot:
            plt.axhline(y=i, color='black', linestyle='--', linewidth=1)
        plt.plot(epsilon_tt_prof, r_plot)

        plt.subplot(336)
        plt.xlabel('Longitudinal Strain (-)')
        plt.ylabel('Radius (mm)')
        plt.grid()
        for i in r_plot:
            plt.axhline(y=i, color='black', linestyle='--', linewidth=1)
        plt.plot(epsilon_zz_prof, r_plot)

        plt.subplot(338)
        plt.xlabel('Radial Displacement (mm)')
        plt.ylabel('Radius (mm)')
        plt.grid()
        for i in r_plot:
            plt.axhline(y=i, color='black', linestyle='--', linewidth=1)
        plt.plot(disp_rr_prof, r_plot)

        plt.show()


    def calc_junction_stress(self):
        """
        Function to calculate the stress distribution of the tank close to the junction
        between the cylindrical and hemispherical part, according to Mengzhao Long's paper
        (Stress Analysis on Hemispherical Head of Internal Pressure Vessel, 2025).

        Returns:
        """
        pass

    def calc_strut_stress(self):
        """
        Function to calculate the maximum stress on the struts of the lattice structure layers.

        Returns:
            stress_max_strut(float): Maximum stress of the struts (MPa).
        """
        all_sigma_max = []
        for i, sheet in enumerate(tank.sheets):
            sheet.sigma_max = []

            # conservative scenario: in the strain tensor only the max strains in each direction are considered
            strain_rr = float(max(self.calc_plane_strain()[4][3 * i], self.calc_plane_strain()[4][3 * i + 1],
                            self.calc_plane_strain()[4][3 * i + 2], key=abs))
            strain_tt = float(max(self.calc_plane_strain()[5][3 * i], self.calc_plane_strain()[5][3 * i + 1],
                            self.calc_plane_strain()[5][3 * i + 2], key=abs))
            strain_zz = float(max(self.calc_plane_strain()[6][3 * i], self.calc_plane_strain()[6][3 * i + 1],
                            self.calc_plane_strain()[6][3 * i + 2], key=abs))

            strain_tensor = [[strain_rr, 0, 0], [0, strain_tt, 0], [0, 0, strain_zz]]

            if sheet.lattice is None:
                continue

            for strut in sheet.lattice.struts:
                # node coordinates of the current strut
                p1 = strut.p1
                p2 = strut.p2

                # Displacements of both nodes and the difference of them
                u1 = np.array(strain_tensor) @ p1.T
                u2 = np.array(strain_tensor) @ p2.T

                du = u2 - u1

                n = strut.n_vec
                l_rod = strut.length

                # axial deformation
                delta_axial = np.dot(n, du)

                # Axial Force and Stress
                f_ax = sheet.lattice.material.e_mod * strut.cs_area / l_rod * delta_axial

                sigma_axial = f_ax / strut.cs_area

                # Displacement in the perpendicular direction of the strut and normalisation of it
                du_perp = du - delta_axial * n
                delta_b = np.linalg.norm(du_perp)

                # Bending moment and stress of te strut
                moment = 6 * sheet.lattice.material.e_mod * strut.second_moment_of_inertia * delta_b / l_rod ** 2

                sigma_bend = moment * (strut.d_rod / 2) / strut.second_moment_of_inertia

                # Both maximum and minimum stresses of the cross-section of the strut are added to the maximum stress list
                sheet.sigma_max.append(sigma_axial + sigma_bend)
                sheet.sigma_max.append(sigma_axial - sigma_bend)

            # The stress with the maximum absolut value from the list above is added to the list with the maximum stresses of each layer
            all_sigma_max.append(max(sheet.sigma_max, key=abs))

        # The maximum stress among the maximum stresses of each layer
        stress_max_strut = max(all_sigma_max, key=abs)

        return stress_max_strut




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
    bcc = lat.UnitCell(name='bcc unit cell', uni_type='bcc', material=p511, l_long_cell=10., l_circ_cell=10., h_cell=10.,
                       d_rod=1.)
    octet = lat.UnitCell(name='octet unit cell', uni_type='octet', material=grey_v4, l_long_cell=10., l_circ_cell=10., h_cell=15.,
                       d_rod=1.)

    # Sheet Definition
    iso_layer = sheet.Sheet(name='Isolation Layer', material=isolation_mat, thickness=15., lattice=None)
    metal_layer = sheet.Sheet(name='Not-Inner Cover Layer', material=p511, thickness=1., lattice=None)
    lat_layer = sheet.Sheet(name='BCC Lattice Layer', material=p511, thickness=bcc.h_cell, lattice=bcc)
    metal_layer_in = sheet.Sheet(name='Inner Cover Layer', material=p511, thickness=1., lattice=None)

    # Sheet Order
    tank_sheets = [metal_layer, lat_layer, metal_layer, lat_layer, metal_layer]

    # Tank Geometry
    tank = geom.RevolutionGeom(name='tank', tank_radius=100., tot_length=400., sheets=tank_sheets)

    model = Structural(tank=tank, press_in=lh2_10bar.pressure, press_out=0.1)

    c = model.plane_strain_system()
    print(c)

    r_plot = model.calc_plane_strain()[0]
    sigma_rr_prof = model.calc_plane_strain()[1]
    sigma_tt_prof = model.calc_plane_strain()[2]
    sigma_zz_prof = model.calc_plane_strain()[3]
    epsilon_rr_prof = model.calc_plane_strain()[4]
    epsilon_tt_prof = model.calc_plane_strain()[5]
    epsilon_zz_prof = model.calc_plane_strain()[6]
    disp_rr_prof = model.calc_plane_strain()[7]

    s_max = model.calc_strut_stress()

    # model.plot_plane_strain()


    # print(r_plot)
    # print(sigma_rr_prof)
    # print(sigma_tt_prof)
    # print(sigma_zz_prof)
    # print(epsilon_rr_prof)
    # print(epsilon_tt_prof)
    # print(epsilon_zz_prof)
    # print(disp_rr_prof)
    print("Maximum Stress at the Struts (MPa):" + str(s_max))
    print(tank.calc_vol() / 1e9)
    end = time.time()
    print(end-start)





