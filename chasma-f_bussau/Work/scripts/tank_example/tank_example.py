# -*- coding: utf-8 -*-
"""
Created on 03.04.2025

@author: i_valais
"""
import numpy as np

import Work.scripts.implementations.revolv_struct as geom
import Work.scripts.implementations.material as mat
import Work.scripts.implementations.lattice_struct as lat
import Work.scripts.implementations.sheet_implement as sheet

# Material Definition
isolation_mat = mat.Material(name='aero', rho=5.4e-12, e_mod=204000, nu=0.3, s_yield=380, s_ult=690, e_break=0.5,
                             therm_cond=0.04)
p511 = mat.Material(name='p511', rho=7.8e-9, e_mod=204000, nu=0.3, s_yield=380, s_ult=690, e_break=0.5,
                    therm_cond=15., rad_emiss=0.4)

# Lattice Definition
lat = lat.UnitCell(name='bcc unit cell', uni_type='bcc', material=p511, l_long_cell=10., l_circ_cell=10., h_cell=10.,
                   d_rod=0.5)

# Sheet Definition
iso_layer = sheet.Sheet(name='iso_layer', material=isolation_mat, thickness=10, lattice=None)
metal_layer = sheet.Sheet(name='metal_layer', material=p511, thickness=1., lattice=None)
lat_layer = sheet.Sheet(name='bcc_layer', material=p511, thickness=10., lattice=lat)
metal_layer_in = sheet.Sheet(name='metal_layer', material=p511, thickness=1., lattice=None)

# Sheet Order
tank_sheets = [metal_layer, lat_layer, metal_layer, lat_layer, metal_layer]

# Tank Geometry
tank = geom.RevolutionGeom(name='tank', tank_radius=100., tot_length=400., sheets=tank_sheets)

vol_tank = round(tank.calc_vol() * 1e-9, 5)
surf_in_tank = round(tank.calc_surf()[-1] * 1e-6, 3)

