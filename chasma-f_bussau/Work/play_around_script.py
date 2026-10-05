# -*- coding: utf-8 -*-
"""
Created on 12.01.2026
 
@author: i_valais
"""

import numpy as np

l_circ_cell = 10.
l_long_cell = 10.
h_cell = 10.
d_str = 2.
t_sheet = 1.5
k_gv4 = 0.28
area_surf = 200 * 200
t_tot = h_cell + 2 * t_sheet
k_air = 0.025
DT = 15
T_meas = 0
stefan_boltzmann = 5.670374419e-11
temp_1 = 275.65 + T_meas + 0
temp_2 = 290.65 + T_meas - 0
eps_resin_bot = 0.0042 * (temp_1 - 273.15) + 0.6727
eps_resin_top = 0.0042 * (temp_2 - 273.15) + 0.6727
alpha = np.arcsin(h_cell / np.sqrt(l_long_cell ** 2 + l_circ_cell ** 2 + h_cell ** 2))
beta = np.arctan(l_circ_cell / l_long_cell)


eps_eff = 1 / (1 / eps_resin_top + 1 / eps_resin_bot - 1)
area_cond_uc = d_str ** 2 / (2 * np.sin(alpha)) * (np.arctan(1 / (np.sin(alpha) * np.tan(beta))) - np.arctan(-np.tan(beta) / np.sin(alpha)))
area_cond_uc_all = 1 * 20 * 20 * area_cond_uc + 38 * np.pi * d_str ** 2 / (4 * (1 + (np.sin(alpha)) ** 2))
area_air_between = area_surf - area_cond_uc_all

print(area_cond_uc_all)

res_cond_sheet = t_sheet / (k_gv4 * area_surf)
res_cond_uc_all = h_cell / (k_gv4 * area_cond_uc_all)
res_cond_air_2 = h_cell / (k_air * area_air_between)
res_rad_surf = 1 / (stefan_boltzmann * eps_eff * (temp_1 + temp_2) * (temp_1 ** 2 + temp_2 ** 2) * area_air_between)

res_par = 1 / (1 / res_cond_air_2 + 1 / res_cond_uc_all + 1 / res_rad_surf)
res_ser = res_par + res_cond_sheet * 2

q = DT / res_ser

k_eff = q * t_tot / (area_surf * DT) * 1000

print(k_eff)
print(eps_resin_top)
print(eps_resin_bot)



l_circ_cell = 10.
l_long_cell = 10.
h_cell = 10.
d_str = .35
t_sheet = 1.
k_gv4 = 15.
area_surf = 200 * 200
t_tot = h_cell + 2 * t_sheet
k_air = 0.025 # * 0.00000000001
DT = 15
T_meas = 0
stefan_boltzmann = 5.670374419e-11
temp_1 = 275.65 + T_meas + 0
temp_2 = 290.65 + T_meas - 0
eps_resin_bot = 0.4
eps_resin_top = 0.41
alpha = np.arcsin(h_cell / np.sqrt(l_long_cell ** 2 + l_circ_cell ** 2 + h_cell ** 2))
beta = np.arctan(l_circ_cell / l_long_cell)


eps_eff = 1 / (1 / eps_resin_top + 1 / eps_resin_bot - 1)
area_cond_uc = d_str ** 2 / (2 * np.sin(alpha)) * (np.arctan(1 / (np.sin(alpha) * np.tan(beta))) - np.arctan(-np.tan(beta) / np.sin(alpha)))
area_cond_uc_all = 1 * 20 * 20 * area_cond_uc + 41 * np.pi * d_str ** 2 / (4 * (1 + (np.sin(alpha)) ** 2))
area_air_between = area_surf - area_cond_uc_all

print(area_cond_uc)

res_cond_sheet = t_sheet / (k_gv4 * area_surf)
res_cond_uc_all = h_cell / (k_gv4 * area_cond_uc_all)
res_cond_air_2 = h_cell / (k_air * area_air_between)
res_rad_surf = 1 / (stefan_boltzmann * eps_eff * (temp_1 + temp_2) * (temp_1 ** 2 + temp_2 ** 2) * area_air_between)

res_par = 1 / (1 / res_cond_air_2 + 1 / res_cond_uc_all + 1 / res_rad_surf)
res_ser = res_par + res_cond_sheet * 2

q = DT / res_ser

k_eff = q * t_tot / (area_surf * DT) * 1000

print(k_eff)
print(eps_resin_top)
print(eps_resin_bot)





l_circ_cell = 10.
l_long_cell = 10.
h_cell = 10.
d_str = 0.5
t_sheet = 1.5
k_gv4 = 0.28
k_p511 = 15.
area_surf = 200 * 200
t_tot = h_cell + 2 * t_sheet
k_air = 0.025
DT = 15
T_meas = 0
stefan_boltzmann = 5.670374419e-11
temp_1 = 275.65 + T_meas + 0
temp_2 = 290.65 + T_meas - 0
eps_resin_bot = 0.0042 * (temp_1 - 273.15) + 0.6727
eps_resin_top = 0.0042 * (temp_2 - 273.15) + 0.6727
alpha = np.arcsin(h_cell / np.sqrt(l_long_cell ** 2 + l_circ_cell ** 2 + h_cell ** 2))
beta = np.arctan(l_circ_cell / l_long_cell)
area_cond_uc = (d_str / 2) ** 2 / (2 * np.sin(alpha)) * (np.arctan(1 / (np.sin(alpha) * np.tan(beta))) - np.arctan(-np.tan(beta) / np.sin(alpha)))
area_inclined_uc = np.pi * (d_str / 2) ** 2 / np.sin(alpha)
h_inc = d_str / (2 * np.cos(alpha))

print("--------------------------")
A1 = (d_str / 2) / (np.cos(alpha) * (area_inclined_uc - area_cond_uc)) * np.log(area_inclined_uc / area_cond_uc)
A2 = 1 / area_inclined_uc * (h_cell / 2 - 2 * h_inc)
print(A1)
print(A2)
print(1000 * k_p511 * 2 * h_cell / (l_circ_cell * l_long_cell * (1 * A2 + 2 * A1)))
print(1000 * k_p511 * area_cond_uc / (l_circ_cell * l_long_cell))


