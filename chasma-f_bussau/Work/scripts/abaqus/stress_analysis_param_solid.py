# -*- coding: utf-8 -*-
"""
Created on 21.04.2026
 
@author: i_valais
"""

import sys

sys.path.append(r"C:\Users\i_valais\PycharmProjects\chasma")

import Work.scripts.tank_example.tank_example as ex

import numpy as np
from abaqus import mdb
from abaqusConstants import *
from caeModules import *
from odbAccess import *
import mesh
from regionToolset import Region

# PACKAGE IMPORTATION
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *

diam0 = 0.35
diam_lin = np.linspace(0.4, 1., 7)
param_list = [0.35] # list(np.append(diam0, diam_lin))
param_type = 'strut_diameter'

max_mises_param = list(np.zeros(len(param_list)))

e11_param = list(np.zeros(len(param_list)))
e22_param = list(np.zeros(len(param_list)))
mass_param = list(np.zeros(len(param_list)))

for index, param in enumerate(param_list):
    # Parametrisation
    r_left = float(ex.tank.tank_radius)
    r_right = float(ex.tank.tank_radius)
    l_tot = float(ex.tank.tot_length)
    l_long_cell = float(ex.lat.l_long_cell)
    l_circ_cell = float(ex.lat.l_circ_cell)
    h_cell = float(ex.lat.h_cell)
    strut_diam = float(ex.lat.d_rod)
    tank_diam = float(ex.tank.tank_radius) * 2
    pressure = 1.
    t_sheet_out = float(ex.tank.sheets[0].thickness)
    t_sheet_in = float(ex.tank.sheets[-1].thickness)

    if param_type == 'strut_diameter':
        ex.lat_layer.lattice.d_rod = param
        strut_diam = float(ex.lat.d_rod)
        model_name = 'tank_p_strut_diam_' + str(int(round(param * 100))).zfill(3)

    elif param_type == 'height':
        ex.lat_layer.lattice.h_cell = param
        h_cell = float(ex.lat.h_cell)
        model_name = 'tank_p_uc_height_' + str(int(round(param)))

    elif param_type == 'pressure':
        pressure = param
        model_name = 'tank_p_press_' + str(int(round(param * 10))).zfill(2)

    elif param_type == 'l_long':
        ex.lat_layer.lattice.l_long_cell = param
        l_long_cell = float(ex.lat.l_long_cell)
        model_name = 'tank_p_l_long_' + str(int(round(param)))

    elif param_type == 'l_circ':
        ex.lat_layer.lattice.l_circ_cell = param
        l_circ_cell = float(ex.lat.l_circ_cell)
        model_name = 'tank_p_l_circ_' + str(int(round(param)))

    elif param_type == 'tank_diameter':
        ex.tank.tank_radius = param / 2
        r_left = float(ex.tank.tank_radius)
        r_right = float(ex.tank.tank_radius)
        model_name = 'tank_p_tank_diameter_' + str(int(round(param)))

    elif param_type == 'tank_length':
        ex.tank.tot_length = param
        l_tot = float(ex.tank.tot_length)
        model_name = 'tank_p_tank_length' + str(int(round(param)))

    else:
        raise ValueError("The type of the parameter studied (param_type) is not valid. Check it once more.")

    epsilon = 1e-3

    # Calculations
    thickness_tot = ex.tank.tot_thickness

    length = np.sqrt(l_long_cell ** 2 + l_circ_cell ** 2 + h_cell ** 2)
    beam_mesh_size = length / 50
    plate_mesh_size = l_long_cell / 10
    x_init_point = length * np.cos(np.arctan(l_circ_cell / l_long_cell))
    y_init_point = length * np.sin(np.arctan(l_circ_cell / l_long_cell))
    strut_radius = strut_diam / 2
    angle_xy_z = np.rad2deg(np.arcsin(h_cell / length))

    # Model Definition
    m = mdb.Model(name=model_name)

    r_list = [ex.tank.tank_radius]

    for i, key in enumerate(ex.tank.sheets):
        r_list.append(r_list[i] - key.thickness)

    # Part Definition
    m.ConstrainedSketch(name='__profile__', sheetSize=1000.)

    s = m.sketches['__profile__']

    s.ConstructionLine(point1=(0.0, -500.0), point2=(0.0, 500.0))
    s.FixedConstraint(entity=s.geometry[2])
    s.ArcByCenterEnds(center=(0, -l_tot / 2 + r_left), direction=CLOCKWISE, point1=(0, -l_tot / 2), point2=(-r_left, -l_tot/2 + r_left))
    s.ArcByCenterEnds(center=(0, -l_tot / 2 + r_left), direction=CLOCKWISE, point1=(0, -l_tot / 2 + thickness_tot), point2=(-r_left + thickness_tot, -l_tot / 2 + r_left))
    s.ArcByCenterEnds(center=(0, l_tot / 2 - r_right), direction=COUNTERCLOCKWISE, point1=(0, l_tot / 2),
                  point2=(-r_right, l_tot / 2 - r_right))
    s.ArcByCenterEnds(center=(0, l_tot / 2 - r_right), direction=COUNTERCLOCKWISE, point1=(0, l_tot / 2 - thickness_tot),
                  point2=(-r_right + thickness_tot, l_tot / 2 - r_right))
    s.Line(point1=(-r_left, -l_tot / 2 + r_left), point2=(-r_right, l_tot/2 - r_right))
    s.Line(point1=(-r_left + thickness_tot, -l_tot / 2 + r_left), point2=(-r_right + thickness_tot, l_tot / 2 - r_right))
    s.Line(point1=(0, -l_tot / 2), point2=(0, -l_tot / 2 + thickness_tot))
    s.Line(point1=(0, l_tot / 2), point2=(0, l_tot / 2 - thickness_tot))

    p = m.Part(dimensionality=THREE_D, name='tank',
                                    type=DEFORMABLE_BODY)
    p.BaseSolidRevolve(angle=360.0, flipRevolveDirection=OFF, sketch=s)

    # Partitioning
    dat_plane = p.DatumPlaneByPrincipalPlane(offset=0.0, principalPlane=XYPLANE)
    # dat_right = p.DatumPlaneByPrincipalPlane(offset=100.0, principalPlane=XZPLANE)
    # dat_left = p.DatumPlaneByPrincipalPlane(offset=-100.0, principalPlane=XZPLANE)

    cell = p.cells.getByBoundingBox(- l_tot - 10., - l_tot - 10., - l_tot - 10., l_tot + 10., l_tot + 10., l_tot + 10.)
    p.PartitionCellByDatumPlane(cells=cell, datumPlane=p.datums[dat_plane.id])
    # p.PartitionCellByDatumPlane(cells=cell, datumPlane=p.datums[dat_right.id])
    # p.PartitionCellByDatumPlane(cells=cell, datumPlane=p.datums[dat_left.id])

    s = m.ConstrainedSketch(name='__profile__', sheetSize=1000.)
    # geom = s.geometry.findAt(coordinates=(0, l_tot/2 - r_list[1]),)
    # s.FixedConstraint(entity=geom)

    for i, key in enumerate(ex.tank.sheets):
        if i == len(ex.tank.sheets):
            break
        s.ArcByCenterEnds(center=(0, -l_tot / 2 + r_left), direction=CLOCKWISE, point1=(0, -l_tot / 2 + r_left - r_list[i+1]),
                          point2=(-r_list[i+1], -l_tot / 2 + r_left))
        s.ArcByCenterEnds(center=(0, l_tot / 2 - r_right), direction=COUNTERCLOCKWISE,
                        point1=(0, l_tot / 2 - r_right + r_list[i + 1]),
                        point2=(-r_list[i + 1], l_tot / 2 - r_right))
        s.Line(point1=(-r_list[i+1], -l_tot / 2 + r_left), point2=(-r_list[i + 1], l_tot / 2 - r_right))

    face = p.faces.findAt(coordinates=(0, l_tot/2 - r_right + r_list[1], 0),)
    p.PartitionFaceBySketch(faces=face, sketch=s)
    del s

    # from here on there seems to be a problem
    e_sweep_1 = p.edges.findAt((0, -l_tot / 2 + r_left, r_left))
    e_sweep_2 = p.edges.findAt((0, -l_tot / 2 + r_left, -r_left))
    for i, key in enumerate(ex.tank.sheets):
        if i == len(ex.tank.sheets):
            break
        e_left = p.edges.findAt(coordinates=(0, - l_tot / 2 + r_left - r_list[i + 1], 0), )
        e_right = p.edges.findAt(coordinates=(0, l_tot / 2 - r_right + r_list[i + 1], 0), )
        e_center = p.edges.findAt(coordinates=(-r_list[i + 1], 0, 0), )

        p.PartitionCellBySweepEdge(cells=cell, edges=(e_left, e_right, e_center), sweepPath=e_sweep_1)
        p.PartitionCellBySweepEdge(cells=cell, edges=(e_left, e_right, e_center), sweepPath=e_sweep_2)















