# -*- coding: utf-8 -*-
"""
Created on 24.11.2025
 
@author: i_valais
"""
# ===========================================
# SciPy Optimize Template
# ===========================================

import numpy as np
from scipy import optimize

def objective(x):
    """
    Objective function to minimize.
    Replace this with your own function.
    """

    import sys

    sys.path.append(r"C:\Users\i_valais\PycharmProjects\chasma")

    import Work.scripts.tank_example.tank_example as ex
    from io import StringIO

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

    # Parametrisation
    r_left = float(ex.tank.tank_radius)
    r_right = float(ex.tank.tank_radius)
    l_tot = float(ex.tank.tot_length)
    l_long_cell = float(ex.lat.l_long_cell)
    l_circ_cell = float(ex.lat.l_circ_cell)
    h_cell = float(ex.lat.h_cell)
    strut_diam = float(ex.lat.d_rod)
    tank_diam = float(ex.tank.tank_radius) * 2
    pressure = 0.4
    cut_switch = False  # True: cut the lattice as planed, False: it is already cut
    rad_check = False

    uc_name = 'BCC_UC' + str(int(round(l_long_cell))) + 'x' + str(int(round(l_circ_cell))) + 'x' + str(
        int(round(h_cell))) + '_D' + str(int(round(strut_diam * 100))).zfill(3)
    model_name = uc_name
    file_name = 'V:/03_Project/CHASMA/03_Work/01_NX/' + uc_name + '.stp'
    part_name = model_name

    material_name = ex.tank.sheets[-2].material.name
    material_rho = ex.tank.sheets[-2].material.rho
    material_e_mod = ex.tank.sheets[-2].material.e_mod
    material_nu = ex.tank.sheets[-2].material.nu
    material_cond = ex.tank.sheets[-2].material.therm_cond
    epsilon = 1e-1

    # Calculations
    thickness_tot = 0
    for i in range(len(ex.tank.sheets)):
        thickness_tot = thickness_tot + ex.tank.sheets[i].thickness

    length = np.sqrt(l_long_cell ** 2 + l_circ_cell ** 2 + h_cell ** 2)
    lat_mesh_size = 0.2
    plate_mesh_size = l_long_cell / 10
    x_init_point = length * np.cos(np.arctan(l_circ_cell / l_long_cell))
    y_init_point = length * np.sin(np.arctan(l_circ_cell / l_long_cell))
    strut_radius = strut_diam / 2
    angle_xy_z = np.arcsin(h_cell / length)

    temp_in = 26
    temp_out = 176

    # Model Definition
    model = mdb.Model(name=model_name)
    mdb.models[model_name].setValues(absoluteZero=0, stefanBoltzmann=5.67e-11)

    # Part Definition
    # lattice
    mdb.openStep(file_name, scaleFromFile=OFF)
    mdb.models[model_name].PartFromGeometryFile(combine=False, dimensionality=
    THREE_D, geometryFile=mdb.acis, name=part_name, type=DEFORMABLE_BODY)

    mdb.models[model_name].PartFromGeometryFile(bodyNum=2, combine=False,
                                                dimensionality=THREE_D, geometryFile=mdb.acis,
                                                name=part_name + '-2',
                                                type=DEFORMABLE_BODY)
    mdb.models[model_name].PartFromGeometryFile(bodyNum=3, combine=False,
                                                dimensionality=THREE_D, geometryFile=mdb.acis,
                                                name=part_name + '-3',
                                                type=DEFORMABLE_BODY)
    mdb.models[model_name].PartFromGeometryFile(bodyNum=4, combine=False,
                                                dimensionality=THREE_D, geometryFile=mdb.acis,
                                                name=part_name + '-4',
                                                type=DEFORMABLE_BODY)
    mdb.models[model_name].PartFromGeometryFile(bodyNum=5, combine=False,
                                                dimensionality=THREE_D, geometryFile=mdb.acis,
                                                name=part_name + '-5',
                                                type=DEFORMABLE_BODY)

    del mdb.models[model_name].parts[part_name]
    del mdb.models[model_name].parts[part_name + '-2']
    del mdb.models[model_name].parts[part_name + '-3']
    del mdb.models[model_name].parts[part_name + '-4']
    mdb.models[model_name].parts.changeKey(fromName=part_name + '-5', toName=
    part_name)

    # Material Definition
    mdb.models[model_name].Material(name=material_name)
    mdb.models[model_name].materials[material_name].Density(table=((material_rho,),))
    mdb.models[model_name].materials[material_name].Elastic(table=((material_e_mod, material_nu),))
    mdb.models[model_name].materials[material_name].Conductivity(table=((material_cond,),))

    # Section Definition
    mdb.models[model_name].HomogeneousSolidSection(material=material_name, name=
    'solid_sec', thickness=None)

    # Section Assignment
    mdb.models[model_name].parts[part_name].Set(cells=
    mdb.models[model_name].parts[model_name].cells.getByBoundingBox(
        - l_circ_cell / 2 - 10.,
        - l_long_cell / 2 - 10.,
        - h_cell / 2 - 10.,
        l_circ_cell / 2 + 10.,
        l_long_cell / 2 + 10.,
        h_cell / 2 + 10.),
        name='Set-1')
    mdb.models[model_name].parts[part_name].SectionAssignment(offset=0.0,
                                                              offsetField='', offsetType=MIDDLE_SURFACE, region=
                                                              mdb.models[model_name].parts[part_name].sets['Set-1'],
                                                              sectionName=
                                                              'solid_sec', thicknessAssignment=FROM_SECTION)

    # Step Definition
    mdb.models[model_name].HeatTransferStep(name='Step-1', previous='Initial', response=STEADY_STATE, timePeriod=1.,
                                            initialInc=1., minInc=1e-5,
                                            maxInc=1.)

    # Assembly Definition
    mdb.models[model_name].rootAssembly.DatumCsysByDefault(CARTESIAN)
    if cut_switch is False:
        mdb.models[model_name].rootAssembly.Instance(dependent=ON, name='final_cut-1',
                                                     part=mdb.models[model_name].parts[part_name])


    else:
        mdb.models[model_name].rootAssembly.Instance(dependent=ON, name=cut_name1 + '-1',
                                                     part=mdb.models[model_name].parts[cut_name1])
        mdb.models[model_name].rootAssembly.Instance(dependent=ON, name=cut_name2 + '-1',
                                                     part=mdb.models[model_name].parts[cut_name2])
        mdb.models[model_name].rootAssembly.Instance(dependent=ON, name=lat_name + '-1',
                                                     part=mdb.models[model_name].parts[lat_name])

        mdb.models[model_name].rootAssembly.translate(instanceList=(
            cut_name1 + '-1',), vector=(0., 0., -h_cell / 2 - 10))
        mdb.models[model_name].rootAssembly.translate(instanceList=(
            cut_name2 + '-1',), vector=(0., 0., -l_long_cell / 2 - 10))
        mdb.models[model_name].rootAssembly.rotate(angle=90, axisDirection=(1.0, 0.0, 0.0), axisPoint=(0.0, 0.0, 0.0),
                                                   instanceList=(cut_name2 + '-1',))
        mdb.models[model_name].rootAssembly.InstanceFromBooleanCut(name='final_cut', instanceToBeCut=
        mdb.models[model_name].rootAssembly.instances[part + '-1'],
                                                                   cuttingInstances=(
                                                                       mdb.models[model_name].rootAssembly.instances[
                                                                           cut_name1 + '-1'],
                                                                       mdb.models[model_name].rootAssembly.instances[
                                                                           cut_name2 + '-1'],),
                                                                   originalInstances=SUPPRESS)

    # Mesh Definition
    mdb.models[model_name].rootAssembly.makeIndependent(
        instances=(mdb.models[model_name].rootAssembly.instances['final_cut-1'],))
    cells = mdb.models[model_name].rootAssembly.instances['final_cut-1'].cells.getByBoundingBox(
        -1000,
        -1000,
        -1000,
        1000,
        1000,
        1000
    )
    region = Region(cells=cells)
    mdb.models[model_name].rootAssembly.seedPartInstance(
        regions=(mdb.models[model_name].rootAssembly.instances['final_cut-1'],), size=lat_mesh_size,
        deviationFactor=0.1)

    mdb.models[model_name].rootAssembly.setMeshControls(
        regions=cells,
        elemShape=TET,
        technique=FREE
    )
    elem_type_1 = mesh.ElemType(elemCode=DC3D20, elemLibrary=STANDARD)
    elem_type_2 = mesh.ElemType(elemCode=DC3D15, elemLibrary=STANDARD)
    elem_type_3 = mesh.ElemType(elemCode=DC3D10, elemLibrary=STANDARD)
    mdb.models[model_name].rootAssembly.setElementType(elemTypes=(elem_type_1,
                                                                  elem_type_2,
                                                                  elem_type_3,), regions=region)

    mdb.models[model_name].rootAssembly.generateMesh(
        regions=(mdb.models[model_name].rootAssembly.instances['final_cut-1'],))

    mdb.models[model_name].rootAssembly.regenerate()

    # (Loads and) BCs
    bot_set = mdb.models[model_name].rootAssembly.instances['final_cut-1'].faces.getByBoundingBox(
        - 1000, - 1000,
        - h_cell / 2 - epsilon,
        1000,
        1000,
        - h_cell / 2 + epsilon)
    top_set = mdb.models[model_name].rootAssembly.instances['final_cut-1'].faces.getByBoundingBox(
        - l_circ_cell / 2 - epsilon, - l_long_cell / 2 - epsilon,
        h_cell / 2 - epsilon,
        l_circ_cell / 2 + epsilon,
        l_long_cell / 2 + epsilon,
        h_cell / 2 + epsilon)

    bot_set_set = mdb.models[model_name].rootAssembly.Set(name='bot_set', faces=bot_set)
    top_set_set = mdb.models[model_name].rootAssembly.Set(name='top_set', faces=top_set)

    mdb.models[model_name].TemperatureBC(name='bot_bc', createStepName='Step-1', region=bot_set_set, magnitude=temp_in,
                                         amplitude=UNSET, distributionType=UNIFORM)
    mdb.models[model_name].TemperatureBC(name='top_bc', createStepName='Step-1', region=top_set_set, magnitude=temp_out,
                                         amplitude=UNSET, distributionType=UNIFORM)

    # Output definition
    bot_surf = mdb.models[model_name].rootAssembly.Surface(name='bot_surf', side1Faces=bot_set)
    top_surf = mdb.models[model_name].rootAssembly.Surface(name='top_surf', side1Faces=top_set)

    bot_int_out = mdb.models[model_name].IntegratedOutputSection(name='bot_int_out', surface=bot_surf)

    mdb.models[model_name].HistoryOutputRequest(name='History-1', createStepName='Step-1', variables=('SOH', 'SOAREA',))
    mdb.models[model_name].historyOutputRequests['History-1'].setValues(
        integratedOutputSection='bot_int_out', rebar=EXCLUDE, sectionPoints=
        DEFAULT)

    # Job definition
    job_name = model_name + '_job'
    mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF,
            explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=OFF,
            memory=90, memoryUnits=PERCENTAGE, model=model_name, modelPrint=OFF,
            multiprocessingMode=DEFAULT, name=job_name, nodalOutputPrecision=SINGLE
            , numCpus=1, numGPUs=0, queue=None, resultsFormat=ODB, scratch='', type=
            ANALYSIS, userSubroutine='', waitHours=0, waitMinutes=0)
    mdb.jobs[job_name].submit(consistencyChecking=OFF)
    mdb.jobs[job_name].waitForCompletion(10)

    # Data Extraction
    odb = openOdb(path=job_name + '.odb')
    soh_val = \
        odb.steps['Step-1'].historyRegions['Surface BOT_SURF'].historyOutputs['SOH  on section BOT_INT_OUT'].data[-1][
            -1]
    volume = mdb.models[model_name].rootAssembly.getMassProperties()['volume']
    print('Mesh Size: ' + str(lat_mesh_size) + '\nSOH:   ' + str(soh_val))

    print("Analysis converged with SOH = " + str(soh_val) + " at mesh size " + str(
        lat_mesh_size) + " mm.")
    print("\n\nSOH: " + str(soh_val) + " mW.\nThermal Conductivity: " + str(
        soh_val * h_cell / (l_long_cell * l_circ_cell * (temp_out - temp_in))) + " W/(mK).\nVolume: " + str(
        volume) + " mm^3.\nRelative density: " + str(volume / l_long_cell / l_circ_cell / h_cell * 100) + "%.")

    # -------------------------------------------------------------------------------------------

    count = 0
    conv_rate_out = 1.
    conv_rate_in = 1.
    max_conv_rate = 0.02
    temp_surf_out_conv = 200.
    temp_surf_in_conv = 34.5

    while conv_rate_out > max_conv_rate or conv_rate_in > max_conv_rate:
        temp_surf_out = temp_surf_out_conv
        temp_surf_in = temp_surf_in_conv

        model_name = 'tank_' + ex.lat.uni_type + '_' + str(int(l_long_cell)) + 'x' + str(int(l_circ_cell)) + 'x' + str(
            int(h_cell)) + '_D' + str(int(round(strut_diam * 100))).zfill(3) + '_' + str(count)

        part_name = model_name + '_sheet'
        therm_cond = soh_val * h_cell / (l_long_cell * l_circ_cell * (temp_out - temp_in))
        solid_mesh_size = 20.

        r_avg = (r_left + r_right) / 2

        rad_emiss = ex.tank.sheets[-1].material.rad_emiss

        material_lat_name = 'ls_p511'
        material_lat_rho = material_rho * volume / (l_long_cell * l_circ_cell * h_cell)
        material_lat_e_mod = material_e_mod * (material_lat_rho / material_rho) ** 2
        material_lat_therm_cond = therm_cond

        # Model
        model = mdb.Model(name=model_name)
        mdb.models[model_name].setValues(absoluteZero=0, stefanBoltzmann=5.67e-11)

        # try things out
        filename_liq = 'C:/Users/i_valais/PycharmProjects/personalproject/input_files/H2_prop_1_13_bar_liquid.csv'

        import csv

        with open(filename_liq, "r") as f:
            text_liq = f.read()

        text_liq = text_liq.replace(",", ".")
        text_liq = text_liq.decode('utf-8')
        text_liq = text_liq.replace("^M", "\n")
        text_liq = text_liq.replace("\r", "\n")

        from io import StringIO

        data_liq = np.genfromtxt(StringIO(text_liq), delimiter=";")

        filename_vap = 'C:/Users/i_valais/PycharmProjects/personalproject/input_files/H2_prop_1_13_bar_vapor.csv'

        import csv

        with open(filename_vap, "r") as f:
            text_vap = f.read()

        text_vap = text_vap.replace(",", ".")
        text_vap = text_vap.decode('utf-8')
        text_vap = text_vap.replace("^M", "\n")
        text_vap = text_vap.replace("\r", "\n")

        from io import StringIO

        data_vap = np.genfromtxt(StringIO(text_vap), delimiter=";")

        # Load CSVs
        liquid = data_liq
        vapor = data_vap

        def get_closest_row(data, pressure):
            idx = np.nanargmin(np.abs(data[:, 1] - pressure))  # pressure = column 1
            return data[idx]

        # Example usage
        phase = 'liquid'

        dataset = liquid if phase == "liquid" else vapor
        row = get_closest_row(dataset, pressure)

        temp_h2 = row[0]  # K
        density_l = row[2]  # kg/m^3
        volume_l = row[3]  # m^3/kg
        int_energy_l = row[4]  # kJ/kg
        enthalpy_l = row[5]  # kJ/kg
        entropy_l = row[6]  # kJ/kg
        cv_l = row[7]  # J/g*K
        cp_l = row[8]  # J/g*K
        sound_spd_l = row[9]  # m/s
        jt_l = row[10]  # K/MPa
        visc_l = row[11]  # Pa*s
        therm_cond_l = row[12]  # W/mK
        surf_tens_l = row[13]  # N/m

        density_l = density_l * 1e-12  # ton/mm3
        int_energy_l = int_energy_l * 1e9  # mJ/ton
        enthalpy_l = enthalpy_l * 1e9  # mJ/ton
        entropy_l = entropy_l * 1e9  # mJ/ton
        cp_l = cp_l * 1e9  # mJ/(ton*K)
        visc_l = visc_l * 1e-6  # MPa*s

        phase = 'vapor'

        dataset = liquid if phase == "liquid" else vapor
        row = get_closest_row(dataset, pressure)

        density_v = row[2]  # kg/m^3
        volume_v = row[3]  # m^3/kg
        int_energy_v = row[4]  # kJ/kg
        enthalpy_v = row[5]  # kJ/kg
        entropy_v = row[6]  # kJ/kg
        cv_v = row[7]  # J/g*K
        cp_v = row[8]  # J/g*K
        sound_spd_v = row[9]  # m/s
        jt_v = row[10]  # K/MPa
        visc_v = row[11]  # Pa*s
        therm_cond_v = row[12]  # W/mK

        density_v = density_v * 1e-12  # ton/mm3
        int_energy_v = int_energy_v * 1e9  # Nmm/ton
        enthalpy_v = enthalpy_v * 1e9  # Nmm/ton
        entropy_v = entropy_v * 1e9  # Nmm/ton
        cp_v = cp_v * 1e9  # mm2/(s2K)
        visc_v = visc_v * 1e-6  # MPa*s

        enthalpy_fg = enthalpy_v - enthalpy_l

        print("Temperature:", temp_h2)
        print("Full row:", row)

        g = 9810  # mm/s^2
        char_length_out = tank_diam
        air_nu = 15.35
        air_prandtl = 0.7148
        air_therm_cond = 2.569e-2
        temp_air = 293.15

        beta_out = 1 / temp_air
        grashof_out = g * char_length_out ** 3 * beta_out * (
                temp_air - temp_surf_out) / air_nu ** 2
        nusselt_out = 0.56 * (air_prandtl ** 2 * grashof_out / (0.864 + air_prandtl)) ** 0.25 + 2
        alpha_fl_out = nusselt_out * air_therm_cond / char_length_out

        # hydrogen alpha_fl
        char_length_in = tank_diam - thickness_tot
        h2_nu = visc_l / density_l
        h2_prandtl = cp_l * visc_l / therm_cond_l

        beta_in = 1 / temp_air
        grashof_in = g * char_length_in ** 3 * beta_in * (temp_surf_in - temp_h2) / h2_nu ** 2
        nusselt_in = 0.56 * (h2_prandtl ** 2 * grashof_in / (0.864 + h2_prandtl)) ** 0.25 + 2
        alpha_fl_in = nusselt_in * therm_cond_l / char_length_in

        # Part
        t_count = 0

        for key, sheet in enumerate(ex.tank.sheets):
            t_count = t_count + sheet.thickness

            mdb.models[model_name].ConstrainedSketch(name='__profile__', sheetSize=1000.)
            mdb.models[model_name].sketches['__profile__'].ConstructionLine(point1=(0.0,
                                                                                    -500.0), point2=(0.0, 500.0))
            mdb.models[model_name].sketches['__profile__'].FixedConstraint(entity=
                                                                           mdb.models[model_name].sketches[
                                                                               '__profile__'].geometry[
                                                                               2])

            mdb.models[model_name].sketches['__profile__'].ArcByCenterEnds(center=(0.0, - l_tot / 2 + r_left),
                                                                           direction=CLOCKWISE,
                                                                           point1=(0.0, - l_tot / 2 + t_count), point2=(
                    -r_left + t_count, - l_tot / 2 + r_left))
            mdb.models[model_name].sketches['__profile__'].ArcByCenterEnds(center=(0.0, - l_tot / 2 + r_left),
                                                                           direction=CLOCKWISE,
                                                                           point1=(
                                                                               0.0,
                                                                               - l_tot / 2 + t_count - sheet.thickness),
                                                                           point2=(
                                                                               -r_left + t_count - sheet.thickness,
                                                                               - l_tot / 2 + r_left))
            mdb.models[model_name].sketches['__profile__'].ArcByCenterEnds(center=(0.0, + l_tot / 2 - r_right),
                                                                           direction=CLOCKWISE,
                                                                           point1=(
                                                                               -r_right + t_count,
                                                                               + l_tot / 2 - r_right),
                                                                           point2=(0.0, + l_tot / 2 - t_count)
                                                                           )
            mdb.models[model_name].sketches['__profile__'].ArcByCenterEnds(center=(0.0, + l_tot / 2 - r_right),
                                                                           direction=CLOCKWISE,
                                                                           point1=(
                                                                               -r_right + t_count - sheet.thickness,
                                                                               + l_tot / 2 - r_right),
                                                                           point2=(
                                                                               0.0,
                                                                               + l_tot / 2 - t_count + sheet.thickness)
                                                                           )

            mdb.models[model_name].sketches['__profile__'].Line(point1=(-r_left + t_count, - l_tot / 2 + r_left),
                                                                point2=(-r_right + t_count, + l_tot / 2 - r_right))
            mdb.models[model_name].sketches['__profile__'].Line(
                point1=(-r_left + t_count - sheet.thickness, - l_tot / 2 + r_left),
                point2=(-r_right + t_count - sheet.thickness, + l_tot / 2 - r_right))
            mdb.models[model_name].sketches['__profile__'].Line(point1=(0., - l_tot / 2 + t_count - sheet.thickness),
                                                                point2=(0., - l_tot / 2 + t_count))
            mdb.models[model_name].sketches['__profile__'].Line(point1=(0., l_tot / 2 - t_count + sheet.thickness),
                                                                point2=(0., l_tot / 2 - t_count))

            mdb.models[model_name].Part(dimensionality=THREE_D, name=part_name + str(key),
                                        type=DEFORMABLE_BODY)
            mdb.models[model_name].parts[part_name + str(key)].BaseSolidRevolve(angle=360.0,
                                                                                flipRevolveDirection=OFF, sketch=
                                                                                mdb.models[model_name].sketches[
                                                                                    '__profile__'])

        # Material
        mdb.models[model_name].Material(name=material_name)
        mdb.models[model_name].materials[material_name].Density(table=((material_rho,),))
        mdb.models[model_name].materials[material_name].Elastic(table=((material_e_mod, material_nu),))
        mdb.models[model_name].materials[material_name].Conductivity(table=((material_cond,),))

        mdb.models[model_name].Material(name=material_lat_name)
        mdb.models[model_name].materials[material_lat_name].Density(table=((material_lat_rho,),))
        mdb.models[model_name].materials[material_lat_name].Elastic(table=((material_lat_e_mod, material_nu),))
        mdb.models[model_name].materials[material_lat_name].Conductivity(table=((material_lat_therm_cond,),))

        # Section Definition
        mdb.models[model_name].HomogeneousSolidSection(material=material_name, name=
        'sheet_sec', thickness=None)
        mdb.models[model_name].HomogeneousSolidSection(material=material_lat_name, name=
        'lat_sec', thickness=None)

        # Section Assignment
        for key, sheet in enumerate(ex.tank.sheets):

            if sheet.lattice is None:
                sec_name = 'sheet_sec'
            else:
                sec_name = 'lat_sec'

            mdb.models[model_name].parts[part_name + str(key)].Set(cells=
            mdb.models[model_name].parts[part_name + str(key)].cells.getByBoundingBox(
                - l_tot - 10.,
                - l_tot - 10.,
                - l_tot - 10.,
                l_tot + 10.,
                l_tot + 10.,
                l_tot + 10.),
                name='Set-1')
            mdb.models[model_name].parts[part_name + str(key)].SectionAssignment(offset=0.0,
                                                                                 offsetField='',
                                                                                 offsetType=MIDDLE_SURFACE,
                                                                                 region=
                                                                                 mdb.models[model_name].parts[
                                                                                     part_name + str(key)].sets[
                                                                                     'Set-1'],
                                                                                 sectionName=
                                                                                 sec_name,
                                                                                 thicknessAssignment=FROM_SECTION)

        # Step
        mdb.models[model_name].HeatTransferStep(name='Step-1', previous='Initial', response=STEADY_STATE, timePeriod=1.,
                                                initialInc=1., minInc=1e-5,
                                                maxInc=1.)

        # Assembly
        mdb.models[model_name].rootAssembly.DatumCsysByDefault(CARTESIAN)
        for key, sheet in enumerate(ex.tank.sheets):
            mdb.models[model_name].rootAssembly.Instance(dependent=ON, name=part_name + str(key) + '-1',
                                                         part=mdb.models[model_name].parts[part_name + str(key)])
        mass = mdb.models[model_name].rootAssembly.getMassProperties()['mass']

        # Mesh
        for key, sheet in enumerate(ex.tank.sheets):
            mdb.models[model_name].rootAssembly.makeIndependent(
                instances=(mdb.models[model_name].rootAssembly.instances[part_name + str(key) + '-1'],))

            cells = mdb.models[model_name].rootAssembly.instances[part_name + str(key) + '-1'].cells.getByBoundingBox(
                - l_tot - 10,
                - l_tot - 10,
                - l_tot - 10,
                l_tot + 10,
                l_tot + 10,
                l_tot + 10
            )

            mdb.models[model_name].rootAssembly.setMeshControls(
                regions=cells,
                elemShape=HEX_DOMINATED,
                technique=SWEEP)

            elem_type_1 = mesh.ElemType(elemCode=DC3D20, elemLibrary=STANDARD)
            elem_type_2 = mesh.ElemType(elemCode=DC3D15, elemLibrary=STANDARD)
            elem_type_3 = mesh.ElemType(elemCode=DC3D10, elemLibrary=STANDARD)
            mdb.models[model_name].rootAssembly.setElementType(elemTypes=(elem_type_1,
                                                                          elem_type_2,
                                                                          elem_type_3,),
                                                               regions=(cells,))

            mdb.models[model_name].rootAssembly.seedPartInstance(
                regions=(mdb.models[model_name].rootAssembly.instances[part_name + str(key) + '-1'],),
                size=solid_mesh_size,
                deviationFactor=0.1)

            mdb.models[model_name].rootAssembly.generateMesh(
                regions=(mdb.models[model_name].rootAssembly.instances[part_name + str(key) + '-1'],))

            mdb.models[model_name].rootAssembly.regenerate()

        # Interactions
        mdb.models[model_name].CavityRadiationProp(name='rad_emiss_prop', property=((rad_emiss,),))

        face_top = list(np.zeros(len(ex.tank.sheets)))
        surface_top = list(np.zeros(len(ex.tank.sheets)))
        face_bot = list(np.zeros(len(ex.tank.sheets)))
        surface_bot = list(np.zeros(len(ex.tank.sheets)))
        t_count = 0
        for key, sheet in enumerate(ex.tank.sheets):
            t_count = t_count + sheet.thickness

            face_top[key] = mdb.models[model_name].rootAssembly.instances[part_name + str(key) + '-1'].faces.findAt(
                ((0., l_tot / 2 - t_count + sheet.thickness, 0.),), ((0., -l_tot / 2 + t_count - sheet.thickness, 0.),),
                ((0., 0., r_avg - t_count + sheet.thickness),))
            face_bot[key] = mdb.models[model_name].rootAssembly.instances[part_name + str(key) + '-1'].faces.findAt(
                ((0., l_tot / 2 - t_count, 0.),), ((0., -l_tot / 2 + t_count, 0.),), ((0., 0., r_avg - t_count),))

            surface_top[key] = mdb.models[model_name].rootAssembly.Surface(name='surf_' + str(key) + '_top',
                                                                           side1Faces=face_top[key])
            surface_bot[key] = mdb.models[model_name].rootAssembly.Surface(name='surf_' + str(key) + '_bot',
                                                                           side1Faces=face_bot[key])

            if key is 0:
                mdb.models[model_name].FilmCondition(name='conv_out', createStepName='Step-1', surface=surface_top[key],
                                                     definition=EMBEDDED_COEFF, sinkTemperature=temp_air,
                                                     filmCoeff=alpha_fl_out)
            if key is len(ex.tank.sheets) - 1:
                mdb.models[model_name].FilmCondition(name='conv_in', createStepName='Step-1', surface=surface_bot[key],
                                                     definition=EMBEDDED_COEFF, sinkTemperature=temp_h2,
                                                     filmCoeff=alpha_fl_in)
            if key is not 0:
                mdb.models[model_name].Tie(name='tie_constraint_' + str(key), master=surface_bot[key - 1],
                                           slave=surface_top[key],
                                           constraintEnforcement=SURFACE_TO_SURFACE)
        if rad_check is True:
            for key, sheet in enumerate(ex.tank.sheets):
                if sheet.lattice is not None:
                    surf_rad = mdb.models[model_name].rootAssembly.Surface(name='surf_rad_' + str(key),
                                                                           side1Faces=(
                                                                               face_top[key + 1], face_bot[key - 1]))

                    mdb.models[model_name].CavityRadiation(name='cavity_radiation_' + str(key), createStepName='Step-1',
                                                           surfaces=(
                                                               mdb.models[model_name].rootAssembly.surfaces[
                                                                   'surf_rad_' + str(key)],),
                                                           surfaceEmissivities=('rad_emiss_prop',), )

        # Output
        surf_in_int_out = mdb.models[model_name].IntegratedOutputSection(name='surf_in_int_out',
                                                                         surface=surface_bot[-1])

        mdb.models[model_name].HistoryOutputRequest(name='History-1', createStepName='Step-1',
                                                    variables=('SOH', 'SOAREA',))
        mdb.models[model_name].historyOutputRequests['History-1'].setValues(
            integratedOutputSection='surf_in_int_out', rebar=EXCLUDE, sectionPoints=
            DEFAULT)

        # Job
        job_name = model_name + '_job'
        mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF,
                explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=OFF,
                memory=90, memoryUnits=PERCENTAGE, model=model_name, modelPrint=OFF,
                multiprocessingMode=DEFAULT, name=job_name, nodalOutputPrecision=SINGLE
                , numCpus=1, numGPUs=0, queue=None, resultsFormat=ODB, scratch='', type=
                ANALYSIS, userSubroutine='', waitHours=0, waitMinutes=0)
        mdb.jobs[job_name].submit(consistencyChecking=OFF)
        mdb.jobs[job_name].waitForCompletion(10)

        # Data Extraction
        odb = openOdb(path=job_name + '.odb')
        soh_val_tank = \
            odb.steps['Step-1'].historyRegions['Surface SURF_' + str(len(ex.tank.sheets) - 1) + '_BOT'].historyOutputs[
                'SOH  on section SURF_IN_INT_OUT'].data[-1][-1]

        temp = odb.steps['Step-1'].frames[-1].fieldOutputs['NT11'].values
        temp_list = []

        for i in range(len(temp)):
            temp_list.append(temp[i].data)

        temp_surf_out_conv = round(max(temp_list), 2)
        temp_surf_in_conv = round(min(temp_list), 2)

        conv_rate_out = abs(temp_surf_out - temp_surf_out_conv) / temp_surf_out
        conv_rate_in = abs(temp_surf_in - temp_surf_in_conv) / temp_surf_in

        count += 1

        print('Heat Transfered: ' + str(soh_val_tank / 1000) + " W.")
        print('Boil-off rate:   ' + str(soh_val_tank / enthalpy_fg * 1e6) + ' g/s.')
        print(temp_surf_out_conv)
        print(temp_surf_in_conv)

    vol_tank = ex.tank.calc_vol()

    uc_therm_cond = round(soh_val * h_cell / (l_long_cell * l_circ_cell * (temp_out - temp_in)), 5)
    q_dot = round(soh_val_tank / 1000, 2)
    boil_off = round(soh_val_tank / enthalpy_fg * 1e6, 3)
    temp_surf_out = temp_surf_out_conv
    temp_surf_in = temp_surf_in_conv
    vol_tank = round(vol_tank * 1e-9, 5)
    mass_lh2 = round(vol_tank * density_l * 1e12, 5)
    bo_rate_per_day = round(boil_off / (mass_lh2 * 1e3) * 86400 * 100)

    print("UC Thermal Conductivity (W/mK): " + str(uc_therm_cond))
    print("Tank Heat Transfer (W): " + str(q_dot))
    print("Boil-off Rate (g/s): " + str(boil_off))
    print("Temperature of Outside Surface (K): " + str(temp_surf_out))
    print("Temperature of Inside Surface (K): " + str(temp_surf_in))
    print("Tank Volume (m^3): " + str(vol_tank))
    print("LH2 Mass (kg): " + str(mass_lh2))
    print("Boil-off Rate (%/day): " + str(bo_rate_per_day))

    x = mass

    return x


def objective_gradient(x):
    """
    Gradient of the objective function (optional).
    """
    return 2 * x + 3


def constraint_function(x):
    """
    Constraint function for constrained optimization.
    Example: inequality constraint c(x) >= 0
    """
    return x - 100


def run_minimization():
    """
    Example minimization using SciPy.
    """
    x0 = 0.0  # initial guess

    cons = {
        "type": "ineq",
        "fun": constraint_function
    }

    result = optimize.minimize(
        objective,
        x0,
        method="SLSQP",
        constraints=[cons],
        options={"disp": True, 'maxiter':5},
    )

    print("\n--- Optimization Result ---")
    print(result)
    return result


if __name__ == "__main__":
    # Choose what to run
    run_minimization()
