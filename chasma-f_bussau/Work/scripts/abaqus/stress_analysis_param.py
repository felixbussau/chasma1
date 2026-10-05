# -*- coding: utf-8 -*-
"""
Created on 13.11.2025
 
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
param_list = list(np.append(diam0, diam_lin))
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

    part_name = model_name

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
    model = mdb.Model(name=model_name)

    # Part Definition
    mdb.models[model_name].ConstrainedSketch(name='__profile__', sheetSize=1000.)
    mdb.models[model_name].sketches['__profile__'].ConstructionLine(point1=(0.0,
                                                                            -500.0), point2=(0.0, 500.0))
    mdb.models[model_name].sketches['__profile__'].FixedConstraint(entity=
                                                                   mdb.models[model_name].sketches[
                                                                       '__profile__'].geometry[
                                                                       2])
    mdb.models[model_name].sketches['__profile__'].ArcByCenterEnds(center=(0.0, - l_tot / 2 + r_left),
                                                                   direction=CLOCKWISE,
                                                                   point1=(0.0, - l_tot / 2 + thickness_tot / 2),
                                                                   point2=(
                                                                       -r_left + thickness_tot / 2,
                                                                       - l_tot / 2 + r_left))
    mdb.models[model_name].sketches['__profile__'].ArcByCenterEnds(center=(0.0, + l_tot / 2 - r_left),
                                                                   direction=CLOCKWISE,
                                                                   point1=(
                                                                       -r_left + thickness_tot / 2,
                                                                       + l_tot / 2 - r_left),
                                                                   point2=(0.0, + l_tot / 2 - thickness_tot / 2)
                                                                   )
    mdb.models[model_name].sketches['__profile__'].Line(point1=(-r_left + thickness_tot / 2, - l_tot / 2 + r_left),
                                                        point2=(-r_left + thickness_tot / 2, + l_tot / 2 - r_left))
    mdb.models[model_name].Part(dimensionality=THREE_D, name=part_name,
                                type=DEFORMABLE_BODY)
    mdb.models[model_name].parts[part_name].BaseShellRevolve(angle=360.0,
                                                             flipRevolveDirection=OFF, sketch=
                                                             mdb.models[model_name].sketches['__profile__'])

    # Material Definition
    material_list = []
    for i, key in enumerate(ex.tank.sheets):
        if key.lattice is None:
            if key.material.name in material_list:
                continue
            else:
                mdb.models[model_name].Material(name=key.material.name)
                mdb.models[model_name].materials[key.material.name].Density(table=((key.material.rho,),))
                mdb.models[model_name].materials[key.material.name].Elastic(
                    table=((key.material.e_mod, key.material.nu),))
                material_list.append(key.material.name)
        else:
            if 'ls_' + key.material.name in material_list:
                continue
            else:
                mdb.models[model_name].Material(name='ls_' + key.material.name)
                mdb.models[model_name].materials['ls_' + key.material.name].Density(
                    table=((ex.lat.material.rho * ex.lat.calc_unit_cell_density(),),))
                mdb.models[model_name].materials['ls_' + key.material.name].Elastic(
                    table=((ex.lat.calc_e_mod_1(), ex.lat.material.nu),))
                material_list.append('ls_' + key.material.name)

    # Composite Creation
    mdb.models[model_name].parts[part_name].CompositeLayup(description='',
                                                           elementType=SHELL, name=part_name, offsetType=MIDDLE_SURFACE,
                                                           symmetric=False, thicknessAssignment=FROM_SECTION)
    mdb.models[model_name].parts[part_name].compositeLayups[part_name].Section(
        integrationRule=SIMPSON, poissonDefinition=DEFAULT, preIntegrate=OFF,
        temperature=GRADIENT, thicknessType=UNIFORM, useDensity=OFF)
    mdb.models[model_name].parts[part_name].compositeLayups[part_name].ReferenceOrientation(
        additionalRotationType=ROTATION_NONE, angle=0.0, axis=AXIS_3, fieldName='',
        localCsys=None, orientationType=GLOBAL)

    composite_face = regionToolset.Region(
        faces=mdb.models[model_name].parts[part_name].faces.getByBoundingBox(-epsilon - l_tot / 2,
                                                                             -l_tot / 2 + thickness_tot / 2 - epsilon,
                                                                             -epsilon - l_tot / 2, l_tot / 2 + epsilon,
                                                                             l_tot / 2 + thickness_tot / 2 + epsilon,
                                                                             l_tot / 2 + epsilon))

    for i, key in enumerate(ex.tank.sheets):
        if key.lattice is None:
            material_name = key.material.name
        else:
            material_name = 'ls_' + key.material.name

        thickness = key.thickness

        mdb.models[model_name].parts[part_name].compositeLayups[part_name].CompositePly(
            additionalRotationField='', additionalRotationType=ROTATION_NONE, angle=0.0
            , axis=AXIS_3, material=material_name, numIntPoints=3, orientationType=
            SPECIFY_ORIENT, orientationValue=0.0, plyName='Ply-' + str(i + 1), region=composite_face, suppressed=False,
            thickness=thickness, thicknessType=
            SPECIFY_THICKNESS)

    # Step Definition
    mdb.models[model_name].StaticStep(name='Step-1', previous='Initial')
    mdb.models[model_name].steps['Step-1'].setValues(initialInc=0.25, maxNumInc=20,
                                                     noStop=OFF, timeIncrementationMethod=FIXED)

    # Field Output
    mdb.models[model_name].FieldOutputRequest(name='F-Output-2', objectToCopy=
    mdb.models[model_name].fieldOutputRequests['F-Output-1'], toStepName=
                                              'Step-1')
    mdb.models[model_name].fieldOutputRequests['F-Output-2'].setValues(
        layupLocationMethod=SPECIFIED, layupNames=(part_name + '-1.' + part_name,),
        outputAtPlyBottom=True, outputAtPlyMid=True, outputAtPlyTop=False, rebar=
        EXCLUDE, variables=('ALPHA', 'ALPHAN', 'BF', 'CENTMAG', 'CENTRIFMAG',
                            'CORIOMAG', 'CS11', 'CTSHR', 'E', 'EE', 'EEQUT', 'ER', 'ESF1', 'GRAV',
                            'HP', 'IE', 'LE', 'MISES', 'MISESMAX', 'MISESONLY', 'NE', 'NFORC',
                            'NFORCSO', 'P', 'PE', 'PEEQ', 'PEEQMAX', 'PEEQT', 'PEMAG', 'PEQC',
                            'PRESSONLY', 'PS', 'RBANG', 'RBFOR', 'RBROT', 'ROTAMAG', 'S', 'SALPHA',
                            'SE', 'SEE', 'SEP', 'SEPE', 'SEQUT', 'SF', 'SPE', 'SSAVG', 'TE', 'TEEQ',
                            'TEVOL', 'THE', 'TRIAX', 'TRNOR', 'TRSHR', 'TSHR', 'VE', 'VEEQ', 'VS',
                            'YIELDPOT'))

    # Assembly Definition
    mdb.models[model_name].rootAssembly.DatumCsysByDefault(CARTESIAN)
    mdb.models[model_name].rootAssembly.Instance(dependent=OFF, name=part_name + '-1',
                                                 part=mdb.models[model_name].parts[part_name])

    # Mesh Generation
    mdb.models[model_name].rootAssembly.seedPartInstance(deviationFactor=0.1,
                                                         minSizeFactor=0.1, regions=(
            mdb.models[model_name].rootAssembly.instances[part_name + '-1'],), size=20)
    mdb.models[model_name].rootAssembly.generateMesh(regions=(
        mdb.models[model_name].rootAssembly.instances[part_name + '-1'],))

    # Boundary Condition Definition
    bc_left = mdb.models[model_name].rootAssembly.instances[part_name + '-1'].vertices.getByBoundingBox(-epsilon,
                                                                                                        -l_tot / 2 + thickness_tot / 2 - epsilon,
                                                                                                        -epsilon,
                                                                                                        epsilon,
                                                                                                        -l_tot / 2 + thickness_tot / 2 + epsilon,
                                                                                                        epsilon)
    mdb.models[model_name].rootAssembly.Set(name='fix_bc', vertices=bc_left)

    mdb.models[model_name].DisplacementBC(amplitude=UNSET, createStepName='Initial',
                                          distributionType=UNIFORM, fieldName='',
                                          localCsys=None,
                                          name='BC-1',
                                          region=mdb.models[model_name].rootAssembly.sets['fix_bc'], u1=SET,
                                          u2=SET,
                                          u3=SET, ur1=SET, ur2=SET, ur3=SET)

    bc_right = mdb.models[model_name].rootAssembly.instances[part_name + '-1'].vertices.getByBoundingBox(-epsilon,
                                                                                                         l_tot / 2 - thickness_tot / 2 - epsilon,
                                                                                                         -epsilon,
                                                                                                         epsilon,
                                                                                                         l_tot / 2 - thickness_tot / 2 + epsilon,
                                                                                                         epsilon)
    mdb.models[model_name].rootAssembly.Set(name='roll_bc', vertices=bc_right)

    mdb.models[model_name].DisplacementBC(amplitude=UNSET, createStepName='Initial',
                                          distributionType=UNIFORM, fieldName='',
                                          localCsys=None,
                                          name='BC-2',
                                          region=mdb.models[model_name].rootAssembly.sets['roll_bc'], u1=UNSET,
                                          u2=UNSET,
                                          u3=SET, ur1=SET, ur2=SET, ur3=UNSET)

    # Load and Coupling Definition
    inner_face = mdb.models[model_name].rootAssembly.instances[part_name + '-1'].faces.getByBoundingBox(
        -l_tot - epsilon,
        -l_tot - epsilon,
        -l_tot - epsilon,
        l_tot + epsilon,
        l_tot + epsilon,
        l_tot + epsilon)
    mdb.models[model_name].rootAssembly.Surface(name='all_in_surf', side1Faces=inner_face)

    mdb.models[model_name].Pressure(name='Inner Pressure', createStepName='Step-1',
                                    region=mdb.models[model_name].rootAssembly.surfaces['all_in_surf'],
                                    magnitude=-pressure)

    # JOB
    job_name = model_name
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
    section_point_plyin_mid = odb.steps['Step-1'].frames[-1].fieldOutputs['E'].locations[0].sectionPoints[-4]
    strain_plyin = odb.steps['Step-1'].frames[-1].fieldOutputs['E'].getSubset(
        sectionPoint=section_point_plyin_mid).values
    e11_list = []
    e22_list = []
    e12_list = []
    for i in range(len(strain_plyin)):
        e11_plyin = strain_plyin[i].data[0]
        e22_plyin = strain_plyin[i].data[1]
        e12_plyin = strain_plyin[i].data[3]

        e11_list.append(e11_plyin)
        e22_list.append(e22_plyin)
        e12_list.append(e12_plyin)

    max_e11_plyin = max(e11_list)
    max_e22_plyin = max(e22_list)
    max_e12_plyin = max(e12_list)

    print('Max e11 strain (circ direction) in the middle of the inner lattice is ' + str(max_e11_plyin) + '.')
    print('Max e22 strain (long direction) in the middle of the inner lattice is ' + str(max_e22_plyin) + '.')
    print('Max e12 strain (shear direction) in the middle of the inner lattice is ' + str(max_e12_plyin) + '.')
    mass = round(mdb.models[model_name].rootAssembly.getMassProperties()['mass'], 6)
    print('Mass of the tank is ' + str(mass * 1000) + ' kg.')

    ########################################################################################################################
    '''
    UNIT CELL
    '''
    ########################################################################################################################


    if param_type == 'strut_diameter':
        model_name = 'BCC_UC' + str(int(round(l_long_cell))) + 'x' + str(int(round(l_circ_cell))) + 'x' + str(
            int(round(h_cell))) + '_D' + str(int(round(strut_diam * 100))).zfill(3)

    elif param_type == 'height':
        model_name = 'BCC_UC' + str(int(round(l_long_cell))) + 'x' + str(int(round(l_circ_cell))) + 'x' + str(
            int(round(h_cell))) + '_D' + str(int(round(strut_diam * 100))).zfill(3)

    elif param_type == 'pressure':
        model_name = 'BCC_UC' + str(int(round(l_long_cell))) + 'x' + str(int(round(l_circ_cell))) + 'x' + str(
            int(round(h_cell))) + '_D' + str(int(round(strut_diam * 100))).zfill(3)
        model_name = model_name + '_P' + str(int(round(pressure * 10))).zfill(2)

    elif param_type == 'l_long':
        model_name = 'BCC_UC' + str(int(round(l_long_cell))) + 'x' + str(int(round(l_circ_cell))) + 'x' + str(
            int(round(h_cell))) + '_D' + str(int(round(strut_diam * 100))).zfill(3)

    elif param_type == 'l_circ':
        model_name = 'BCC_UC' + str(int(round(l_long_cell))) + 'x' + str(int(round(l_circ_cell))) + 'x' + str(
            int(round(h_cell))) + '_D' + str(int(round(strut_diam * 100))).zfill(3)

    elif param_type == 'tank_diameter':
        model_name = 'BCC_UC' + str(int(round(l_long_cell))) + 'x' + str(int(round(l_circ_cell))) + 'x' + str(
            int(round(h_cell))) + '_D' + str(int(round(strut_diam * 100))).zfill(3)
        model_name = model_name + '_tank_diam' + str(int(round(param)))

    elif param_type == 'tank_length':
        model_name = 'BCC_UC' + str(int(round(l_long_cell))) + 'x' + str(int(round(l_circ_cell))) + 'x' + str(
            int(round(h_cell))) + '_D' + str(int(round(strut_diam * 100))).zfill(3)
        model_name = model_name + '_tank_length' + str(int(round(param)))

    epsilon = 1e-3
    if ex.tank.sheets[-2].lattice.uni_type == 'bcc':

        # Model Definition
        model = mdb.Model(name=model_name)
        mdb_model = mdb.models[model_name]

        # Part Definition
        mdb.models[model_name].ConstrainedSketch(name='__profile__', sheetSize=200.)
        mdb.models[model_name].sketches['__profile__'].Line(point1=(0.0, 0.0),
                                                            point2=(x_init_point, y_init_point))
        mdb.models[model_name].Part(dimensionality=THREE_D, name='strut_1',
                                    type=DEFORMABLE_BODY)
        mdb.models[model_name].parts['strut_1'].BaseWire(sketch=
                                                         mdb.models[model_name].sketches['__profile__'])

        # Material Definition
        material_name = ex.tank.sheets[-2].material.name
        material_rho = ex.tank.sheets[-2].material.rho
        material_e_mod = ex.tank.sheets[-2].material.e_mod
        material_nu = ex.tank.sheets[-2].material.nu
        mdb.models[model_name].Material(name=material_name)
        mdb.models[model_name].materials[material_name].Density(table=((material_rho,),))
        mdb.models[model_name].materials[material_name].Elastic(table=((material_e_mod, material_nu),))

        # Section and Profile Definition
        # mdb.models[model_name].HomogeneousShellSection(idealization=NO_IDEALIZATION,
        #                                                integrationRule=SIMPSON, material='P511', name='shell_sec',
        #                                                nodalThicknessField='', numIntPts=5, poissonDefinition=DEFAULT,
        #                                                preIntegrate=OFF, temperature=GRADIENT, thickness=t_sheet, thicknessField='',
        #                                                thicknessModulus=None, thicknessType=UNIFORM, useDensity=OFF)

        mdb.models[model_name].CircularProfile(name='strut_prof', r=strut_radius)

        mdb.models[model_name].BeamSection(consistentMassMatrix=False, integration=
        DURING_ANALYSIS, material=material_name, name='strut_sec', poissonRatio=0.0,
                                           profile='strut_prof', temperatureVar=LINEAR)

        # SECTION ASSIGNMENT
        mdb.models[model_name].parts['strut_1'].Set(
            edges=mdb.models[model_name].parts['strut_1'].edges, name='Set-1')
        mdb.models[model_name].parts['strut_1'].SectionAssignment(offset=0.0,
                                                                  offsetField='', offsetType=MIDDLE_SURFACE, region=
                                                                  mdb.models[model_name].parts['strut_1'].sets['Set-1'],
                                                                  sectionName=
                                                                  'strut_sec', thicknessAssignment=FROM_SECTION)

        # Step Definition
        mdb.models[model_name].StaticStep(name='Step-1', previous='Initial')
        mdb.models[model_name].steps['Step-1'].setValues(initialInc=0.25, maxNumInc=20,
                                                         noStop=OFF, timeIncrementationMethod=FIXED)

        # Assembly Definition (parametrise for the angles of rotation)
        mdb.models[model_name].rootAssembly.DatumCsysByDefault(CARTESIAN)
        mdb.models[model_name].rootAssembly.Instance(dependent=ON, name='strut_1-1',
                                                     part=mdb.models[model_name].parts['strut_1'])
        mdb.models[model_name].rootAssembly.Instance(dependent=ON, name='strut_1-2',
                                                     part=mdb.models[model_name].parts['strut_1'])
        mdb.models[model_name].rootAssembly.Instance(dependent=ON, name='strut_1-3',
                                                     part=mdb.models[model_name].parts['strut_1'])
        mdb.models[model_name].rootAssembly.Instance(dependent=ON, name='strut_1-4',
                                                     part=mdb.models[model_name].parts['strut_1'])

        mdb.models[model_name].rootAssembly.rotate(angle=-angle_xy_z, axisDirection=(-y_init_point, x_init_point,
                                                                                     0.0), axisPoint=(0.0, 0.0, 0.0),
                                                   instanceList=('strut_1-1',))
        mdb.models[model_name].rootAssembly.rotate(angle=-angle_xy_z, axisDirection=(-y_init_point, x_init_point,
                                                                                     0.0), axisPoint=(0.0, 0.0, 0.0),
                                                   instanceList=('strut_1-2',))
        mdb.models[model_name].rootAssembly.rotate(angle=-angle_xy_z, axisDirection=(-y_init_point, x_init_point,
                                                                                     0.0), axisPoint=(0.0, 0.0, 0.0),
                                                   instanceList=('strut_1-3',))
        mdb.models[model_name].rootAssembly.rotate(angle=-angle_xy_z, axisDirection=(-y_init_point, x_init_point,
                                                                                     0.0), axisPoint=(0.0, 0.0, 0.0),
                                                   instanceList=('strut_1-4',))

        mdb.models[model_name].rootAssembly.rotate(angle=90, axisDirection=(0.0, 0.0,
                                                                            1.0), axisPoint=(0.0, 0.0, 0.0),
                                                   instanceList=('strut_1-2',))
        mdb.models[model_name].rootAssembly.translate(instanceList=(
            'strut_1-2',), vector=(l_long_cell, 0.0, 0.0))

        mdb.models[model_name].rootAssembly.rotate(angle=180, axisDirection=(0.0, 0.0,
                                                                             1.0), axisPoint=(0.0, 0.0, 0.0),
                                                   instanceList=('strut_1-3',))
        mdb.models[model_name].rootAssembly.translate(instanceList=(
            'strut_1-3',), vector=(l_long_cell, l_circ_cell, 0.0))

        mdb.models[model_name].rootAssembly.rotate(angle=270, axisDirection=(0.0, 0.0,
                                                                             1.0), axisPoint=(0.0, 0.0, 0.0),
                                                   instanceList=('strut_1-4',))
        mdb.models[model_name].rootAssembly.translate(instanceList=(
            'strut_1-4',), vector=(0.0, l_circ_cell, 0.0))

        mdb.models[model_name].rootAssembly.InstanceFromBooleanMerge(domain=GEOMETRY,
                                                                     instances=(
                                                                         mdb.models[model_name].rootAssembly.instances[
                                                                             'strut_1-1'],
                                                                         mdb.models[model_name].rootAssembly.instances[
                                                                             'strut_1-2'],
                                                                         mdb.models[model_name].rootAssembly.instances[
                                                                             'strut_1-3'],
                                                                         mdb.models[model_name].rootAssembly.instances[
                                                                             'strut_1-4']),
                                                                     name='unit_cell_inst', originalInstances=SUPPRESS)
        assembly = mdb_model.rootAssembly
        part = mdb.models[model_name].parts['unit_cell_inst']  # Be sure this is the correct part name!

        assembly.Instance(name='unit_cell_inst-1', part=part, dependent=OFF)
        instance = assembly.instances['unit_cell_inst-1']

        # Mesh Generation
        assembly.seedPartInstance(regions=(instance,), size=beam_mesh_size, deviationFactor=0.1)
        assembly.generateMesh(regions=(instance,))
        assembly.regenerate()

        # Beam Orientation
        mdb.models[model_name].parts['unit_cell_inst'].assignBeamSectionOrientation(method=
                                                                                    N1_COSINES, n1=(0.0, 0.0, -1.0),
                                                                                    region=
                                                                                    mdb.models[model_name].parts[
                                                                                        'unit_cell_inst'].sets['Set-1'])
    elif ex.tank.sheets[-2].lattice.uni_type == 'octet':
        model_name = 'Unit Cell Octet Truss'

        # Model Definition
        model = mdb.Model(name=model_name)
        mdb_model = mdb.models[model_name]
        part_name = 'unit_cell_inst'

        L = l_circ_cell
        B = l_long_cell
        H = h_cell
        R = strut_radius

        # Material Definition
        material_name = ex.tank.sheets[-2].material.name
        material_rho = ex.tank.sheets[-2].material.rho
        material_e_mod = ex.tank.sheets[-2].material.e_mod
        material_nu = ex.tank.sheets[-2].material.nu
        mdb.models[model_name].Material(name=material_name)
        mdb.models[model_name].materials[material_name].Density(table=((material_rho,),))
        mdb.models[model_name].materials[material_name].Elastic(table=((material_e_mod, material_nu),))

        # Step Definition
        mdb.models[model_name].StaticStep(name='Step-1', previous='Initial')
        mdb.models[model_name].steps['Step-1'].setValues(initialInc=0.25, maxNumInc=20,
                                                         noStop=OFF, timeIncrementationMethod=FIXED)

        # Part Definition
        myPart = model.Part(name=part_name, dimensionality=THREE_D, type=DEFORMABLE_BODY)

        # Define named points
        points = {
            # TopPlane
            'A1': (L, 0.0, H),
            'A2': (0.0, 0.0, H),
            'A3': (L / 2, B / 2, H),
            'A4': (L, B, H),
            'A5': (0.0, B, H),
            # MidPlane
            'B1': (L / 2, 0.0, H / 2),
            'B2': (L, B / 2, H / 2),
            'B3': (0.0, B / 2, H / 2),
            'B4': (L / 2, B, H / 2),
            # BottomPlane
            'C1': (L, 0.0, 0.0),
            'C2': (0.0, 0.0, 0.0),
            'C3': (L / 2, B / 2, 0.0),
            'C4': (L, B, 0.0),
            'C5': (0.0, B, 0.0),

        }

        # Create datum points and store their references
        datum_dict = {}
        for name, coord in points.items():
            datum = myPart.DatumPointByCoordinate(coords=coord)
            datum_dict[name] = datum.id

        # Define connections as pairs of point names
        connections = [
            # A plane connections
            ('A1', 'B2'),
            ('A2', 'B3'),
            ('A4', 'B2'),
            ('A5', 'B3'),

            # B plane points
            ('B1', 'A3'),
            ('B1', 'B2'),
            ('B1', 'B3'),
            ('B1', 'C3'),
            ('B4', 'A3'),
            ('B4', 'B2'),
            ('B4', 'B3'),
            ('B4', 'C3'),

            # B plane and xz plane
            ('B2', 'A3'),
            ('B2', 'C3'),
            ('B3', 'A3'),
            ('B3', 'C3'),

            # C plane connections
            ('C1', 'B2'),
            ('C2', 'B3'),
            ('C4', 'B2'),
            ('C5', 'B3'),
            ('C1', 'B1'),
            ('C2', 'B1'),
            ('C5', 'B4'),
            ('C4', 'B4'),
            ('A1', 'B1'),
            ('A2', 'B1'),
            ('A5', 'B4'),
            ('A4', 'B4'),
            # selectively connect points
        ]

        # Create wires (edges) between specified pairs
        for p1_name, p2_name in connections:
            id1 = datum_dict[p1_name]
            id2 = datum_dict[p2_name]
            vert1 = myPart.datums[id1]
            vert2 = myPart.datums[id2]
            myPart.WirePolyLine(points=((vert1, vert2),), mergeType=IMPRINT, meshable=True)

        print("Finished creating points and wires.")

        # Profile creation
        mdb.models[model_name].CircularProfile(name='Circle', r=R)

        # Section creation
        mdb.models[model_name].BeamSection(name='strut_sec',
                                           integration=DURING_ANALYSIS, poissonRatio=0.0, profile='Circle',
                                           material=material_name, temperatureVar=LINEAR, consistentMassMatrix=False)

        # #Section Assignment
        p = mdb.models['Unit Cell Octet Truss'].parts['unit_cell_inst']
        e = p.edges
        mdb.models[model_name].parts['unit_cell_inst'].Set(edges=e, name='Set-1')
        mdb.models[model_name].parts['unit_cell_inst'].SectionAssignment(offset=0.0,
                                                                         offsetField='', offsetType=MIDDLE_SURFACE,
                                                                         region=
                                                                         mdb.models[model_name].parts[
                                                                             'unit_cell_inst'].sets['Set-1'],
                                                                         sectionName=
                                                                         'strut_sec', thicknessAssignment=FROM_SECTION)

        # Beam orientations
        # Semicircular orientation (fix edges)
        mdb.models[model_name].parts['unit_cell_inst'].assignBeamSectionOrientation(method=
                                                                                    N1_COSINES, n1=(0.0, 0.0, -1.0),
                                                                                    region=
                                                                                    mdb.models[model_name].parts[
                                                                                        'unit_cell_inst'].sets['Set-1'])

        mdb.models[model_name].rootAssembly.Instance(dependent=ON, name='unit_cell_inst-1',
                                                     part=mdb.models[model_name].parts['unit_cell_inst'])

        assembly = mdb.models[model_name].rootAssembly
        part = mdb.models[model_name].parts['unit_cell_inst']  # Be sure this is the correct part name!

        assembly.Instance(name='unit_cell_inst-1', part=part, dependent=OFF)
        instance = assembly.instances['unit_cell_inst-1']

        # Mesh Generation
        assembly.seedPartInstance(regions=(instance,), size=beam_mesh_size, deviationFactor=0.1)
        assembly.generateMesh(regions=(instance,))
        assembly.regenerate()

    else:
        raise NotImplementedError

    # Strain values
    exx = max_e11_plyin
    eyy = max_e22_plyin
    ezz = - pressure / ex.lat.calc_e_mod_3() - ex.lat.material.nu * (max_e11_plyin + max_e22_plyin)

    # RP definition
    rp = mdb.models[model_name].rootAssembly.ReferencePoint(point=(l_long_cell / 2, l_circ_cell / 2, h_cell / 2))
    ref_nodes = mdb.models[model_name].rootAssembly.referencePoints
    rp_id = rp.id
    rpn = (ref_nodes[rp_id],)
    rp_name = 'RP'

    mdb.models[model_name].rootAssembly.Set(referencePoints=(
        mdb.models[model_name].rootAssembly.referencePoints[rp_id],), name='RP_Set')

    # Example node array — modify this with your own nodes or from Abaqus part
    instance_nodes = mdb.models[model_name].rootAssembly.instances['unit_cell_inst-1'].nodes
    nodes = np.array([node.coordinates for node in instance_nodes])

    # Face dictionaries: map face to list of (index, sort_key)
    face_defs = {
        'x0': [],
        'xL': [],
        'y0': [],
        'yL': [],
        'z0': [],
        'zL': [],
    }

    # Assign each node to the appropriate faces
    for idx, (x, y, z) in enumerate(nodes):
        if abs(x - 0.0) < epsilon:
            face_defs['x0'].append((idx, (round(y, 6), round(z, 6))))
        if abs(x - l_circ_cell) < epsilon:
            face_defs['xL'].append((idx, (round(y, 6), round(z, 6))))
        if abs(y - 0.0) < epsilon:
            face_defs['y0'].append((idx, (round(x, 6), round(z, 6))))
        if abs(y - l_long_cell) < epsilon:
            face_defs['yL'].append((idx, (round(x, 6), round(z, 6))))
        if abs(z - 0.0) < epsilon:
            face_defs['z0'].append((idx, (round(x, 6), round(y, 6))))
        if abs(z - h_cell) < epsilon:
            face_defs['zL'].append((idx, (round(x, 6), round(y, 6))))

    # Assign labels per face
    node_labels = {}  # node index → list of labels

    for face, node_list in face_defs.items():
        # Sort by orthogonal coordinates for consistent pairing
        sorted_nodes = sorted(node_list, key=lambda item: item[1])
        for i, (idx, _) in enumerate(sorted_nodes):
            label = "{}_{:03d}".format(face, i + 1)
            if idx not in node_labels:
                node_labels[idx] = []
            node_labels[idx].append(label)

    # Print results
    named_nodes = {}

    for idx in sorted(node_labels):
        coords = list(nodes[idx])
        labels = node_labels[idx]

        # print("Node {} at {} -> {}".format(idx, list(nodes[idx]), node_labels[idx]))

        for label in labels:
            named_nodes[label] = \
            mdb.models[model_name].rootAssembly.instances['unit_cell_inst-1'].nodes.getByBoundingBox(
                list(nodes[idx])[0] - epsilon, list(nodes[idx])[1] - epsilon, list(nodes[idx])[2] - epsilon,
                list(nodes[idx])[0] + epsilon, list(nodes[idx])[1] + epsilon, list(nodes[idx])[2] + epsilon)[0].label

    print(named_nodes)

    # 1. Define periodic BC equations
    face_pairs = []

    # You already have this dictionary
    # named_nodes = { 'x0_001': 17, 'xL_001': 23, ... }

    face_pairs_info = {
        'x': ('x0', 'xL', (1, 0, 0)),
        'y': ('y0', 'yL', (0, 1, 0)),
        'z': ('z0', 'zL', (0, 0, 1)),
    }

    for direction, (face0, faceL, normal) in face_pairs_info.items():
        i = 1
        while True:
            label0 = "{}_{:03d}".format(face0, i)
            labelL = "{}_{:03d}".format(faceL, i)

            if label0 not in named_nodes or labelL not in named_nodes:
                break  # no more pairs in this direction

            node0 = named_nodes[label0]
            nodeL = named_nodes[labelL]

            face_pairs.append((direction, str(node0), str(nodeL), normal))
            i += 1

    strain_map = {'x': exx, 'y': eyy, 'z': ezz}
    direction_map = {'x': 1, 'y': 2, 'z': 3}

    eq_counter = 1
    for face, nodeA_key, nodeB_key, normal in face_pairs:
        strain = strain_map[face]
        dof = direction_map[face]

        assembly = mdb.models[model_name].rootAssembly
        instance = assembly.instances['unit_cell_inst-1']

        # Define sets AT THE ASSEMBLY LEVEL using instance nodes
        assembly.Set(name='Node' + nodeA_key, nodes=instance.nodes.sequenceFromLabels((int(nodeA_key),)))
        assembly.Set(name='Node' + nodeB_key, nodes=instance.nodes.sequenceFromLabels((int(nodeB_key),)))

        mdb_model.Equation(name='Eq-%02d' % eq_counter,
                           terms=((-1.0, 'Node' + nodeA_key, dof),
                                  (1.0, 'Node' + nodeB_key, dof),
                                  (-1.0, 'RP_Set', dof)))
        eq_counter += 1

    # 2. Apply displacements to RP
    mdb_model.DisplacementBC(name='MacroStrain',
                             createStepName='Step-1',
                             region=assembly.sets['RP_Set'],
                             u1=exx * l_circ_cell,
                             u2=eyy * l_long_cell,
                             u3=ezz * h_cell)

    # 3. Fix node 000 (0,0,0)
    assembly.Set(name='Node0',
                 nodes=instance.nodes.getByBoundingBox(-epsilon, -epsilon, -epsilon, epsilon, epsilon, epsilon))
    mdb_model.DisplacementBC(name='FixOrigin',
                             createStepName='Step-1',
                             region=assembly.sets['Node0'],
                             u1=0.0, u2=0.0, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0)

    # Job submit
    job_name = model_name
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
    mises_all = odb.steps['Step-1'].frames[-1].fieldOutputs['S'].values
    s_mises_list = []

    for i in range(len(mises_all)):
        s_mises_all = mises_all[i].mises
        s_mises_list.append(s_mises_all)

    max_mises_all = round(max(s_mises_list), 3)

    max_mises_param[index] = max_mises_all
    mass_param[index] = mass * 1000

print('Max Mises (MPa): ' + str(max_mises_param))
print('Mass (kg): ' + str(mass_param))