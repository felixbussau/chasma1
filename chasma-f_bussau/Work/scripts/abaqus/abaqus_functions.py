# -*- coding: utf-8 -*-
"""
Created on 25.11.2025
 
@author: i_valais
"""
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

import sys

sys.path.append(r"C:\Users\i_valais\PycharmProjects\chasma")
import Work.scripts.tank_example.tank_example as ex

import numpy as np


class AbaqusFunctions:
    """
    This class is used to implement different Abaqus Routines that require more lines than the usual, in order to be easier to change them.
    """

    def __init__(self, model):
        """
        Function to initialise the class AbaqusFunctions.

        :param model (obj): Abaqus model in the for of mdb.models[model_name]
        """
        self.model = model

    def create_uc_wire_part(self, part_name, material_name, uc_type, l_circ_cell, l_long_cell, h_cell, d_strut):
        """
        Function that creates a unit cell part with wires, depending on the geometric characteristics and the unit cell type the user defines.

        :param part_name (str): Name of the part that is going to be created.
        :param material_name (str): Name of the material that is going to be created.
        :param uc_type (str): Unit cell type, depending on the list included in the library (bcc)
        :param l_circ_cell (float): Dimension of the unit cell in the circumferential direction, in mm.
        :param l_long_cell (float): Dimension of the unit cell in the longitudinal direction, in mm.
        :param h_cell (float): Dimension of the unit cell in the radial direction, in mm.
        :param d_strut (float): Diameter of the strut, in mm.
        :return:
        """
        self.model.Part(dimensionality=THREE_D, name=part_name, type=DEFORMABLE_BODY)

        if uc_type == 'bcc':
            # definition of nodes independent from unit cell dimensions
            n1 = (0, 0, 0)
            n2 = (0, 1, 0)
            n3 = (0, 1, 1)
            n4 = (0, 0, 1)
            n5 = (.5, .5, .5)
            n6 = (1, 0, 0)
            n7 = (1, 1, 0)
            n8 = (1, 1, 1)
            n9 = (1, 0, 1)

            nodes = [n1, n2, n3, n4, n5, n6, n7, n8, n9]

        elif uc_type == 'octet':
            pass

        else:
            raise NotImplementedError("The suggested unit cell type is not part of the existing library.")

        dimensions = (l_circ_cell, l_long_cell, h_cell)
        points = []
        strut_radius = d_strut / 2
        self.model.CircularProfile(name='strut_prof', r=strut_radius)

        self.model.BeamSection(consistentMassMatrix=False, integration=
        DURING_ANALYSIS, material=material_name, name='strut_sec', poissonRatio=0.0,
                               profile='strut_prof', temperatureVar=LINEAR)

        for key in nodes:
            points.append(tuple(np.array(key) * np.array(dimensions)))

        if uc_type == 'bcc':
            # definition of the connections between the nodes of the unit cell
            connections = [(points[0], points[4]), (points[1], points[4]), (points[2], points[4]),
                           (points[3], points[4]), (points[4], points[5]), (points[4], points[6]),
                           (points[4], points[7]), (points[4], points[8])]

        elif uc_type == 'octet':
            pass

        for i, key in enumerate(connections):
            self.model.Part(name='line_' + str(i), dimensionality=THREE_D, type=DEFORMABLE_BODY)
            p = self.model.parts['line_' + str(i)]
            p.WirePolyLine(
                points=(key[0], key[1]),
                meshable=True)

            p.Set(edges=p.edges, name='Set-' + str(i + 1))
            p.SectionAssignment(offset=0.0,
                                offsetField='', offsetType=MIDDLE_SURFACE, region=
                                p.sets['Set-' + str(i + 1)],
                                sectionName=
                                'strut_sec', thicknessAssignment=FROM_SECTION)

            self.model.rootAssembly.DatumCsysByDefault(CARTESIAN)
            self.model.rootAssembly.Instance(dependent=ON, name='line_' + str(i) + '-1',
                                             part=self.model.parts['line_' + str(i)])

        # for i, key in enumerate(self.model.parts):
        #     self.model.rootAssembly.Instance(dependent=ON, name='line_' + str(i) + '-1',
        #                                      part=self.model.parts['line_' + str(i)])

        instance_names = ['line_' + str(i) + '-1' for i in range(len(connections))]
        instance_tuple = tuple(self.model.rootAssembly.instances[name] for name in instance_names)
        self.model.rootAssembly.InstanceFromBooleanMerge(domain=GEOMETRY,
                                                         instances=instance_tuple,
                                                         name=part_name,
                                                         originalInstances=SUPPRESS)

        for i in range(len(connections)):
            del self.model.parts['line_' + str(i)]

    def create_periodic_bcs(self, part_name, l_circ_cell, l_long_cell, h_cell, exx, eyy, ezz):
        """
        Function to define the Periodic Boundary Conditions on a specific rectangular cuboid (all 6 surfaces are the one parallel to the other in 3 pairs).

        :param part_name (str): Name of the part to apply the periodic boundary conditions on.
        :param l_circ_cell (float): Dimension of the unit cell in the circumferential direction, in mm.
        :param l_long_cell (float): Dimension of the unit cell in the longitudinal direction, in mm.
        :param h_cell (float): Dimension of the unit cell in the radial direction, in mm.
        :param exx (float): strains of the tank at the lattice structure layer in the circumferential direction.
        :param eyy (float): strains of the tank at the lattice structure layer in the longitudinal direction.
        :param ezz (float): strains of the tank at the lattice structure layer in the radial direction.
        :return:
        """

        epsilon = 1e-3

        # RP definition
        rp = self.model.rootAssembly.ReferencePoint(point=(l_circ_cell / 2, l_long_cell / 2, h_cell / 2))
        ref_nodes = self.model.rootAssembly.referencePoints
        rp_id = rp.id
        rpn = (ref_nodes[rp_id],)
        rp_name = 'RP'

        self.model.rootAssembly.Set(referencePoints=(
            self.model.rootAssembly.referencePoints[rp_id],), name='RP_Set')

        # Example node array — modify this with your own nodes or from Abaqus part
        instance_nodes = self.model.rootAssembly.instances[part_name + '-1'].nodes
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
                    self.model.rootAssembly.instances[part_name + '-1'].nodes.getByBoundingBox(
                        list(nodes[idx])[0] - epsilon, list(nodes[idx])[1] - epsilon, list(nodes[idx])[2] - epsilon,
                        list(nodes[idx])[0] + epsilon, list(nodes[idx])[1] + epsilon, list(nodes[idx])[2] + epsilon)[
                        0].label

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

        assembly = self.model.rootAssembly
        instance = self.model.rootAssembly.instances[part_name + '-1']

        eq_counter = 1
        for face, nodeA_key, nodeB_key, normal in face_pairs:
            strain = strain_map[face]
            dof = direction_map[face]

            # Define sets AT THE ASSEMBLY LEVEL using instance nodes
            self.model.rootAssembly.Set(name='Node' + nodeA_key, nodes=self.model.rootAssembly.instances[
                part_name + '-1'].nodes.sequenceFromLabels((int(nodeA_key),)))
            self.model.rootAssembly.Set(name='Node' + nodeB_key, nodes=self.model.rootAssembly.instances[
                part_name + '-1'].nodes.sequenceFromLabels((int(nodeB_key),)))

            self.model.Equation(name='Eq-%02d' % eq_counter,
                                terms=((-1.0, 'Node' + nodeA_key, dof),
                                       (1.0, 'Node' + nodeB_key, dof),
                                       (-1.0, 'RP_Set', dof)))
            eq_counter += 1

        # 2. Apply displacements to RP
        self.model.DisplacementBC(name='MacroStrain',
                                  createStepName='Step-1',
                                  region=self.model.rootAssembly.sets['RP_Set'],
                                  u1=exx * l_circ_cell,
                                  u2=eyy * l_long_cell,
                                  u3=ezz * h_cell)

        # 3. Fix node 000 (0,0,0)
        self.model.rootAssembly.Set(name='Node0',
                                    nodes=self.model.rootAssembly.instances[part_name + '-1'].nodes.getByBoundingBox(
                                        -epsilon, -epsilon, -epsilon, epsilon, epsilon, epsilon))
        self.model.DisplacementBC(name='FixOrigin',
                                  createStepName='Step-1',
                                  region=self.model.rootAssembly.sets['Node0'],
                                  u1=0.0, u2=0.0, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0)

    def create_multiple_ucs(self, initial_part_name, l_circ_cell, l_long_cell, n_circ, n_long):
        """
        Function to create multiple unit cells in the form of a plate.

        :param initial_part_name (str): Name of the part to be multiplied.
        :param l_circ_cell (float): length of the unit cell in the circumferential direction
            (will be used for the spacing in that direction, so it is only useful if the unit
            cells are connected to each other).
        :param l_long_cell (float): length of the unit cell in the longitudinal direction
            (will be used for the spacing in that direction, so it is only useful if the unit
            cells are connected to each other).
        :param n_circ (int): number of unit cells required in the circumferential direction.
        :param n_long (int): number of unit cells required in the longitudinal direction.
        :return:
        """

        self.model.rootAssembly.LinearInstancePattern(instanceList=[initial_part_name + '-1'], number1=n_circ,
                                                      number2=n_long, spacing1=l_circ_cell, spacing2=l_long_cell)

        part_name = initial_part_name + '_plate_' + str(n_circ) + 'x' + str(n_long)

        instance_names = []
        for i in range(n_circ):
            for j in range(n_long):
                if i == 0 and j == 0:
                    instance_names.append(initial_part_name + '-1')
                else:
                    instance_names.append(initial_part_name + '-1-lin-' + str(i + 1) + '-' + str(j + 1))

        instance_tuple = tuple(self.model.rootAssembly.instances[name] for name in instance_names)

        self.model.rootAssembly.InstanceFromBooleanMerge(domain=GEOMETRY,
                                                         instances=instance_tuple,
                                                         name=part_name,
                                                         originalInstances=SUPPRESS)
        del self.model.parts[initial_part_name]

