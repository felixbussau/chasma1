# -*- coding: utf-8 -*-
"""
Created on 04.03.2026
 
@author: i_valais
"""
import numpy as np



class Strut:
    """
    Class that implements a cylindrical straight strut.
    """
    def __init__(self, p1, p2, d_rod):
        """
        Function to initialise the class Strut.

        Args:
        p1(np.array): Coordinates of the first node of the strut. (mm)
        p2(np.array): Coordinates of the first node of the strut. (mm)
        d_rod(float): Diameter of the strut. (mm)
        """
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.d_rod = d_rod

        self.length = None
        self.n_vec = None
        self.stress = None

    def compute_geometry(self):
        """
        Function to calculate some basic properties of the strut, like the
        cross-sectional area, the second moment of inertia, and the torsional
        moment of inertia.
        """
        d_vec = self.p2 - self.p1
        self.length = np.linalg.norm(d_vec)
        self.n_vec = d_vec / self.length

        z_vec = (0, 0, 1)
        n_vec_norm = np.linalg.norm(self.n_vec)
        z_vec_norm = np.linalg.norm(z_vec)
        self.alpha = np.arccos(np.dot(z_vec, self.n_vec) / (n_vec_norm * z_vec_norm))
        self.cs_area = np.pi * self.d_rod ** 2 / 4
        self.second_moment_of_inertia = np.pi * self.d_rod ** 4 / 64
        self.torsional_constant = np.pi * self.d_rod ** 4 / 32