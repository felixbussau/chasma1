# -*- coding: utf-8 -*-
"""
Created on 12.05.2025
 
@author: i_valais
"""

import numpy as np

from . import material as mat
from . import lattice_struct as lat

class Sheet:
    """
    Class to define the sheet types present to the tank. If the
    sheet is lattice, the material and thickness of the Lattice
    object will be overwritten by those of the Sheet object.

    Args:
        name (str): Name of the sheet.
        material (obj): Material object of the sheet.
        thickness (float or int): thickness of the sheet (in mm).
        lattice (None or obj): if None, the sheet consists of bulk
                material, if not None, a Lattice object is needed.
        n_cpl (int): Number of unit cells per lattice structure layer

    TODO:
        - Add the possibility to add more than one lattice layers on one sheet. (check if it works)
    """

    def __init__(self, name=None, material=None, thickness=None, lattice=None, n_cpl= 1):
        """
        Function to initialise class Sheet.
        """
        self.name = name
        self.material = material
        self.thickness = thickness
        self.lattice = lattice
        self.n_cpl = n_cpl

        if self.lattice is not None:
            self.uni_type = self.lattice.uni_type
            self.l_long_cell = self.lattice.l_long_cell
            self.l_circ_cell = self.lattice.l_circ_cell
            self.d_rod = self.lattice.d_rod

            self.lattice.h_cell = self.thickness
            self.lattice.material = self.material

            # if self.lattice is not None:
            #     self.check_knudsen(temp=30, press=0.1e-6)

    def check_knudsen(self, temp, press, d=3.7e-7):
        """
        Checks if the conduction in the lattice structure layers needs also to be calculated.

        Returns:

        """
        k_boltzmann = 1.38e-20
        knudsen_num = k_boltzmann * temp / (np.sqrt(2) * np.pi * d ** 2 * press * self.thickness)

        if knudsen_num < 10:
            raise Warning("The conduction of " + str(self.name) + " is not negligible.")


if __name__ == '__main__':

    p511 = mat.Material(name='p511', rho=7.8e-9, e_mod=204000, nu=0.3, s_yield=380, s_ult=690, e_break=0.5,
                        therm_cond=15)

    # Lattice Definition
    bcc = lat.UnitCell(name='bcc unit cell', uni_type='bcc', material=p511, l_long_cell=10, l_circ_cell=10, h_cell=10, d_rod=3)

    # Sheet Definition
    lat_layer = Sheet(name='lattice_layer', material=p511, thickness=1, lattice=bcc)



