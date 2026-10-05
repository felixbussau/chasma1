# -*- coding: utf-8 -*-
"""
Created on 24.11.2025
 
@author: i_valais
"""
import subprocess
import numpy as np
from scipy.optimize import minimize
import time
import os


# ------------------------------------------------------
# 1. Objective function
# ------------------------------------------------------
def objective(params):
    """
    Objective function that:
        - writes parameters to a file
        - runs Abaqus
        - reads results
        - returns objective value
    """

    thickness = params[0]
    radius = params[1]

    print("\nRunning iteration with params:", params)

    # ---------------------------------------
    # (A) Write parameters to file (Python 2.7 compatible)
    # ---------------------------------------
    f = open("params.txt", "w")
    f.write(str(thickness) + " " + str(radius) + "\n")
    f.close()

    # You can also update the INP directly here.

    # ---------------------------------------
    # (B) Run Abaqus
    # ---------------------------------------
    abaqus_cmd = "abaqus job=my_job input=my_model.inp interactive"

    print("Running Abaqus...")
    p = subprocess.Popen(abaqus_cmd, shell=True)
    p.wait()   # wait for Abaqus to finish

    time.sleep(1)

    # ---------------------------------------
    # (C) Read the results
    # ---------------------------------------
    result_value = read_result_from_file("results.txt")

    print("Result read = ", result_value)

    return result_value


# ------------------------------------------------------
# 2. Read result file
# ------------------------------------------------------
def read_result_from_file(filename):
    """
    Reads result value written by your Abaqus script.
    """
    f = open(filename, "r")
    content = f.read().strip()
    f.close()
    return float(content)


# ------------------------------------------------------
# 3. Optimization loop
# ------------------------------------------------------
def run_optimization():
    x0 = np.array([5.0, 10.0])  # initial guess

    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",  # works well with noisy output
        options={"maxiter": 20, "disp": True}
    )

    print("\nOptimization finished:")
    print(result)


if __name__ == "__main__":
    run_optimization()
