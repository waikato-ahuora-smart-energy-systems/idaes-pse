#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################
"""General forms of Thermal Conductivity expressions for Helmholtz EoS functions"""

__author__ = "Stephen Burroughs"

import os
import math
import pyomo.environ as pyo
from idaes.core.util.math import smooth_max
from idaes.core.util.constants import Constants

def lambda_0_type01(model, parameters):
    """Type01 expression for the dilute gas thermal conductivity

    Args:
        model (Block): Pyomo model
        parameters (dict): Main parameters dictionary
    Returns:
        Expression for dilute gas thermal conductivity   """
    a = parameters["a"]

    return 1000 *sum( a[i] * (1/model.tau) **(i) for i in range(len(a)) ) #mW/m/K

def lambda_r_type01(model, parameters):
    """Type01 expression for the residual thermal conductivity

    Args:
        model (Block): Pyomo model
        parameters (dict): Main parameters dictionary
    Returns:
        Expression for residual thermal conductivity   """
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    return 1000* sum( #mW/m/K
        (b1[i] + b2[i] * (1/model.tau)) * (model.delta**(i+1))
        for i in range(len(b1))
    )

def lambda_c_type01(model, parameters):
    """Type01 expression for the emperical critical enhancement of thermal conductivity

    Args:
        model (Block): Pyomo model
        parameters (dict): Main parameters dictionary
    Returns:
        Expression for critical enhancement of thermal conductivity   """
    c = parameters["c"]
    return (c[0] 
            /(c[1] + abs(1/model.tau - 1))
            * pyo.exp((-c[2] * (model.delta -1))**2)) * 1000 #mW/m/K


def lambda_c_type02(model, parameters):
    """Type02 expression for the simplified crossover model of critical enhancement of thermal conductivity

    Args:
        model (Block): Pyomo model
        comp (str): Component name
        parameters (dict): Main parameters dictionary
    Returns:
        Expression for critical enhancement of thermal conductivity   """
    model.cp = pyo.ExternalFunction(library="", function="cp")
    model.cv = pyo.ExternalFunction(library="", function="cv")
    model.mu = pyo.ExternalFunction(library="", function="mu")
    model.itc = pyo.ExternalFunction(library="", function="itc")
    comp = model.name
    MW = model.MW / 1000 #kg/mol
    rho_star = model.rho_star / MW #mol/m^3
    rho = model.delta * rho_star
    T = model.T_star / model.tau
    k = Constants.boltzmann_constant
    qd = 1/parameters["qd"]
    xi_0 = parameters["xi_0"]
    gamma = parameters["gamma"]
    big_gamma = parameters["big_gamma"]
    v = parameters["v"]
    t_ref = parameters["t_ref"]
    R = parameters["R"]
    cp = model.cp(comp, model.delta, model.tau)
    cv = model.cv(comp, model.delta, model.tau)
    mu = model.mu(comp, model.delta, model.tau) /1e6 #convert from microPa-s to Pa-s

    drho_dp = model.itc(comp, model.delta, model.tau) * model.delta * rho_star
    drho_dp_ref = model.itc(comp, model.delta, model.T_star/t_ref) * model.delta * rho_star
    deltchi = smooth_max(model.Pc/1000 * rho / (big_gamma * rho_star ** 2) * (drho_dp - drho_dp_ref * t_ref/T), 0, 1e-8)

    xi = (
        xi_0 * deltchi ** (v/gamma)
    )

    Omega = (
        2 / math.pi * (
            (cp - cv) / cp
            * pyo.atan(qd * xi)
            + cv / cp * qd * xi
        )
    )
    Omega_0 = (
        2 / math.pi * (
            1 - pyo.exp(
                - 1 /((qd * xi) ** -1 + ((qd * xi * 1 / model.delta)**2) /3)
            )
        )
    )

    return (1000*model.MW*
        (rho * cp *R * k * T) / 
        (6 * math.pi * mu * xi) *
        (Omega - Omega_0)
        )
