#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2024 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################
"""Predefined expression for Helmholtz EoS functions
"""
__author__ = "Stephen Burroughs"

import pyomo.environ as pyo


def phi_ideal_expressions_type05(model, parameters):
    """Type05 expression for the ideal part of dimensionless Helmholtz free energy for use in helmholz air

    Args:
        model (Block): Pyomo model
        parameters (dict): Main parameters dictionary

    Returns:
        dict: Expressions for ideal part of Helmholtz free energy
    """
    n0 = parameters["eos"]["n0"]
    g0 = parameters["eos"]["g0"]
    Tc = parameters["basic"]["Tc"]
    return {
        "phii": pyo.log(model.delta) # correct
        + sum(n0[i] * model.tau ** (i-4)
              for i in range (1,6))
        + n0[6] * model.tau ** 1.5 #correct
        + n0[7] * pyo.log(model.tau)#correct
        + n0[8] * pyo.log(1 - pyo.exp(-g0[8] * model.tau))#Ben Says correct look at the FluidLibrary.h file for IdealGasHelmholtzPlanckEinstein c = 1 d=-1 g sign fliped
        + n0[9] * pyo.log(1 - pyo.exp(-g0[9] * model.tau))#Ben Says correct look at the FluidLibrary.h file for IdealGasHelmholtzPlanckEinstein c = 1 d=-1 g sign fliped
        + n0[10] * pyo.log(2 / 3 + pyo.exp(g0[10] * model.tau)),#correct
        "phii_d": 1 / model.delta,
        "phii_dd": -1 / model.delta ** 2,
        "phii_t": sum (((i-4) * n0[i] * model.tau ** (i-5))
                       for i in range(1,6))
                       + 1.5 * n0[6] * model.tau ** 0.5
                       + n0[7]/model.tau
                       + (n0[8] * g0[8])
                       / (pyo.exp(g0[8]*model.tau)-1)
                       + (n0[9] * g0[9])
                       / (pyo.exp(g0[9]*model.tau)-1)
                       + (n0[10] * g0[10])
                       / (2/3 * pyo.exp(-g0[10]*model.tau)+ 1),
        "phii_tt": sum((i-5) * (i-4)
                       * n0[i] * model.tau
                       ** (i-6)
                       for i in range(1,6))
                       + 0.75 * n0[6] 
                       * model.tau ** -0.5
                       - n0[7] / model.tau **2
                       - n0[8] * g0[8] ** 2
                       * pyo.exp(g0[8] * model.tau)
                       / (pyo.exp(g0[8] * model.tau) - 1) ** 2
                       - n0[9] * g0[9] ** 2
                       * pyo.exp(g0[9] * model.tau)
                       / (pyo.exp(g0[9] * model.tau) - 1) ** 2
                       + 2/3 * n0[10] * g0[10] ** 2
                       * pyo.exp(-g0[10] * model.tau)
                       /(2/3 * pyo.exp(-g0[10] * model.tau) + 1) ** 2,
        "phii_dt": 0
    }
