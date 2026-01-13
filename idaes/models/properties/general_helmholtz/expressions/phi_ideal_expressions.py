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
"""Predefined ideal expressions for Helmholtz EoS functions
"""

__author__ = "Stephen Burroughs"

import pyomo.environ as pyo


def phi_ideal_expressions_lead(model, parameters):
    """lead expression for the ideal part of dimensionless Helmholtz free energy

    Args:
        model (Block): Pyomo model
        parameters (dict): Main parameters dictionary

    Returns:
        dict: Expressions for lead part of ideal Helmholtz free energy
    """
    n0 = parameters["eos"]["n0"]
    return {
        "phii": pyo.log(model.delta)
        + n0[1]
        + n0[2] * model.tau,
        "phii_d": 1.0 / model.delta,
        "phii_dd": -1.0 / model.delta**2,
        "phii_t": n0[2],
        "phii_tt": 0,
        "phii_dt": 0,
    }

def phi_ideal_expressions_logtau(model, parameters):
    """logtau expression for the ideal part of dimensionless Helmholtz free energy

    Args:
        model (Block): Pyomo model
        parameters (dict): Main parameters dictionary

    Returns:
        dict: Expressions for logtau part of ideal Helmholtz free energy
    """
    n0 = parameters["eos"]["n0"]
    return {
        "phii": n0[3] * pyo.log(model.tau),
        "phii_d": 0,
        "phii_dd": 0,
        "phii_t": n0[3] / model.tau,
        "phii_tt": -n0[3] / model.tau**2,
        "phii_dt": 0,
    }

def phi_ideal_expressions_planck_einstein1(model, parameters):
    """Type01 expression for the first Planck Einstein part of dimensionless ideal Helmholtz free energy

    Args:
        model (Block): Pyomo model
        parameters (dict): Main parameters dictionary

    Returns:
        dict: Expressions for first Planck Einstein part of ideal Helmholtz free energy
    """
    last_term = parameters["eos"]["last_term_ideal"]
    n0 = parameters["eos"]["n0"]
    g0 = parameters["eos"]["g0"]
    rng = range(4, last_term + 1)##Need to evaluate range
    return {
        "phii": sum(n0[i] * pyo.log(1 - pyo.exp(-g0[i] * model.tau)) for i in rng),
        "phii_d": 0,
        "phii_dd": 0,
        "phii_t": sum(n0[i] * g0[i] / (pyo.exp(g0[i] * model.tau) - 1) for i in rng),
        "phii_tt": -sum(n0[i] * g0[i] ** 2 * pyo.exp(-g0[i] * model.tau) / (1 - pyo.exp(-g0[i] * model.tau)) ** 2 for i in rng),
        "phii_dt": 0,
    }

def phi_ideal_expressions_planck_einstein2(model, parameters):
    """Second Planck Einstein expression for the ideal part of dimensionless Helmholtz free energy

    Args:
        model (Block): Pyomo model
        parameters (dict): Main parameters dictionary

    Returns:
        dict: Expressions for second Planck Einstein part of ideal Helmholtz free energy
    """
    last_term = parameters["eos"]["last_term_ideal"]
    n0 = parameters["eos"]["n0"]
    g0 = parameters["eos"]["g0"]
    rng = range(4, last_term + 1)
    return {
        "phii": sum(n0[i] * pyo.log(1 - pyo.exp(-g0[i] * model.tau)) for i in rng),
        "phii_d": 1.0 / model.delta,
        "phii_dd": -1.0 / model.delta**2,
        "phii_t": n0[2]
        + n0[3] / model.tau
        + sum(n0[i] * g0[i] / (pyo.exp(g0[i] * model.tau) - 1) for i in rng),
        "phii_tt": -n0[3] / model.tau**2
        - sum(
            n0[i]
            * g0[i] ** 2
            * pyo.exp(-g0[i] * model.tau)
            / (1 - pyo.exp(-g0[i] * model.tau)) ** 2
            for i in rng
        ),
        "phii_dt": 0,
    }

def phi_ideal_expressions_cp_constant(model, parameters):
    """Type01 expression for the cp constant part of ideal dimensionless Helmholtz free energy

    Args:
        model (Block): Pyomo model
        parameters (dict): Main parameters dictionary

    Returns:
        dict: Expressions for cp constant part of ideal Helmholtz free energy
    """
    last_term = parameters["eos"]["last_term_ideal"]
    n0 = parameters["eos"]["n0"]
    g0 = parameters["eos"]["g0"]
    rng = range(4, last_term + 1)
    return {
        # "phii": pyo.log(model.delta)
        # + n0[1]
        # + n0[2] * model.tau
        # + n0[3] * pyo.log(model.tau)
        # + sum(n0[i] * pyo.log(1 - pyo.exp(-g0[i] * model.tau)) for i in rng),
        # "phii_d": 1.0 / model.delta,
        # "phii_dd": -1.0 / model.delta**2,
        # "phii_t": n0[2]
        # + n0[3] / model.tau
        # + sum(n0[i] * g0[i] / (pyo.exp(g0[i] * model.tau) - 1) for i in rng),
        # "phii_tt": -n0[3] / model.tau**2
        # - sum(
        #     n0[i]
        #     * g0[i] ** 2
        #     * pyo.exp(-g0[i] * model.tau)
        #     / (1 - pyo.exp(-g0[i] * model.tau)) ** 2
        #     for i in rng
        # ),
        # "phii_dt": 0,
    }

def phi_ideal_expressions_power(model, parameters):
    """Type01 expression for the Power part of dimensionless ideal Helmholtz free energy

    Args:
        model (Block): Pyomo model
        parameters (dict): Main parameters dictionary

    Returns:
        dict: Expressions for Power part of ideal Helmholtz free energy
    """
    last_term = parameters["eos"]["last_term_ideal"]
    n0 = parameters["eos"]["n0"]
    g0 = parameters["eos"]["g0"]
    rng = range(4, last_term + 1)
    return {
        # "phii": pyo.log(model.delta)
        # + n0[1]
        # + n0[2] * model.tau
        # + n0[3] * pyo.log(model.tau)
        # + sum(n0[i] * pyo.log(1 - pyo.exp(-g0[i] * model.tau)) for i in rng),
        # "phii_d": 1.0 / model.delta,
        # "phii_dd": -1.0 / model.delta**2,
        # "phii_t": n0[2]
        # + n0[3] / model.tau
        # + sum(n0[i] * g0[i] / (pyo.exp(g0[i] * model.tau) - 1) for i in rng),
        # "phii_tt": -n0[3] / model.tau**2
        # - sum(
        #     n0[i]
        #     * g0[i] ** 2
        #     * pyo.exp(-g0[i] * model.tau)
        #     / (1 - pyo.exp(-g0[i] * model.tau)) ** 2
        #     for i in rng
        # ),
        # "phii_dt": 0,
    }

def phi_ideal_expressions_AlyLee(model, parameters):
    """Type01 expression for the AlyLee part of dimensionless ideal Helmholtz free energy

    Args:
        model (Block): Pyomo model
        parameters (dict): Main parameters dictionary

    Returns:
        dict: Expressions for AlyLee part of ideal Helmholtz free energy
    """
    last_term = parameters["eos"]["last_term_ideal"]
    n0 = parameters["eos"]["n0"]
    g0 = parameters["eos"]["g0"]
    rng = range(4, last_term + 1)
    return {
        # "phii": pyo.log(model.delta)
        # + n0[1]
        # + n0[2] * model.tau
        # + n0[3] * pyo.log(model.tau)
        # + sum(n0[i] * pyo.log(1 - pyo.exp(-g0[i] * model.tau)) for i in rng),
        # "phii_d": 1.0 / model.delta,
        # "phii_dd": -1.0 / model.delta**2,
        # "phii_t": n0[2]
        # + n0[3] / model.tau
        # + sum(n0[i] * g0[i] / (pyo.exp(g0[i] * model.tau) - 1) for i in rng),
        # "phii_tt": -n0[3] / model.tau**2
        # - sum(
        #     n0[i]
        #     * g0[i] ** 2
        #     * pyo.exp(-g0[i] * model.tau)
        #     / (1 - pyo.exp(-g0[i] * model.tau)) ** 2
        #     for i in rng
        # ),
        # "phii_dt": 0,
    }