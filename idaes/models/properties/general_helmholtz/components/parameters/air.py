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
"""Generate parameter and expression files for Air
"""

__author__ = "Stephen Burroughs"

import os
import math
import pyomo.environ as pyo
from pyomo.common.fileutils import this_file_dir
from pyomo.common.fileutils import find_library
from idaes.core.util.math import smooth_max
from idaes.models.properties.general_helmholtz.helmholtz_parameters import (
    WriteParameters,
)


def thermal_conductivity_rule(m):
    """Thermal conductivity rule

    Lemmon, E.W. , R.T. Jacobsen (2003). 
    "Viscosity and Thermal Conductivity Equations for Nitrogen, Oxygen, Argon, and Air.
    """
    a = {
        0: 0.431,
        1: -0.4623,
        2: 0.08406,
        3: 0.005341,
        4: -0.00331
    }
    n = {
        1: 1.308,
        2: 1.405,
        3: -1.036,
        4: 8.743,
        5: 14.76,
        6: -16.62,
        7: 3.793,
        8: -6.142,
        9: -0.3778
    }
    t = {
        2: -1.1,
        3: -0.3,
        4: 0.1,
        5: 0,
        6: 0.5,
        7: 2.7,
        8: 0.3,
        9: 1.3
    }
    d = {
        4: 1,
        5: 2,
        6: 3,
        7: 7,
        8: 7,
        9: 11
    }

    y = {
        4: 0,
        5: 0,
        6: 1,
        7: 1,
        8: 1,
        9: 1
    }

    l = {
        4: 0,
        5: 0,
        6: 2,
        7: 2,
        8: 2,
        9: 2
    }

    T = m.T_star / m.tau#Check that values match expected.
    rho = m.delta * m.rho_star
    Ts = T / 103.5
    return(
        n[1] * (
        0.0266958
        * pyo.sqrt(28.9586*T)#check molar weights - inconsistent between thermocond paper and initial param values
        / (0.360 ** 2 * pyo.exp(sum(aval * pyo.log(Ts) ** i for i, aval in a.items()))))
        + n[2] * m.tau ** t[2] 
        + n[3] * m.tau ** t[3]
        + sum(n[i] * m.tau ** t[i] * m.delta ** d[i]
              * pyo.exp(-y[i] * m.delta ** l[i])
              for i in range (4, len(n.items())+1))
    )

def viscosity_rule(m):
    """Viscosity rule

    Lemmon, E.W. , R.T. Jacobsen (2003). 
    "Viscosity and Thermal Conductivity Equations for Nitrogen, Oxygen, Argon, and Air.
    """

    a = {
        0: 0.431,
        1: -0.4623,
        2: 0.08406,
        3: 0.005341,
        4: -0.00331
    }
    n = {
        1: 10.72,
        2: 1.122,
        3: 0.002019,
        4: -8.876,
        5: 0.02916
    }
    t = {
        1: 0.2,
        2: 0.005,
        3: 2.4,
        4: 0.6,
        5: 3.6
    }
    d = {
        1: 1,
        2: 4,
        3: 9,
        4: 1,
        5: 8
    }
    l = {
        1:0,
        2:0,
        3:0,
        4:1,
        5:1
    }

    T = m.T_star / m.tau#Check that values match expected.
    rho = m.delta * m.rho_star
    Ts = T / 103.5
    return (
        0.0266958
        * pyo.sqrt(28.9586*T)#check molar weights - inconsistent between thermocond paper and initial param values
        / (0.360 ** 2 * pyo.exp(sum(aval * pyo.log(Ts) ** i for i, aval in a.items())))
        + sum(n[i] * m.tau ** t[i] * m.delta ** d[i]
              * pyo.exp(-l[i] * m.delta ** l[i])
              for i in range (1, len(n.items())+1))
    )


def main(dry_run=False):
    """Generate parameter and expression files.

    Args:
        dry_run (bool): If dry run don't generate files

    Returns:
        WriteParameters
    """
    main_param_file = os.path.join(this_file_dir(), "h2o.json")
    we = WriteParameters(parameters=main_param_file)
    we.add(
        {
            "viscosity": viscosity_rule,
            "thermal_conductivity": thermal_conductivity_rule,
        }
    )
    we.write(dry_run=dry_run)
    return we


if __name__ == "__main__":
    main()
