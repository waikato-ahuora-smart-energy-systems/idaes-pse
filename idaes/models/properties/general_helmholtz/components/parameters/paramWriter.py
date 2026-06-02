from idaes.models.properties.general_helmholtz.helmholtz_parameters import (
    WriteParameters,
)
from idaes.models.properties.general_helmholtz import(
     HelmholtzParameterBlock, 
     AmountBasis, 
     HelmholtzThermoExpressions
)
from idaes.models.properties.general_helmholtz.helmholtz_functions import(
StateVars, _get_data_dir, add_helmholtz_external_functions
) 
import pyomo.environ as pyo
from pyomo.common.fileutils import find_library
from idaes.core import FlowsheetBlock
import CoolProp.CoolProp as CP


# fluidSet = ["1-butene.json", "acetone.json", "air.json", "ammonia.json", "argon.json", "benzene.json", "carbondioxide.json", "carbonmonoxide.json", "carbonylsulfide.json", "chlorine.json", "cis-2-butene.json", "cyclohexane.json", "cyclopentane.json", "d4.json", "d5.json", "deuterium.json", "dichloroethane.json", "diethylether.json", "dimethylcarbonate.json", "dimethylether.json", "ethane.json", "ethanol.json", "ethylbenzene.json", "ethylene.json", "ethyleneoxide.json", "fluorine.json", "heavywater.json", "helium.json", "hydrogen.json", "hydrogenchloride.json", "hydrogensulfide.json", "isobutane.json", "isobutene.json", "isohexane.json", "isopentane.json", "krypton.json", "m-xylene.json", "md2m.json", "md3m.json", "md4m.json", "mdm.json", "methane.json", "mm.json", "n-butane.json", "n-decane.json", "n-dodecane.json", "n-hexane.json", "n-nonane.json", "n-octane.json", "n-pentane.json", "n-perfluorobutane.json", "n-perfluorohexane.json", "n-perfluoropentane.json", "n-propane.json", "neon.json", "neopentane.json", "nitrogen.json", "nitrousoxide.json", "novec649.json", "o-xylene.json", "orthodeuterium.json", "orthohydrogen.json", "oxygen.json", "p-xylene.json", "paradeuterium.json", "parahydrogen.json", "propylene.json", "propyleneglycol.json", "r1123.json", "r113.json", "r1130(e).json", "r1132(e).json", "r115.json", "r116.json", "r12.json", "r1224yd(z).json", "r1233zd(e).json", "r1234yf.json", "r1234ze(e).json", "r1234ze(z).json", "r124.json", "r1243zf.json", "r125.json", "r1336mzz(e).json", "r134a.json", "r13i1.json", "r141b.json", "r142b.json", "r152a.json", "r161.json", "r218.json", "r227ea.json", "r23.json", "r236ea.json", "r236fa.json", "r245ca.json", "r245fa.json", "r32.json", "r365mfc.json", "r40.json", "r404a.json", "r407c.json", "r41.json", "r410a.json", "r507a.json", "ses36.json", "sulfurdioxide.json", "sulfurhexafluoride.json", "tetrahydrofuran.json", "toluene.json", "trans-2-butene.json", "vinylchloride.json", "water.json", "xenon.json"]
fluidSet = ["fluorine.json"]
for fluid in fluidSet:
    try:
        wp = WriteParameters(parameters=fluid)
        wp.write(dry_run=False)
    except Exception as e:
        print(f"Error processing {fluid}: {e}")
        break

