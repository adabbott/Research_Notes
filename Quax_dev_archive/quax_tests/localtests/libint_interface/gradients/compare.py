import psijax
import numpy as onp
#from psijax import external_integrals.libint_interface.eri_deriv as eri_deriv

deriv = psijax.external_integrals.libint_interface.eri_deriv("geom.xyz", "sto-3g", onp.array([0,0,1,0,0,0]))
print(deriv)

