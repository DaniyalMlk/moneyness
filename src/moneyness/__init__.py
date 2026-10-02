"""Options pricing, Greeks and implied volatility.

The public surface is deliberately small: a value object describing the option
and its market, and free functions over it.
"""

from .american import bjerksund_stensland, bjerksund_stensland_2002, trigger_price
from .bivariate import norm_cdf2
from .bsm import (
    Inputs,
    OptionType,
    d1_d2,
    forward,
    intrinsic,
    log_moneyness,
    parity_gap,
    price,
)
from .greeks import (
    charm,
    colour,
    delta,
    dual_delta,
    dual_gamma,
    forward_delta,
    gamma,
    rho,
    rho_carry,
    speed,
    theta,
    vanna,
    vega,
    veta,
    volga,
    zomma,
)
from .heston import (
    Branch,
    Contract,
    Heston,
    SmilePoint,
    branch_discrepancy,
    char_func,
    gil_pelaez_price,
    lewis_price,
    smile,
)
from .heston_mc import (
    MartingaleCorrectionError,
    Scheme,
    conditional_mean,
    conditional_variance,
    variance_path,
)
from .implied import Bounds, Method, Quote, Solution, bounds, implied_vol, solve
from .lattice import (
    Exercise,
    Lattice,
    LatticePrice,
    boundary,
    min_steps,
    price_lattice,
    richardson,
)
from .monte_carlo import Barrier, Estimate, Settings, asian, barrier, european, geometric_asian
from .normal import norm_cdf, norm_pdf, norm_ppf
from .quadrature import (
    QuadratureError,
    adaptive_quad,
    fixed_quad,
    gauss_legendre,
    semi_infinite_quad,
)

# ``sabr.calibrate``, ``sabr.density`` and ``sabr.smile`` are deliberately not
# re-exported here. ``svi`` already has a ``calibrate`` and a ``density`` and
# ``heston`` already has a ``smile``, and they are different functions rather
# than alternative spellings: SVI's density is in log-moneyness coordinates,
# SABR's is in the strike, and flattening them into one namespace would make
# whichever import came last silently win. They are reached as
# ``moneyness.sabr.calibrate`` and so on.
from .sabr import (
    Calibration,
    SabrParameters,
    Smile,
    bachelier,
    bachelier_vega,
    density_floor,
    implied_normal_vol,
    lognormal_volatility,
    normal_volatility,
    shifted_lognormal_volatility,
)
from .surface import Calendar, LocalVol, Surface
from .svi import SVI, Butterfly, Fit, calibrate, density, durrleman
from .variance import BadStrip, Replication, fair_variance, truncation_error

# Kept in step with the version in pyproject.toml by a test, because the two
# are written in different files and nothing else would notice them drifting.
__version__ = "0.1.0"

__all__ = [
    "SVI",
    "BadStrip",
    "Barrier",
    "Bounds",
    "Branch",
    "Butterfly",
    "Calendar",
    "Calibration",
    "Contract",
    "Estimate",
    "Exercise",
    "Fit",
    "Heston",
    "Inputs",
    "Lattice",
    "LatticePrice",
    "LocalVol",
    "MartingaleCorrectionError",
    "Method",
    "OptionType",
    "QuadratureError",
    "Quote",
    "Replication",
    "SabrParameters",
    "Scheme",
    "Settings",
    "Smile",
    "SmilePoint",
    "Solution",
    "Surface",
    "__version__",
    "adaptive_quad",
    "asian",
    "bachelier",
    "bachelier_vega",
    "barrier",
    "bjerksund_stensland",
    "bjerksund_stensland_2002",
    "boundary",
    "bounds",
    "branch_discrepancy",
    "calibrate",
    "char_func",
    "charm",
    "colour",
    "conditional_mean",
    "conditional_variance",
    "d1_d2",
    "delta",
    "density",
    "density_floor",
    "dual_delta",
    "dual_gamma",
    "durrleman",
    "european",
    "fair_variance",
    "fixed_quad",
    "forward",
    "forward_delta",
    "gamma",
    "gauss_legendre",
    "geometric_asian",
    "gil_pelaez_price",
    "implied_normal_vol",
    "implied_vol",
    "intrinsic",
    "lewis_price",
    "log_moneyness",
    "lognormal_volatility",
    "min_steps",
    "norm_cdf",
    "norm_cdf2",
    "norm_pdf",
    "norm_ppf",
    "normal_volatility",
    "parity_gap",
    "price",
    "price_lattice",
    "rho",
    "rho_carry",
    "richardson",
    "semi_infinite_quad",
    "shifted_lognormal_volatility",
    "smile",
    "solve",
    "speed",
    "theta",
    "trigger_price",
    "truncation_error",
    "vanna",
    "variance_path",
    "vega",
    "veta",
    "volga",
    "zomma",
]
