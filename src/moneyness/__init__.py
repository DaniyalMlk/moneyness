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
from .surface import Calendar, LocalVol, Surface
from .svi import SVI, Butterfly, Fit, calibrate, density, durrleman

# Kept in step with the version in pyproject.toml by a test, because the two
# are written in different files and nothing else would notice them drifting.
__version__ = "0.1.0"

__all__ = [
    "SVI",
    "Barrier",
    "Bounds",
    "Branch",
    "Butterfly",
    "Calendar",
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
    "Scheme",
    "Settings",
    "SmilePoint",
    "Solution",
    "Surface",
    "__version__",
    "adaptive_quad",
    "asian",
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
    "dual_delta",
    "dual_gamma",
    "durrleman",
    "european",
    "fixed_quad",
    "forward",
    "forward_delta",
    "gamma",
    "gauss_legendre",
    "geometric_asian",
    "gil_pelaez_price",
    "implied_vol",
    "intrinsic",
    "lewis_price",
    "log_moneyness",
    "min_steps",
    "norm_cdf",
    "norm_cdf2",
    "norm_pdf",
    "norm_ppf",
    "parity_gap",
    "price",
    "price_lattice",
    "rho",
    "rho_carry",
    "richardson",
    "semi_infinite_quad",
    "smile",
    "solve",
    "speed",
    "theta",
    "trigger_price",
    "vanna",
    "variance_path",
    "vega",
    "veta",
    "volga",
    "zomma",
]
