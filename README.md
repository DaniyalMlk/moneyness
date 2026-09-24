# moneyness

Options pricing, Greeks and implied volatility, in Python with no runtime
dependencies.

The library prices European options under the generalised Black-Scholes-Merton
model, differentiates them analytically, and recovers the volatility implied by
a quoted price. It is written to be correct at the boundaries — zero time, zero
volatility, deep wings, quotes sitting exactly on the arbitrage bound — because
those are the inputs that break naive implementations, and they arrive in real
data more often than the textbook middle does.

## Installing

```bash
pip install moneyness
```

> **Not on the package index yet.** The `pip install` line above is what it
> will be; until the first release lands, install from source:
>
> ```bash
> pip install "git+https://github.com/DaniyalMlk/moneyness.git"
> ```

Python 3.10 or newer. The library itself imports only the standard library, so
there is nothing else to resolve and nothing to compile.

Working on it instead of with it:

```bash
pip install -e ".[dev]"
```

`pytest`, `mypy`, `ruff` and `mpmath` are used for development and testing.
`mpmath` is not optional for the suite — see the decision below.

## Releasing

The version lives in `pyproject.toml` and is mirrored by `moneyness.__version__`;
a test asserts the two agree, and the release refuses to run if the tag
disagrees with either.

```bash
# after the version bump has landed on main
git tag v0.1.0
git push origin v0.1.0
```

The tag builds the sdist and the wheel, installs each into a clean environment,
runs the entry point out of both, and then publishes. Publishing uses the index's
trusted-publishing flow, so there is no API token in this repository, in the
workflow, or in the repository's secrets. Registering the publisher on the index
is a one-time step done there, not here, and it names this repository, the
`release.yml` workflow and the `pypi` environment.

## Using it

```python
from moneyness import Inputs, OptionType, price, delta, vega

option = Inputs(spot=100.0, strike=95.0, time=0.5, rate=0.04, vol=0.22)
price(option, OptionType.CALL)   # 10.043415628...
delta(option, OptionType.CALL)   #  0.704045742...
vega(option)                     # 24.433893679...
```

Recovering the volatility from a quoted price:

```python
from moneyness.implied import Quote, solve

quote = Quote(spot=100.0, strike=95.0, time=0.5, rate=0.04, price=10.0434156286)
solve(quote, OptionType.CALL).vol   # 0.22
```

The four market conventions are reached by moving the cost of carry rather than
by calling a different function:

```python
Inputs(100.0, 95.0, 0.5, 0.04, 0.22)                          # non-dividend share
Inputs.with_dividend(100.0, 95.0, 0.5, 0.04, 0.22, 0.015)     # share paying a yield
Inputs.on_future(100.0, 95.0, 0.5, 0.04, 0.22)                # option on a future
Inputs(100.0, 95.0, 0.5, 0.04, 0.22, carry=0.04 - 0.01)       # currency
```

## The decision that mattered

The tests check the library against `mpmath` evaluating the same mathematics at
fifty decimal digits, not against constants transcribed from a reference book.

The distinction is not pedantic. A transcribed constant tests the transcription:
if a digit is fat-fingered the suite fails for a reason that has nothing to do
with the code, and — worse — if the implementation and the constant are wrong in
the same way, nothing fails at all. The high-precision oracle shares no code, no
algebraic rearrangement and no branch structure with the implementation under
test. When the two agree to 1e-13 across a grid of several hundred inputs, that
is evidence about the implementation.

It also changes what the tests can assert. Because the oracle is available at
any input, the suite checks *relative* accuracy in the far tail, where the true
value is 1e-89 and an absolute tolerance would be satisfied by returning zero.
That caught the specific reason `norm_cdf` is built on `erfc` rather than `erf`:
the `erf` form loses the entire left tail to cancellation below about x = -8.

The same approach sets the tolerances honestly. The round trip
`norm_ppf(norm_cdf(x))` cannot be exact for x above about 4, because the
probability is then within a few floating-point steps of 1 and the information
is destroyed at storage rather than in the algorithm. Rather than loosening the
tolerance until it passes, the suite asserts against the conditioning bound
`ulp(p) / phi(x)` — which still fails if the upper branch regresses.

## From the command line

```bash
moneyness price  --spot 100 --strike 95 --time 0.5 --rate 0.04 --vol 0.22
moneyness greeks --spot 100 --strike 95 --time 0.5 --rate 0.04 --vol 0.22
moneyness iv     --spot 100 --strike 95 --time 0.5 --rate 0.04 --price 10.0434156286
moneyness ladder --spot 100 --time 0.25 --vol 0.3 --rate 0.03 --low 80 --high 120 --steps 5
moneyness american --spot 100 --strike 100 --time 1 --rate 0.06 --vol 0.25 \
                   --dividend 0.06 --put --steps 400 --boundary
```

`iv` reports the method used, the iteration count and the residual alongside the
volatility, and prints the no-arbitrage range when a quote cannot be inverted.

`american` prints the lattice value beside the European value *on the same
lattice* and the European closed form, so the early-exercise premium and the
discretisation error can be read as separate numbers rather than conflated. It
also prints the Bjerksund-Stensland closed-form approximation and its trigger
price, which is a fast independent check on the lattice rather than a second
opinion from the same machinery. `--boundary` tabulates the early-exercise
boundary, and says so plainly when the exercise region is empty.

## A second decision: what the tolerances mean

Several tests assert against a *conditioning bound* rather than a fixed
tolerance, and the distinction is what makes them worth having.

Recovering volatility from the price of a deep in-the-money option is limited by
the quote, not by the solver. A price is a double, so it moves in steps of
`ulp(price)`; dividing that by `dPrice/dVol` gives the finest volatility the
quote can distinguish at all. For a call struck at 20 against a spot of 100 the
entire time value is 1.4e-9 and that ratio is 2.6e-7 relative — so no solver can
do better, because the information is not in the input.

A fixed tolerance has to be either tight enough to fail on those quotes or loose
enough to stop detecting regressions in the well-conditioned majority. Asserting
against the bound itself avoids the choice, and documents the real guarantee.

The same reasoning fixed a genuine defect. The implied-volatility solver
originally stopped when the price residual was small, which is the natural thing
to write and is wrong: where the price is nearly flat in volatility, that
threshold is met while the volatility is still incorrect in its sixth digit.
Converging on the step in volatility instead — the quantity that is well posed —
brought it to the precision the quote supports, and the disagreement with Brent
that exposed the problem is now a standing test.

## A third decision: why the obvious accuracy trick had to be earned

A lattice price carries an `O(1/n)` error in the layer count, so the textbook
move is Richardson extrapolation: price at `n` and `2n`, and take `2P(2n) - P(n)`
to cancel the leading term. Measured on a European call, doing that to a raw
lattice makes the answer about **twice worse**, not four times better.

The premise is what fails. Richardson assumes the error is a smooth `c/n`, and a
lattice error is not: it carries a large component that oscillates with `n`,
because what the lattice gets wrong depends on where the strike falls between
terminal nodes, and that position jumps around as `n` changes. Subtracting one
price from twice another amplifies exactly that component.

The fix is to remove the cause rather than tune around it. The oscillation comes
from the kink in the terminal payoff being sampled at nodes, so the layer before
expiry is replaced with the closed-form European price — the Broadie-Detemple
smoothing — which integrates across the kink exactly. The remaining error is
smooth, and extrapolation then does what it promised:

| lattice | fine lattice | + Richardson | + smoothing | + both |
|---|---|---|---|---|
| Cox-Ross-Rubinstein | 2.9e-03 | 5.5e-03 | 1.4e-03 | 4.5e-05 |
| Jarrow-Rudd | 2.7e-03 | 6.8e-03 | 1.4e-03 | 5.0e-05 |
| trinomial | 1.6e-03 | 2.3e-03 | 7.2e-04 | 4.7e-06 |

Mean absolute error against the closed form over layer counts from 60 to 120.
Smoothing and extrapolation are each worth a factor of two on their own; together
they are worth 30x to 150x. Both remain switchable, and the failure above is
itself a test, so the default cannot quietly stop being the right one.

## A fourth decision: bracket the approximation, do not just measure it

The American price has no closed form, so there is nothing to check an American
implementation against — which usually means settling for "the two methods agree
to three decimals" and hoping that is enough.

There is something better available here. The Bjerksund-Stensland formula is the
exact value of a *flat-boundary* exercise strategy: exercise the moment the
underlying first crosses a fixed level. That strategy is one the holder could
really follow, just not the best one, so its value is a genuine lower bound on
the American price and an upper bound on the European price:

```
European <= Bjerksund-Stensland <= American
```

The suite asserts that inequality on every row of its grid, for both option
types, rather than only asserting closeness. It is the sharper statement: a
tolerance says two numbers are near each other, while the bracket says which side
of the lattice the approximation has to fall on, and an implementation that
drifted above the lattice would fail it while still looking accurate to three
decimals. A companion tolerance stops the bracket being satisfied the lazy way,
by returning the European value and sitting at the bottom of the band.

Puts are not implemented twice. They route through the call by the
McDonald-Schroder transformation, `P(S, K, T, r, b, v) = C(K, S, T, r - b, -b, v)`,
so there is one exercise rule in the codebase rather than two that can drift
apart — and the transformation itself is asserted, not assumed.

## A fifth decision: let the arbitrage conditions do the work

A volatility surface has to satisfy two conditions to describe a possible
market. Each slice must imply a non-negative probability density — no butterfly
arbitrage — and total variance must not fall as maturity grows at fixed
log-moneyness, or one could buy the longer option, sell the shorter, and collect
a certain profit.

It is tempting to treat these as validation: fit the surface, then check it.
They turn out to be more useful than that, because of what Dupire's identity
looks like in these coordinates:

```
sigma_local^2(k, T) = (dw/dT) / g(k, T)
```

where `w` is total variance and `g` is the Durrleman function — the same `g`
whose sign *is* the butterfly condition. The numerator is non-negative exactly
when there is no calendar arbitrage. The denominator is positive exactly when
there is no butterfly arbitrage. So a local volatility exists, as a real
non-negative number, if and only if the surface admits neither.

The two conditions are not hygiene. They are precisely the conditions under
which "what volatility would reproduce these prices" has an answer at all. So
`local_vol` returns the numerator and the denominator alongside the result,
because when the answer is unusable the caller needs to know which one failed:
a negative numerator is slices crossing, which is a data problem, and a
non-positive denominator is a density defect inside one slice, which is a
fitting problem. They have different fixes.

The choice of interpolation follows from the same thinking. Linear in total
variance at fixed log-moneyness makes the calendar condition automatic — a
linear function between two ordered endpoints is monotone, so non-crossing
slices cannot interpolate to crossing ones. The butterfly condition gets no such
argument, since `g` is nonlinear in `w`. A randomised search over tens of
thousands of admissible, non-crossing pairs failed to find an interpolated
violation, which is suggestive and is not a proof, so the surface recomputes `g`
on interpolated slices rather than assuming it, and checks the midpoints of the
gaps by default.

Calibration exploits the shape of SVI rather than throwing five parameters at an
optimiser. Substituting `y = (k - m) / s` makes the parametrisation *linear* in
the three remaining coefficients, so the fit is a three-variable least squares
solved in closed form inside a two-variable search. The constraint that keeps a
slice non-negative everywhere turns out to be a second-order cone, which is
projected onto exactly rather than clamped coordinate by coordinate.

```python
from moneyness import Surface, calibrate

fit = calibrate(log_moneyness, total_variances)
fit.rmse, fit.slice_.wing_slopes

surface = Surface([(0.25, near), (1.0, mid), (2.0, far)])
[entry.free for entry in surface.calendar()]      # no slices crossing
surface.local_vol(k=-0.2, time=0.8).volatility
```

## A sixth decision: keep the simulation exact, and say what the error is

The Monte Carlo does not run an Euler scheme. Geometric Brownian motion has
lognormal increments in closed form, so a path can be drawn with exactly the
right joint law at its monitoring dates, and there is no step-size bias to trade
against sample count.

That separation earns its keep on barrier options. A barrier monitored weekly
genuinely is a different contract from one monitored continuously — not a worse
approximation to it — and if the simulation carried a discretisation error on
top, the two effects would be impossible to tell apart. Here the only
discretisation is the monitoring schedule, which is a term of the contract.

Standard errors are computed on the count of *independent* samples, not paths.
Under antithetic sampling a path and its mirror are not independent, which is
the entire point of drawing them, so the pair is one sample. Dividing by the
path count instead understates the error by about `sqrt(2)` — silently, and in
the direction that flatters the estimate. The suite checks the reported standard
error against the spread actually observed across sixty seeds, which is the
assertion that would catch it.

Three claims here are measured rather than asserted:

| claim | measured |
|---|---|
| convergence at `n^-1/2` | RMS error over 40 seeds falls by 2.02x, 2.13x, 2.34x per 4x sample increase |
| the geometric-average control on an arithmetic Asian | standard error falls by more than 10x |
| the Brownian bridge on a barrier | step-dependence falls from 1.03 to 0.07 across an 80x sweep of monitoring frequency, about 15x |

Knock-ins are priced as the vanilla minus the knock-out on the same paths, so
in-out parity holds exactly rather than to within two simulations' noise. The
barrier prices are validated against a Reiner-Rubinstein formula implemented
separately in the test suite — the bridge is essentially unbiased at ten
monitoring dates, where the naive estimate is off by many standard errors.

```python
from moneyness import Inputs, OptionType, Settings, Barrier, barrier

option = Inputs(100.0, 100.0, 1.0, 0.05, 0.2)
estimate = barrier(option, OptionType.CALL, 90.0, Barrier.DOWN_AND_OUT, 50)
estimate.value, estimate.standard_error, estimate.interval(0.95)
```

## From the command line, on a whole surface

`surface` reads a CSV of `maturity,strike,vol` — a file, or `-` for standard
input — fits an SVI slice per maturity, and reports what the result implies.

```bash
moneyness surface --quotes quotes.csv --spot 100 --rate 0.05 --local-vol
```

```
 maturity  quotes        rmse     max err         a        b      rho        m        s    left   right    worst g
------------------------------------------------------------------------------------------------------------------------
   0.2500       9   4.504e-12   9.550e-12   0.02000  0.15000 -0.35000 -0.00000  0.12000  -0.203   0.097    0.24751
   1.0000       9   1.420e-11   2.529e-11   0.05000  0.22000 -0.30000 -0.00000  0.18000  -0.286   0.154    0.24879
   2.0000       9   1.808e-11   3.465e-11   0.10000  0.30000 -0.25000  0.00000  0.25000  -0.375   0.225    0.25048

calendar (total variance must not fall as maturity grows)
  0.2500 -> 1.0000  clear, closest approach 0.05052
  1.0000 -> 2.0000  clear, closest approach 0.08452

verdict: no arbitrage found
```

The wing slopes are printed next to the fit quality on purpose: a slope near
Lee's bound of 2 is a warning about the tails whatever the residual says.

It exits 1 when the fitted surface admits arbitrage, so it can be used as a
check in a pipeline. That is a finding about the data rather than a failure of
the request, which is why it is an exit status and not an exception.

## A seventh decision: take the maximum of admissible bounds

The Bjerksund-Stensland approximation is the exact value of a *flat-boundary*
exercise strategy — admissible but not optimal — which is what makes
`European <= approximation <= American` hold by construction rather than by
luck. The 2002 refinement lets the boundary step once partway through the
option's life, which needs a bivariate normal distribution function: the joint
event of not having crossed the first level before the step and finishing in
the money is inherently two-dimensional.

It is tempting to argue the two-step value can never be worse than the flat
one, since a flat boundary is the special case of the two levels coinciding.
That argument is about the *optimal* two-step boundary. The levels here come
from a closed-form heuristic optimised for neither, and measured over 240
random markets the raw 2002 formula falls below the 1993 value on nine of them
— and in one case below the European price, which no American value may do,
since holding to expiry is always available.

So `bjerksund_stensland_2002` returns the largest of the European price, the
1993 value and the two-step value. Each is the value of a strategy the holder
could actually follow, so each is a lower bound, and the largest of several
lower bounds is both the sharpest available and still a bound:

```
European <= max(European, 1993, 2002) <= American
```

That is not a patch over a numerical problem. It is the right way to combine
admissible strategies when none dominates everywhere. It binds on about four
per cent of that grid, and the suite asserts it binds rather than sitting
unused — with the specific market that triggers it written into a test.

| | mean absolute error vs. a converged lattice |
|---|---|
| 1993 single boundary | 0.0666 |
| 2002, guarded | 0.0420 |

The bivariate normal underneath is built from Sheppard's identity with the
substitution `t = sin(theta)`, which removes the `1/sqrt(1 - t^2)` singularity
exactly where it would otherwise bite — at correlations near one, which is
where the two-step boundary evaluates it. Worst absolute error against
fifty-digit integration over 810 points, correlations from -0.9999 to 0.9999
and arguments to eight standard deviations: **2.2e-16**.

Its Gauss-Legendre nodes are computed from the Legendre recurrence at import
rather than transcribed. Forty table entries all look equally plausible, and a
single wrong digit gives a rule that is slightly wrong everywhere — accurate
enough to look fine and never exactly right. The tests check the rule by the
property that defines it: exact on every polynomial up to degree `2n - 1`, and
not beyond.

## Running the tests

```bash
pytest                 # test suite
mypy                   # strict type checking, sources and tests
ruff check .           # lint
```

The same three run on every push and pull request, and the suite runs on each
Python version the package claims to support rather than only the newest. That
matters more here than it would in most projects: there are no dependencies to
resolve differently, so an interpreter difference surfaces directly as a
difference in floating-point or `math` behaviour — which is precisely what a
single-version matrix is the wrong shape to catch.

A fourth job builds the wheel, installs it into an empty virtualenv with no
source tree in sight, and runs the console entry point. An entry point can be
perfectly healthy in a source checkout and broken the moment it is installed,
and nothing short of installing it will say so.

## Status

Phases 1 to 3 of [the roadmap](ROADMAP.md) are in place — the pricing core, the
Greeks, and implied-volatility solving — along with phase 4, which adds American
exercise on binomial and trinomial lattices with the early-exercise boundary and
the Bjerksund-Stensland closed form, and phase 5, the volatility surface: SVI
slices and their calibration, both arbitrage conditions, and Dupire local
volatility. Most of the command-line interface from phase 7 is there too.

Phase 6 is in place too: an exact Monte Carlo with antithetic and control
variates, Asian and barrier payoffs, and the Brownian bridge correction.

Phase 7 is complete: the command line covers pricing, Greeks, implied
volatility, a strike ladder, a term structure, American valuation and surface
fitting with arbitrage reporting.

Every item on [the roadmap](ROADMAP.md) is now done, including the 2002
two-step boundary and the bivariate normal distribution function it needed.
