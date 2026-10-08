# Roadmap

Phases are ordered by dependency. Each one is expected to land with tests that
check the numerics against closed-form results, published reference values, or
an independent method that should agree to a stated tolerance.

## Phase 1 — Pricing core

- [x] Normal density, distribution and quantile functions accurate into the tails
- [x] Generalised Black-Scholes-Merton price under a cost-of-carry parameter
- [x] Spot, forward, futures and currency conventions expressed through that parameter
- [x] Degenerate inputs: zero volatility, zero time, zero spot, zero strike
- [x] Put-call parity asserted as an identity, not an example
- [x] Log-moneyness evaluated so that it survives the approach to the strike
- [x] Validation against an independent evaluation at fifty decimal digits

## Phase 2 — Greeks

- [x] First order: delta, vega, theta
- [x] The discount-rate and cost-of-carry sensitivities, taken separately
- [x] Second order: gamma, vanna, volga
- [x] Third order and the remaining cross-sensitivities: charm, veta, speed, colour, zomma
- [x] Every analytic form checked against a numerical derivative of the price
- [x] Dual delta and dual gamma, the strike sensitivities
- [x] Forward delta, for the currency quoting convention
- [x] Degenerate inputs rejected, where the payoff is kinked and no derivative exists

## Phase 3 — Implied volatility

- [x] No-arbitrage price bounds, and rejection of quotes outside them
- [x] The inversion posed in total volatility on the forward, free of carry and discounting
- [x] Initial guess from the normalised price
- [x] Newton iteration on vega, safeguarded by a maintained bracket
- [x] Brent's method on the bracket, as an independent derivative-free check
- [x] Convergence judged on the step in volatility rather than the price residual
- [x] Round-trip recovery across strike, maturity and volatility, to the bound the quote supports
- [x] Behaviour at the bound: intrinsic-value quotes, and quotes no volatility attains

## Phase 4 — American exercise

- [x] Cox-Ross-Rubinstein and Jarrow-Rudd binomial lattices
- [x] Trinomial lattice, and the stability condition on the layer count
- [x] Early-exercise boundary extracted from the lattice
- [x] Richardson extrapolation over the layer count
- [x] Broadie-Detemple smoothing, without which the extrapolation makes the answer worse
- [x] Convergence to the closed form for the European case
- [x] Bjerksund-Stensland closed-form approximation as an independent check
- [x] The 2002 two-step boundary, on a bivariate normal distribution function built for it

## Phase 5 — Volatility surface

- [x] Total implied variance in log-moneyness coordinates
- [x] Raw SVI slice, and its calibration to a set of quotes
- [x] Butterfly arbitrage: the Durrleman condition on a slice
- [x] Calendar arbitrage: monotone total variance across maturities
- [x] Interpolation in maturity that preserves the calendar condition by
      construction, and is checked against the butterfly condition rather than
      assumed to preserve it — a randomised search for a counterexample came up
      empty, which is evidence and not a proof
- [x] Local volatility from the surface, by the Dupire identity

## Phase 6 — Monte Carlo

- [x] Geometric Brownian motion, exact on the terminal law and on a path
- [x] Antithetic variates and a control variate from the closed form
- [x] Standard error reported with every estimate, and the confidence interval
- [x] Asian and barrier payoffs, including the Brownian-bridge barrier correction
- [x] Convergence to the closed form at the stated rate

## Phase 7 — Interface

- [x] Command-line pricing, Greeks and implied-volatility entry points
- [x] Table output for a strike ladder
- [x] Command-line American valuation, with the exercise boundary
- [x] Table output across maturities
- [x] Surface fitting and arbitrage reporting from the command line
- [x] Continuous integration for the test suite and the type checker, across every
      supported interpreter, with the built wheel's entry point exercised

## Phase 8 — Distribution

- [x] MIT licence text in the tree, and inside both the wheel and the sdist
- [x] Distribution metadata an index can present: authors, keywords, classifiers,
      project URLs, and the licence as an SPDX expression
- [x] `moneyness.__version__`, asserted against both the installed metadata and
      the version declared in `pyproject.toml`
- [x] A release driven by a version tag, publishing with the index's trusted
      publishing flow, so no upload credential exists in the repository — and
      refusing to publish when the tag and the declared version disagree
- [x] The sdist and the wheel each installed into a clean environment, with the
      entry point run out of both, on pull requests as well as on a tag
- [ ] A first release on the index, which waits on the publisher being registered
      there for this project

## Phase 9 — Stochastic volatility

- [x] The Heston variance process as a validated parameter object, reporting the
      Feller condition rather than assuming it
- [x] The characteristic function of the log forward, in the branch-stable
      grouping, with the textbook grouping kept alongside so its failure can be
      measured rather than asserted
- [x] Gauss-Legendre quadrature in the standard library, adaptive on a finite
      range and doubling on the half line, shared with the bivariate normal
- [x] European pricing by Lewis's single integral, and by Heston's pair of
      probabilities as an independent route to the same number
- [x] The zero-volatility-of-variance limit equal to the lognormal price on the
      model's own expected integrated variance, to double precision
- [x] A strike ladder from the command line, reported as implied volatilities so
      the generated smile is readable

## Phase 10 — Simulating the variance process

- [x] Andersen's quadratic-exponential scheme, fitting a non-negative law to the
      exact conditional moments of the square-root process, so the Feller
      condition never enters
- [x] The log price carrying both endpoint variances with weights, and the
      martingale correction that makes the simulated forward exact per step
- [x] European, Asian and barrier payoffs with a standard error, antithetic
      sampling and a control variate, on the interface the lognormal simulation
      already uses
- [x] A Brownian-bridge continuity correction for the barrier, using the step's
      own variance, and the discretely monitored contract available as itself
- [x] Agreement with the transform within the reported error at four steps a
      year, including where the Feller condition fails
- [x] The naive alternative implemented and measured rather than dismissed
- [x] Simulated payoffs from the command line, reporting the gap to the
      transform in standard errors where a transform price exists

## Phase 11 — SABR, and where its formula stops being a price

- [x] The four parameters, each with the constraint it actually has, and a
      correlation of one refused because `x(z)` divides by `1 - rho`
- [x] Hagan's lognormal volatility, with the `z / x(z)` factor taken from its
      series where the ratio has lost its digits
- [x] Hagan's normal volatility as its own expansion rather than a conversion
      of the first
- [x] The displaced variant, for the markets whose forwards are negative
- [x] A Bachelier price, its vega and its inversion, which this package had
      none of
- [x] Calibration of `rho` and `nu` with `alpha` solved from the quote nearest
      the money, so every candidate fits that quote by construction
- [x] The risk-neutral density the smile carries, and a search for where it
      turns negative
- [x] A command-line entry point reporting both conventions, the price each
      implies, and the density

The limits are the whole test strategy, because an asymptotic expansion has
nothing exact to be compared against. Three of them are closed forms sharing no
code with the formula. At `nu = 0` and `beta = 1` the implied volatility is
`alpha` — exactly, measured gap 0.0, because every correction term carries a
factor of `nu` or `(1 - beta)`. At `nu = 0` and `beta = 0` the normal volatility
is `alpha` with the same exactness, which depends on its two moneyness brackets
coinciding at a zero exponent. At `nu = 0` with a general exponent the model is
CEV and the formula is *approximate* — and wrong by the amount its own next term
predicts: at a 2% forward and a 3% strike the gap is 1.704e-03 of the volatility
against the `(1-beta)^2 log^2(F/K) / 24` the expansion carries, which is
1.713e-03.

The two conventions have to price the same option, and how closely depends on
the maturity. They are separate second-order expansions in one small parameter,
not two writings of one expression, so measured two standard deviations out the
gap is 0.0009% of the option's value at a quarter of a year, 0.0081% at one year
and 0.1154% at five. Convertible at the short end, not at the long one.

`beta` is not identifiable from one smile, so it is an argument. Fitting the
same five quotes at four exponents moves `alpha` by a factor of 15.5 — 0.001828
at 0.3 against 0.028328 at 1.0 — and `rho` from -0.263 to -0.389, while the
fitted smile moves by at most 6.1 basis points of volatility.

Two defects in the code, and the second one invalidated a finished measurement.

The `z / x(z)` series had the sign of its linear term wrong. The expansion is
`1 - rho z / 2 + ...` and a plus leaves it wrong by `rho z`, which is 3e-03 at a
`z` of 1e-02 — so the series never beat the ratio at any threshold, and that
reads as a badly chosen threshold rather than as a sign error. With the sign
right the two cross at `|z|` of about 2e-04 where each is wrong by around 5e-13,
and below that the ratio's cancellation takes over entirely: 8.3e-08 wrong at
1e-10 and 8.9e-05 at 1e-12, however small `z` gets.

And the density was being differenced from *call* prices. Below the forward a
call is intrinsic plus a whisper: at a 1% strike against a 2% forward it is
0.0094, four of its last bits divided by a squared step of 1e-06 is 6e-06, and
the density there is smaller than that. That produced a confident "negative
density below 1.0649%, sixteen standard deviations out, nine orders of magnitude
down the tail" — written up, measured across four parameter sets, and rounding
in its entirety. Pricing the out-of-the-money option instead, put-call parity
being linear in the strike so the two second differences agree in exact
arithmetic and only one of them in double precision, the finding inverts:

* No one-year smile tested has a negative density at all down to 1e-04 of the
  forward, at any vol-of-vol up to 1.2 and any correlation.
* The defect needs maturity or vol-of-vol. Over five `nu`, four `rho` and four
  maturities, 61 of 80 combinations have one and all 19 that do not are
  short-dated.
* Where it appears it is not a tail curiosity. At `nu` 0.8, `rho` -0.3 and ten
  years it is negative below 1.7935%, which is 0.795 standard deviations below
  the forward, and ten per cent further down it is -7.118 against a peak of
  749.7 — nearly a per cent of the peak, stable to four digits across three
  decades of differencing step.
* It is on both wings at ten years, not only the low one: -1.61 at a 2.67%
  strike. `density_floor` searches downwards only, which its name says.

So a short-dated SABR smile is a distribution and a long-dated one is not, which
is a more useful statement than either "the formula is fine" or "the formula
admits arbitrage".

Three defects in the tests. The `z / x(z)` series error was bounded linearly
where it is cubic. The at-the-money continuity test was written at offsets where
the smile's own skew moves the volatility by more than the tolerance, which is
the skew working rather than a discontinuity. And the one-year high wing was read
off a column printed to six decimals as `0.000000` and asserted to be exactly
zero; it is 4.2e-10 and perfectly well resolved, which is the same lesson as the
density defect in a smaller form.

## Phase 12 — Variance and volatility swaps

- [x] Fair variance by static replication of the log contract, integrated in
      log-moneyness so both wings are one integrand
- [x] The two exact targets: `sigma^2` under a flat smile and
      `E[int V]/T` under Heston, both reproduced from prices alone
- [x] The cost of a finite strike range in closed form, so a quote ladder can
      be judged before it is trusted
- [x] The discrete sum a desk runs, with the centring term the split at a
      listed strike introduces, exactly rather than to leading order
- [x] The variance of integrated variance for the square-root process, and the
      volatility swap strike it implies
- [x] A command-line entry point over a flat volatility, a Heston model and a
      file of quotes

The replication is model-free, which means for once there are answers known in
advance rather than only answers to compare against each other. A flat
Black-Scholes smile must give back `sigma^2`, and it does, to two or three
units in the last place across twelve volatility and maturity pairs. A Heston
smile must give back `E[int V]/T`, which the model already supplies in closed
form, and it does to 1e-8 relative once the strikes reach far enough.

**That qualification is the finding.** The natural rule for how far is a few
standard deviations of log-moneyness, and under a flat smile it is right: five
leaves 4.4e-08 relative and six leaves 1.2e-10. Under Heston at the same
equivalent volatility it is not close. At a vol-of-vol of 0.3, ten standard
deviations leaves 5.7e-07; at 0.5, 5.3e-05; at 0.8, **1.3e-03** — and reaching
1e-09 there takes thirty, which is log-moneyness of six, or strikes from a
quarter of a per cent of the forward to four hundred times it. The log-contract
weight is `1/K^2`, which is exactly the weight that keeps a fat tail relevant,
and stochastic volatility supplies one. So the width is an argument and the
truncation is reported.

It is reported in closed form, and the closed form is the useful kind. Beyond
the last strike the replicating portfolio stops paying
`g(S) = -log(S/F) + S/F - 1` and pays the *tangent* to it, because there is
nothing further out to buy, so the error is the expected excess of `g` over
that tangent — an elementary lognormal expectation. Against the measurement it
is right to 1e-14 relative at one standard deviation and 2e-11 at four. At
equal log-distance the put wing is the dearer one to lose: 1.32 times at one
standard deviation and 2.32 at four, so a desk one strike short should buy the
low one.

**The market's centring correction is the leading term of an exact
expression.** Splitting puts from calls at `K0`, the largest listed strike at
or below the forward, replicates the log contract centred on `K0` rather than
on the forward, and the identity says precisely what that costs:
`2 (F/K0 - 1 - log(F/K0)) / T`. The convention subtracts `(F/K0 - 1)^2 / T`,
which is its first term and overstates it by 0.67% at a one per cent gap, 6.6%
at ten per cent and 32% at a half. Either one restores second-order convergence
in the strike spacing — halving ratios of 4.00 from a spacing of 2.0 down —
against an uncorrected sum whose ratios are 6.64, 10.71, 2.65, 1.71, 7.23 and
1.46, because its error is set by where the forward happens to fall relative to
the lattice rather than by the spacing. At a coarse lattice the discretisation
error is larger than the difference between the two corrections and can cancel
against it, so the exact form being exact does not make it the better estimate
until the spacing is fine.

**A volatility swap is not the square root of a variance swap, and the usual
correction overshoots.** The second-order expansion needs the variance of
integrated variance, which follows from `Cov(V_s, V_t) = e^{-kappa(t-s)}
Var(V_s)` integrated twice over the triangle; it agrees with an independent
double quadrature of that same covariance to 1e-14 relative, and collapses to
the stationary form at `v0 = theta`. Against 60,000 simulated paths at one
year the predicted discount is 0.0195 in volatility against 0.0153 realised at
a vol-of-vol of 0.5, 0.0343 against 0.0245 at 0.8 and 0.0142 against 0.0114 at
0.3 — ratios of 1.27, 1.40 and 1.25, and 13, 22 and 12 standard errors of the
simulation. Right sign, a quarter to a half too large.

Three defects, and all three were in what was asserted rather than in what was
computed.

The unbounded version came first and does not work, for a reason that belongs
to the price function rather than to the quadrature. A transform price has an
*absolute* accuracy floor: asking for 1e-13 returns 1.48e-12 at log-moneyness 2
and 3.02e-12 at 5, which is not even monotone, and 1.381e-03 at 40 where the
true price is zero to hundreds of digits, while the cost per evaluation rises
from 2.3ms to 34ms. Dividing by the strike makes all of that harmless to the
integral — 1.4e-03 over 2.4e+17 is 6e-21 — but a walk looking for a panel that
contributes nothing is looking for a property the integrand does not have, and
it did not terminate in minutes.

The flat-smile test was written at a fixed width in log-moneyness, which is the
wrong unit: a flat 20% at a quarter of a year and a flat 80% at five years
differ by a factor of eighteen in how far the strikes must reach, so five of
twelve cases were failing on truncation and reading as a broken replication.
The quadrature tolerance has the same problem in reverse, being absolute
against an integral of `sigma^2 T / 2` that spans four orders of magnitude over
the same grid.

And truncation was asserted to be monotone in the width everywhere, which is
true only while truncation is resolved. At a quarter of a year, four units of
log-moneyness is already ample, both numbers sit at the rounding floor, and
neither the ordering nor the sign means anything there — so the test now names
the floor and asserts the monotonicity above it. The same lesson in the other
direction: the truncation prediction cannot be checked past six standard
deviations, where the thing predicted is 4.6e-12 against a fair variance of
0.04.

One more worth recording, because it wasted a measurement. A simulation study
that reuses one seed across every row makes that draw's sampling error look
like a systematic bias: the first attempt at the integrated-variance variance
showed the mean 0.5% high and the variance 1% high in all nine rows, uniformly
enough to read as a scheme defect. With independent seeds the same code agrees
to within 1.5 standard errors.

## Phase 13 — Pricing with the local volatility, not just computing it

- [x] Backward finite differences in log-spot for a general local volatility,
      with a tridiagonal solve and no dependencies
- [x] The payoff kink on a grid node, and the spot read off three nodes rather
      than pinned to one
- [x] Crank-Nicolson started by fully implicit steps, judged on gamma rather
      than on the price
- [x] The time-dependent coefficient read at the step midpoint, and the
      calendar times where it jumps forced onto the time grid
- [x] A bridge from the surface that does the forward-log-moneyness mapping in
      one place
- [x] The front stub: total variance ramped from zero, because the surface read
      literally has a calendar arbitrage at the origin
- [x] The surface recovered from prices solved under its own local volatility
- [x] American exercise, the early-exercise premium differenced on one mesh, and
      the exercise boundary
- [x] Repricing a fitted surface against its own quotes from the command line

The identity was already here. What was missing was any way to check it against
a price, and that matters because the identity is stated in coordinates. Local
variance is `(dw/dT) / g` at log-moneyness measured from the *forward*, and the
local volatility it returns belongs to the spot level whose forward
log-moneyness is `k`. Reading `k` as spot log-moneyness differentiates the same
function and so satisfies every test that differentiates anything. At a carry
of 8% it tilts the recovered smile: `-1.8e-02` in volatility on the left wing
and `+1.0e-02` on the right, against `1.1e-04` done correctly. A level error
would hide in a recalibration. A tilt will not.

**The front of the surface cannot be taken literally.** Total variance is held
flat outside the quoted maturities, so `dw/dT` is zero below the first quote
and the local variance with it, while the implied variance at the first quote
is not zero. That is not an artefact of the extrapolation rule, it is a
calendar arbitrage at the origin: total variance has to vanish as maturity
does, and a surface holding it at its first quoted level all the way down does
not. Nothing reproduces such a surface, and the solver handed it returns
exactly zero for a front-slice at-the-money call. The damage does not stay at
the front either — the two-year point still comes back 0.037 low in volatility,
because variance missed before the first quote is never made up afterwards.
Ramping total variance linearly from zero is the default for that reason, and
it brings the whole ladder back to within 1.5e-03.

**Two measurements that went against the first guess.** Widening the spatial
domain was supposed to be a free safety margin. It is not: at a fixed node
count a wider domain is a coarser one, and the `h^2` term it adds dominates the
truncation it removes by four orders of magnitude, so the error rises
monotonically with the width — `2.4e-04`, `4.2e-04`, `6.4e-04`, `9.2e-04`,
`1.6e-03`, `3.7e-03` from three standard deviations to twelve. Truncation
itself, measured properly with the spacing held fixed, is spent by five: a
one-unit widening moves the price by `2.8e-04` at width two, `5.9e-09` at
three, `6.1e-12` at four, and nothing resolvable after that.

And the implicit start is not about the price. Pure Crank-Nicolson prices a
year-long at-the-money call to `-3.4e-03` on a 400-by-40 mesh, which is
respectable, and returns a gamma of 0.0449 against a true 0.0193 — wrong by
132%. One implicit step gives the *best* price of any setting, `-1.3e-03`, and
still leaves gamma out by `-5.6e-04`. Two cost the price `7.3e-04` and bring
gamma to `+2.2e-05`. Each step after that is a first-order step and costs
about `1.3e-03` for nothing. So the default is two, chosen on a quantity the
price never would have selected.

**The bug the round trip caught.** A theta-scheme that evaluates the spatial
operator at both ends of a step and averages them is second order for a
coefficient smooth in time and first order for one that is not. A local
volatility built from an interpolated surface is piecewise constant in time, so
the step beginning at a quoted maturity was receiving the mean of the two sides
of the jump — wrong by `O(1)` in the coefficient, by `O(dtau)` in the price,
once, and never refining away. Through a single slice, where nothing jumps, the
round trip was already converging at ratios of 3.98 to 4.02. Through four
slices it was going 1.63, 1.83, 1.92. Reading one operator per step at the step
*midpoint* fixed it: 4.00, 3.99, 3.96, and the residual at 3200 nodes fell from
`2.48e-05` to `1.02e-06` in implied volatility. Constant volatility is
unaffected to the last digit, which is the check that this changed how the
coefficient is sampled rather than the scheme.

Aligning the time grid to those jumps is a separate matter, and what it buys is
not a smaller error. On a 1.37-year round trip the aligned error falls at
ratios of 3.77, 4.11, 3.94 and 3.98 under doubling, and the unaligned one goes
1.93, 18.08, 6.54 and then changes sign — happening to be smaller at the two
finest meshes and larger at the two coarsest. Which side of a jump each step
reads depends on where the steps fall, so the sequence scatters. An error that
is not monotone in the mesh cannot be extrapolated, and should not be trusted
at any single mesh either.

One correction the suite forced. The ramped stub makes `dw/dT` constant across
the stub, which was described here as making the local variance constant too.
It does not: Durrleman's denominator depends on the *level* of total variance,
and that level is ramping, so the local variance falls by about a quarter from
the origin to the first quote. At the money the denominator does tend to one,
so the local variance there tends to the front slice's own at-the-money implied
variance — an exact limit, now asserted as one.

## Phase 14 — Where the path went, not just where it ended

- [x] The eight standard single-barrier Europeans in closed form, with the
      ``in`` and the ``out`` forms each written out rather than subtracted
- [x] The package's existing barrier vocabulary reused rather than a second one
      invented
- [x] The structural cases answered exactly: the spot on the barrier, and a
      knock-out that cannot pay without breaching
- [x] A knock-out in the finite-difference solver, with the barrier on a node
      and the domain ending there
- [x] The closed forms checked against the solver as a convergence order, and
      against the existing simulation as a z-score
- [x] Discrete monitoring, with the continuity correction's own accuracy
      measured against the discrete contract rather than assumed
- [x] A command-line entry point that shows the price turning over in
      volatility

This is the first contract here that depends on the running extreme rather than
on the terminal distribution alone. The closed form's own parity identity is
worth having and is weaker than it looks: the table is algebraically consistent
with ``in + out = vanilla``, so asserting it to **7.1e-15** catches a term
copied with the wrong sign and cannot catch the table being the wrong table.

**So the price has three derivations.** The solver marches a grid backwards with
a Dirichlet condition at the barrier and converges on the closed forms at
**ratios of 4.00** under doubling of both mesh dimensions — the order the scheme
supports, so the two agree at the mesh's own accuracy rather than to a chosen
tolerance. `monte_carlo.barrier` reaches them from paths through a Brownian
bridge, within **0.8 standard errors** on all four styles. Nothing is shared
between the three.

**The barrier outranks the strike on the grid, and by how much is measured.**
The domain ends at the barrier, so a barrier between nodes is a barrier moved by
up to half a spacing — first order, against the kink's second order. Displacing
it by half a spacing moves the price by **35, 69, 137 and 273 times** the
on-node error as the mesh refines, the ratio growing because the lower-order
term is taking over. Both levels keep their nodes where they are more than a
step apart, by choosing the number of steps between them rather than the
spacing; closer, the node count is capped at twice the request and the strike is
given up. That cap was eight times the request first and let a barrier five
basis points from the strike build 2442 nodes for a request of 400, which is the
same collapse the strike anchoring was written to avoid, under a guard meant to
prevent it.

**The bug was in the domain, not the spacing**, and it produced the worst kind of
wrong answer. The reach was sized from the barrier rather than from the spot, so
a barrier at ten times the spot put the whole grid above the spot and the
quadratic read extrapolated off the nearest three nodes to **-1183.8 for a call
worth 8.65**. Only one test would have caught it — the one comparing a distant
barrier against the unbarriered solve, which exists to isolate the plumbing from
the formula. Every comparison against the closed form uses a barrier within 20%
of the spot, where the mis-sized domain still contained it. The containment is
now swept directly over both sides, barriers from 1% away to ten times the spot,
four widths and four node counts, rather than guarded and hoped for.

**A knock-out is the one price in this package that falls as volatility rises.**
An up-and-out call peaks at **7.51%** and loses 94% of its value between there
and 40%, because the volatility paying for the optionality also pays for the
knock-out. Vega is +31.0 at 5% and -11.9 at 20%, so it changes sign inside the
quoted range and an implied-volatility solve has two roots or none at almost
every price. None is offered, and the command line says why where someone would
look for one.

**And the continuity correction's accuracy is simulated.** A daily close is
worth +13.3% over continuous monitoring, weekly +28.3% and monthly +54.7%, from
barrier shifts of 0.737%, 1.63% and 3.42% — the price is levered about eighteen
times to the level. Against the discrete contract priced directly with the
bridge turned off, the Broadie-Glasserman-Kou shift is **-0.33%** off daily,
**+0.92%** weekly and **+6.82%** monthly. Excellent daily, fine weekly, seven
per cent high monthly. The docstring said "good for daily, rough for monthly"
before any of this was measured; it now says by how much, and says to simulate
the monthly case instead of quoting the correction.

One name collision, caught before it shipped. The module arrived with its own
``Barrier`` dataclass beside ``Side`` and ``Knock``, which collided with
``monte_carlo.Barrier`` in the package namespace and would have left two ways to
say "up-and-out" in one library. That is the mistake `tenor.g2` made with a
call/put flag, and the fix is the same: one vocabulary per package. Reusing the
enum also gave the simulation the same argument order as the formula, which is
what made it usable as a check at all.

## Phase 15 — A price that can only be bracketed

- [x] The exact first two moments of the discretely monitored arithmetic average
- [x] Put-call parity for the average, which needs only the first moment
- [x] A moment-matched lognormal price, with the regime where it fails measured
      rather than asserted
- [x] Curran's conditioning lower bound, by one root solve and a sum
- [x] Rigorous bounds from AM-GM and from convexity, which hold as inequalities
- [x] The side AM-GM falls on determined by the payoff rather than assumed
- [x] A command-line entry point that leads with the interval and flags a price
      that leaves it

Every price before this one is a formula, a grid or a simulation of something
with a known distribution. The arithmetic average of lognormals has no tractable
distribution at all, which is why `monte_carlo.asian` existed and nothing
analytic did. The gap that leaves is not speed. **A simulation cannot bound
anything:** it returns an estimate and a standard error, so it can say a price is
probably near a number and never that it is certainly above or below one.

Two bounds here are inequalities rather than approximations, and each rests on a
statement about the payoff that holds path by path. AM-GM gives `G <= A`, so
`max(G - K, 0) <= max(A - K, 0)`. Convexity gives
`max(A - K, 0) <= (1/m) sum_i max(S_i - K, 0)`, so an Asian is worth at most the
average of ordinary options on its monitoring dates.

**Which side AM-GM falls on depends on the payoff, and that is the trap.** The
same inequality that makes the geometric *call* cheaper makes the geometric *put*
dearer, because `max(K - G, 0) >= max(K - A, 0)`. Measured at the money with 20%
volatility and twelve fixings: a geometric call of 5.9402 under a true 6.1571,
and a geometric put of 3.6517 *over* a true 3.5355. Treating it as a lower bound
for both would have looked right on every call that was tried. For a put it is
therefore an upper bound, and a far better one than convexity — which allows the
fixings to be independent, where averaging destroys much more variance than that.
The put's interval is **0.118** wide against the call's **0.681**.

**Conditioning on the geometric average rather than discarding it is worth a
factor of six hundred.** Curran's bound exercises when `E[A | G] > K`, which is a
strategy a holder could follow, so its value is below the option's. Because the
two averages move together it is nearly exact: against two million paths it is
low by 6.0e-05 relative at 20% volatility, 3.2e-04 at 40% and 1.5e-03 at 80%,
roughly quadrupling per doubling. At 10% volatility the shortfall came out at 0.3
standard errors of that run, so it is reported as unresolved rather than as a
number — an earlier version of the test asserted 5.1e-06 from exactly that draw.

**The moments do three jobs.** `E[S_i S_j] = S^2 exp(b(t_i + t_j) + v^2 min(t_i,
t_j))` makes both moments finite sums with nothing approximate in them. They give
parity exactly, which is the one check that applies to the simulation, the bound
and the approximation alike. They give the moment-matched price. And they give an
oracle the simulation has to reproduce, which is the only check on that
simulation that does not pass through another approximation.

They also say why an Asian is cheap: twelve monthly fixings leave a matched
log-variance of **0.01526** against the terminal price's 0.04 at 20% volatility,
so an effective volatility of **12.35%**. The dense-fixing limit is the textbook
`v^2 T / 3`, approached as `1/3 + 1/(2m)` — at three thousand fixings that
predicts 0.333500 against a measured **0.333503**. That third is a *zero-carry*
statement: at a 5% cost of carry the limit sits at 0.3377 instead, because
`exp(b t)` weights the later and more variable part of the path more heavily. The
test asserted a third with a carry in force and failed by eight times its own
tolerance.

**Moment matching's error changes sign twice, so no tolerance describes it.** At
the money it reads high — 6.1e-04, 1.4e-03, 3.0e-03, 8.5e-03 and 3.0e-02 relative
at 5%, 10%, 20%, 40% and 80% volatility over a year, two to three times worse per
doubling rather than the order of magnitude the shape of the problem suggests. Out
of the money it reads **low enough to break the lower bound**: a call struck at
120 against a spot of 100 is 9.2% below Curran's bound at 10% volatility, 3.4%
below at 20%, 1.8% at 30% and 0.79% at 40%.

Then it crosses. At 60% the same option is 0.9% *above* the bound and at 80% it is
2.7% above, so the shrinking violations at moderate volatility point the wrong way
about what happens next. The 40% figure was first recorded as 7.9% from misreading
`-7.86e-03` as a percentage, which made the trend look monotone and hid the
crossing entirely.

That is what the bounds are for. Nothing else in the module could tell that the
moment-matched price at 120 is wrong, because there is no closed form to compare
it against and the simulation's standard error is wider than the error being
looked for at low volatility. A rigorous inequality can.

## Phase 16 — Two assets, and a bound that reads the correlation

- [x] A pair of lognormal assets carrying two forwards, two volatilities and a
      correlation, with each asset's carry applied on the way in
- [x] Margrabe's exchange option, exact, as the zero-strike reference
- [x] The exact price by conditioning on the second asset, as one integral of a
      Black-76 call against a normal density
- [x] The conditional option's passage through the money located and placed on
      a panel edge, which is where the integrand is kinked rather than merely
      awkward
- [x] A closed-form truncation bound, and the ceiling solved from the tolerance
      rather than chosen
- [x] The accuracy floor measured and reported, since round-off scales with the
      payoff and not with the answer
- [x] Kirk's approximation, with its error measured across strike and
      correlation instead of asserted to be small
- [x] A sub-replicating half-space as a rigorous lower bound, valid for every
      direction and offset so that the search cannot break it
- [x] Super-replication by two vanilla options, shown to equal the price at a
      correlation of minus one rather than merely to bound it
- [x] A command-line entry point that leads with the interval and flags an
      approximation that leaves it

Three routes to one price, and the reason for three is the same as in Phase 15:
the difference of two lognormals has no closed form, so an approximation on its
own cannot be checked. Conditioning on the second asset is exact — given the
variate that drives it, the first asset is still lognormal with a shifted
forward and a reduced variance, so the conditional payoff is a Black-76 call
struck at `S2(z) + K`. Margrabe's formula is the independent reference at a zero
strike, and two more reductions check the rest: with no volatility on the second
asset the spread option is a Black call struck at `F2 + K`, and with none on the
first it is a Black put on the second struck at `F1 - K`. All three agree to
better than 1e-12 relative.

**The kink has to be on a panel edge, and missing it does not look like an
error.** With no volatility on the first asset the integrand is genuinely
kinked at one point. Uniform panels at 24, 48, 96 and 192 panels give relative
errors of 9.6e-06, **1.8e-05**, 3.3e-06 and 1.2e-06 — the first refinement makes
it worse, so a two-point convergence check would have reported the method
diverging rather than crawling. With the root of `F1(z) - S2(z) - K` inserted as
an edge the same quadrature is exact to 4.5e-14. That is the fifth module in
which a discontinuity on a panel edge or a grid node was worth four to eight
orders of magnitude.

**The upper bound is not an accuracy, it is a correlation reading.** For any
level `a`, `max(S1 - S2 - K, 0) <= max(S1 - a, 0) + max(a - S2 - K, 0)` path by
path, so two vanilla options bound the spread with no correlation in them at
all. That is also exactly what they can do: a bound holding for every coupling
of the two marginals is the price under the worst one, and for lognormals that
coupling is attainable. So the optimised portfolio does not approximate the
upper extreme — it **equals** the spread price at a correlation of minus one, to
between 4.2e-14 and 6.7e-14 relative across five strikes. Two routes with
nothing in common, one being two Black formulas and the other a conditional
quadrature, agreeing to machine precision.

Which is also why the bracket is 1.1% of the price wide near a correlation of
minus one and 484% wide at plus 0.9. The width measures distance from the worst
case, not the quality of the method.

### Kirk is not arbitrage-free, and that is measurable

Kirk's approximation reads low at positive correlation and short strikes and
high everywhere else, so no tolerance describes it. At 18 of 84 points swept it
is outside the rigorous interval: fifteen below the sub-replicating lower bound,
by up to 1.1e-03 relative at a correlation of 0.99 and a strike of 2, and three
above the super-replicating upper bound at correlations near minus one and far
strikes, by **1.9e-02** at a correlation of -0.99 and a strike of 40. The second
is the worse kind of wrong: a price above the vanilla super-replication cost is
above what *any* coupling of the two marginals can produce, so it is not a
mispricing of this model but a number no model can return.

The size of the error is monotone in the strike only where its sign is fixed. At
all six non-positive correlations swept it reads high at every strike and grows
with it; at correlations of 0.5 and 0.95 the crossing lands inside the strike
range and the ordering fails. Neither of those is a correlation a sweep of the
negative side would have reached, which is how the first version of that test
passed on a false claim.

### The lower bound's gap is not monotone in the correlation

The half-space bound is exact at both ends — one driving variate at either, so
the exercise region genuinely is a half-space — and loosest in between. At a
strike of 5 its relative gap runs 3.1e-10, 1.0e-06, 5.9e-06, 2.4e-05, 6.9e-05,
1.2e-04 and back down to 1.0e-06 as the correlation goes -0.99, -0.9, 0, 0.5,
0.8, 0.95, 0.999. Asserting that it tightens with the correlation would have
passed on any sweep stopping at 0.9.

It is usually the better number. Over the whole grid its gap is a median sixty
times smaller than Kirk's error and up to ten million times smaller at a
correlation of -0.99. But it loses at eight of eighty-four points, by up to a
factor of three at high correlation and far strikes, so the case for it is not
that it is always closer. It is that its error has a sign.
