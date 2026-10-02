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
