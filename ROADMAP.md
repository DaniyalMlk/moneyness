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

- [ ] No-arbitrage price bounds, and rejection of quotes outside them
- [ ] Rational initial guess from the normalised price
- [ ] Newton iteration on vega with a bracketed fallback
- [ ] Brent's method on the bracket, for the wings where Newton stalls
- [ ] Round-trip recovery across strike, maturity and volatility, to a stated tolerance
- [ ] Behaviour at the bound: intrinsic-value quotes and vanishing vega

## Phase 4 — American exercise

- [ ] Cox-Ross-Rubinstein and Jarrow-Rudd binomial lattices
- [ ] Trinomial lattice, and the stability condition on the layer count
- [ ] Early-exercise boundary extracted from the lattice
- [ ] Richardson extrapolation over the layer count
- [ ] Convergence to the closed form for the European case
- [ ] Bjerksund-Stensland closed-form approximation as an independent check

## Phase 5 — Volatility surface

- [ ] Total implied variance in log-moneyness coordinates
- [ ] Raw SVI slice, and its calibration to a set of quotes
- [ ] Butterfly arbitrage: the Durrleman condition on a slice
- [ ] Calendar arbitrage: monotone total variance across maturities
- [ ] Interpolation in maturity that preserves both conditions
- [ ] Local volatility from the surface, by the Dupire identity

## Phase 6 — Monte Carlo

- [ ] Geometric Brownian motion, exact on the terminal law and on a path
- [ ] Antithetic variates and a control variate from the closed form
- [ ] Standard error reported with every estimate, and the confidence interval
- [ ] Asian and barrier payoffs, including the Brownian-bridge barrier correction
- [ ] Convergence to the closed form at the stated rate

## Phase 7 — Interface

- [ ] Command-line pricing, Greeks and implied-volatility entry points
- [ ] Table output for a strike ladder across maturities
- [ ] Surface fitting and arbitrage reporting from the command line
- [ ] Continuous integration for the test suite and the type checker
