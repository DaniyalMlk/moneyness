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
pip install -e ".[dev]"
```

Python 3.10 or newer. The library itself imports only the standard library;
`pytest`, `mypy`, `ruff` and `mpmath` are used for development and testing.

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

## Running the tests

```bash
pytest                 # test suite
mypy                   # strict type checking, sources and tests
ruff check .           # lint
```

## Status

Phases 1 to 3 of [the roadmap](ROADMAP.md) are in place — the pricing core, the
Greeks, and implied-volatility solving — along with phase 4, which adds
American exercise on binomial and trinomial lattices with the early-exercise
boundary and the Bjerksund-Stensland closed form, and the command-line interface
from phase 7. The volatility surface and Monte Carlo are next, along with the
2002 two-step refinement of the closed form, which needs a bivariate normal
distribution function.

Continuous integration is not yet configured, so the suite is run locally; the
commands above are the whole of it.
