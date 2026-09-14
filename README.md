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
from moneyness import Inputs, OptionType, price

option = Inputs(spot=100.0, strike=95.0, time=0.5, rate=0.04, vol=0.22)
price(option, OptionType.CALL)   # 9.606...
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

## Running the tests

```bash
pytest                 # test suite
mypy                   # strict type checking, sources and tests
ruff check .           # lint
```

## Status

Phases 1 to 3 of [the roadmap](ROADMAP.md) are in place: the pricing core, the
Greeks, and implied-volatility solving. American exercise, the volatility
surface and Monte Carlo are next.

Continuous integration is not yet configured, so the suite is run locally; the
commands above are the whole of it.
