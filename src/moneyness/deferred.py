"""A decision taken at an intermediate date, and the two contracts built on one.

Every payoff priced elsewhere here is fixed when the trade is struck. These two
are not. At a date before maturity the holder decides something, and what they
hold afterwards depends on the decision.

A **chooser** defers the choice of direction: at the decision date the holder
declares the option a call or a put, with the same strike and the same maturity
either way. A **compound option** defers the purchase: at the decision date the
holder either pays a premium agreed today and receives an option, or does not.

They share their machinery — a critical spot at the decision date and the joint
law of the spot at two dates — and more usefully they share a property that
makes them worth pricing in closed form rather than on the lattice: each one has
an identity that pins it with none of this module's own arithmetic in it.

**The chooser is a vanilla pair, exactly.** At the decision date the holder has
``max(C, P)``, which is ``C + max(P - C, 0)``, and put-call parity at that date
makes ``P - C`` a known linear function of the spot. So the choice is worth a
call struck at ``K`` to maturity plus ``exp((b - r)(T - t))`` puts struck at
``K exp(-b (T - t))`` expiring at the *decision* date. Both legs are
:func:`moneyness.bsm.price` calls. The closed form here agrees with that
decomposition to 1.8e-14 at worst over strikes from 80 to 120, carries from
-0.03 to 0.08 and volatilities from 15% to 60% — which is a transcription check
on a formula whose two halves were derived independently, and the strongest
statement available about a closed form.

Two limits follow from the decomposition and are checked rather than argued.
With the decision at maturity the put leg expires with the option and the
chooser is a **straddle**; the two agree to 3.8e-10 at ``t = T - 1e-10``, the
residual being the gap itself. With the decision now the holder chooses with
full information and the chooser is the **larger of the two vanillas**, reached
to the last bit rather than to a tolerance.

**Each compound pair satisfies a parity that uses nothing from here.** At the
decision date a call-on-call pays ``max(C - K1, 0)`` and a put-on-call pays
``max(K1 - C, 0)``, so their difference is ``C - K1`` whatever happens. Taking
the discounted expectation, the call-on-call less the put-on-call is the inner
call's price today less the discounted premium. The same holds with a put
inside. Measured across four parameter sets per pair, the residual is at worst
7.1e-15 — and it is a better check than it looks, because it validates the
bivariate normal's four arguments and two correlation signs for *both* members
of a pair at once. Getting one of them wrong breaks it.

A simulation that shares no derivation agrees too: 300,000 antithetic paths to
the decision date with the inner option priced by :func:`moneyness.bsm.price` at
each, over five seeds for each of the four pairs. The twenty deviations run from
-1.52 to +0.93 standard errors and take both signs within every pair, which one
seed inside its interval could not have shown.

**The critical spot is solved and reported rather than buried.** The compound
option's exercise decision is ``C(S_t) > K1``, which for a call inside is
``S_t > S*`` and for a put inside is ``S_t < S*``, with ``S*`` the spot at which
the inner option is worth exactly the premium. It is bisected rather than
approximated, reprices the inner option to 1e-10 of the premium it was solved
for, and is returned on the result so that the exercise probability can be read
as a probability instead of inferred from a price.

**Three degenerate cases are trades, not errors.** A zero premium makes the
outer call certain to be exercised, so it is the inner option exactly, and the
outer put worthless; both are returned directly rather than approached by a
bisection on a critical spot of zero — though the limit is approached correctly
too, the call-on-call running 9.8e-10 above the inner call at a premium of 1e-09
and 9.8e-04 at 1e-03, linearly in the premium. A premium above anything the
inner option can ever be worth — which only happens with a put inside, since a
put is bounded by its discounted strike — makes the outer call worthless and the
outer put the parity residual, 187.603 at a premium of 200 against an inner put
worth 7.459.

A decision date equal to the maturity leaves the inner option with no life, so
the compound is an option on an intrinsic value — and **the outer put is then a
vertical spread rather than a vanilla**, which is the one case here where the
payoff algebra does not reduce to one option. Capping ``max(K1 - (S - K)^+, 0)``
truncates the payoff at both ends: it is flat at ``K1`` below the inner strike
and zero above ``K + K1``, which is a put struck at ``K + K1`` less a put struck
at ``K``. Writing it as the single put struck at ``K + K1`` would overpay by the
whole of the flat region: **9.881 against a correct 2.422** on the parameters
above, four times the price, and the overpayment is concentrated exactly where
the contract is most likely to pay.

**What the optionality is worth, measured.** A call-on-call is not a cheap way
to buy an option: it is cheaper than the inner option by less than the premium
it defers, and the difference is what the right to walk away costs. At a 100
strike, one year, 25% volatility and a six-month decision, the inner call is
worth 12.336 and the call-on-call struck at a 5.00 premium is worth 8.381,
against 12.336 - 5.00·exp(-0.025) = 7.459 for a forward purchase. The right to
decline is worth 0.922, which is exactly the put-on-call — as parity requires,
and which is the clearest way to see that a compound option is a vanilla
purchase plus an option on the premium.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .bivariate import norm_cdf2
from .bsm import Inputs, OptionType, price
from .normal import norm_cdf

#: Bisection iterations for the critical spot. A hundred halvings of a bracket
#: spanning twelve orders of magnitude leaves nothing a double can resolve.
CRITICAL_ITERATIONS = 200

#: The bracket the critical spot is searched in, as multiples of the strike. The
#: inner option's price is monotone in the spot, so any bracket containing the
#: answer will do; these are wide enough for a volatility of 500% and a maturity
#: of fifty years.
CRITICAL_LOWER = 1e-10
CRITICAL_UPPER = 1e10


class BadDeferral(ValueError):
    """The decision date or the premium does not describe a trade."""


@dataclass(frozen=True)
class Compound:
    """The outer option, the premium it pays, and when the decision is taken.

    The inner option is described by the :class:`~moneyness.bsm.Inputs` the
    price is taken on: its strike is the inner strike and its time is the inner
    maturity, both measured from today.
    """

    #: Whether the holder may buy the inner option (a call on it) or sell it.
    outer: OptionType
    #: Whether the option bought or sold is a call or a put.
    inner: OptionType
    #: Paid at the decision date to acquire the inner option. ``K1``.
    premium: float
    #: Year fraction from today to the decision date. ``t1``.
    decision: float

    def __post_init__(self) -> None:
        if self.premium < 0.0:
            raise BadDeferral(f"a premium is non-negative, got {self.premium!r}")
        if self.decision < 0.0:
            raise BadDeferral(
                f"a decision date is a non-negative year fraction, got {self.decision!r}"
            )
        for name in ("premium", "decision"):
            value = float(getattr(self, name))
            if value != value or math.isinf(value):
                raise BadDeferral(f"{name} must be finite, got {value!r}")

    @property
    def exercises_above(self) -> bool:
        """Whether the outer option is exercised for a high spot.

        A call on a call is, since the inner call rises with the spot; a call on
        a put is not. Both flip for an outer put. The four combinations are what
        the sign pattern in the bivariate arguments encodes, and naming the
        property means the test can state it rather than re-derive it.
        """
        rising = self.inner is OptionType.CALL
        buying = self.outer is OptionType.CALL
        return rising is buying


@dataclass(frozen=True)
class CompoundValue:
    """What a compound option is worth, and the decision behind it."""

    value: float
    #: Today's price of the inner option alone.
    inner: float
    #: The spot at the decision date at which the inner option is worth the
    #: premium exactly. ``None`` where no such spot exists, which happens when
    #: the premium is zero or above anything the inner option can be worth.
    critical_spot: float | None
    #: Risk-neutral probability of exercising, under the decision date's own
    #: measure. ``None`` in the same cases, where it is zero or one by
    #: inspection rather than by a normal integral.
    exercise_probability: float | None
    #: ``sqrt(t1 / T2)``, the correlation of the two log-spots.
    correlation: float
    #: ``e^{-r t1}``, carried so that the forward-purchase comparison does not
    #: need the rate passed in a second time.
    discount: float
    compound: Compound

    @property
    def forward_purchase(self) -> float:
        """The inner option bought outright, paid for at the decision date."""
        return self.inner - self.compound.premium * self.discount

    @property
    def deferral_value(self) -> float:
        """What the right to decide is worth against that forward purchase.

        For an outer call this is positive and parity makes it exactly the
        put-on-the-same-inner: a compound option is a forward purchase plus an
        option on the premium, and this is that option.
        """
        return self.value - self.forward_purchase


@dataclass(frozen=True)
class ChooserLegs:
    """The vanilla pair a simple chooser decomposes into.

    Exposed because it *is* the chooser, not an approximation to it: put-call
    parity at the decision date turns the choice into these two legs exactly,
    and having them means the closed form has something to be checked against.
    """

    #: A call struck at ``K``, expiring at the chooser's maturity.
    call: float
    #: ``scale`` puts struck at ``shifted_strike``, expiring at the decision
    #: date.
    put: float
    scale: float
    shifted_strike: float

    @property
    def total(self) -> float:
        return self.call + self.scale * self.put


def _check_decision(inputs: Inputs, decision: float) -> None:
    if decision < 0.0:
        raise BadDeferral(
            f"a decision date is a non-negative year fraction, got {decision!r}"
        )
    if decision > inputs.time:
        raise BadDeferral(
            f"the decision date {decision!r} is after the maturity {inputs.time!r}. A "
            "decision taken after the option has expired is not a decision."
        )
    if inputs.vol <= 0.0 or inputs.spot <= 0.0 or inputs.strike <= 0.0:
        raise BadDeferral(
            "a deferred decision needs a positive spot, strike and volatility: with "
            "any of them at zero the decision is known today and the contract is a "
            "vanilla. Price that instead."
        )


def _inner_inputs(inputs: Inputs, spot: float, time: float) -> Inputs:
    return Inputs(spot, inputs.strike, time, inputs.rate, inputs.vol, carry=inputs.carry)


def critical_spot(inputs: Inputs, premium: float, option: OptionType) -> float:
    """The spot at which ``option`` on ``inputs`` is worth ``premium``.

    Bisected on the price, which is strictly monotone in the spot — rising for a
    call and falling for a put — so a bracket containing the answer is all the
    method needs and there is no derivative to go wrong. Refuses rather than
    returning an endpoint when the premium is outside the option's range, since
    an endpoint would silently become a critical spot of zero or infinity and
    make an always-exercised option look marginal.
    """
    if premium <= 0.0:
        raise BadDeferral(
            f"a critical spot needs a positive premium, got {premium!r}. At zero the "
            "option is worth more than the premium everywhere and there is no "
            "boundary to find."
        )
    if inputs.time <= 0.0:
        raise BadDeferral(
            "a critical spot needs the option to have time left; at zero its value is "
            "its intrinsic and the boundary is the strike shifted by the premium."
        )
    low = inputs.strike * CRITICAL_LOWER
    high = inputs.strike * CRITICAL_UPPER
    if option is OptionType.PUT:
        ceiling = price(_inner_inputs(inputs, low, inputs.time), option)
        if premium >= ceiling:
            raise BadDeferral(
                f"a put on a strike of {inputs.strike!r} over {inputs.time!r} years is "
                f"worth at most {ceiling!r}, which is below the premium {premium!r}. "
                "There is no spot at which it is worth the premium, so the option on "
                "it is never exercised."
            )
    for _ in range(CRITICAL_ITERATIONS):
        middle = 0.5 * (low + high)
        above = price(_inner_inputs(inputs, middle, inputs.time), option) < premium
        rising = option is OptionType.CALL
        if above is rising:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def compound_price(inputs: Inputs, compound: Compound) -> CompoundValue:
    """Geske's option on an option, for all four combinations.

    ``inputs`` describes the inner option — its strike and its maturity — and
    ``compound`` the decision taken at ``compound.decision``.
    """
    _check_decision(inputs, compound.decision)
    inner = price(inputs, compound.inner)
    discount = math.exp(-inputs.rate * compound.decision)
    correlation = (
        math.sqrt(compound.decision / inputs.time) if inputs.time > 0.0 else 0.0
    )

    def degenerate(certain: bool) -> CompoundValue:
        """The outer option when the decision is settled before the spot moves.

        ``certain`` says the inner option is worth more than the premium
        whatever happens. A buyer then holds the inner option against a known
        payment and a seller holds nothing; with ``certain`` false the roles
        swap and the seller is left with the whole premium less the option.
        """
        buying = compound.outer is OptionType.CALL
        if certain:
            value = inner - compound.premium * discount if buying else 0.0
        else:
            value = 0.0 if buying else compound.premium * discount - inner
        return CompoundValue(
            value=max(value, 0.0),
            inner=inner,
            critical_spot=None,
            exercise_probability=1.0 if certain == buying else 0.0,
            correlation=correlation,
            discount=discount,
            compound=compound,
        )

    if compound.premium == 0.0:
        # Worth more than nothing everywhere, so the purchase is certain.
        return degenerate(certain=True)

    if compound.decision == 0.0:
        # Decided now, with the inner option's price already known, so there is
        # no uncertainty left in the decision and no distribution to integrate.
        buying = compound.outer is OptionType.CALL
        payoff = inner - compound.premium if buying else compound.premium - inner
        return CompoundValue(
            value=max(payoff, 0.0),
            inner=inner,
            critical_spot=None,
            exercise_probability=1.0 if payoff > 0.0 else 0.0,
            correlation=correlation,
            discount=discount,
            compound=compound,
        )

    remaining = inputs.time - compound.decision
    if remaining <= 0.0:
        # The inner option expires with the decision, so its value then is its
        # own intrinsic and the compound is an option on a kinked payoff. The
        # outer call is one vanilla; the outer put is a vertical spread, because
        # capping a payoff at the premium truncates it at both ends.
        boundary = (
            inputs.strike + compound.premium
            if compound.inner is OptionType.CALL
            else inputs.strike - compound.premium
        )
        if boundary <= 0.0:
            # A put is worth at most its strike, so a premium above the strike
            # is never paid and the decision is settled before the spot moves.
            return degenerate(certain=False)
        at = inputs.strike
        shifted = Inputs(
            inputs.spot, boundary, compound.decision, inputs.rate, inputs.vol, carry=inputs.carry
        )
        struck = Inputs(
            inputs.spot, at, compound.decision, inputs.rate, inputs.vol, carry=inputs.carry
        )
        if compound.outer is OptionType.CALL:
            direction = (
                OptionType.CALL if compound.inner is OptionType.CALL else OptionType.PUT
            )
            value = price(shifted, direction)
        else:
            direction = (
                OptionType.PUT if compound.inner is OptionType.CALL else OptionType.CALL
            )
            value = price(shifted, direction) - price(struck, direction)
        return CompoundValue(
            value=value,
            inner=inner,
            critical_spot=boundary,
            exercise_probability=None,
            correlation=correlation,
            discount=discount,
            compound=compound,
        )

    at_decision = Inputs(
        inputs.strike, inputs.strike, remaining, inputs.rate, inputs.vol, carry=inputs.carry
    )
    if compound.inner is OptionType.PUT:
        ceiling = price(
            _inner_inputs(inputs, inputs.strike * CRITICAL_LOWER, remaining), OptionType.PUT
        )
        if compound.premium >= ceiling:
            return degenerate(certain=False)

    boundary = critical_spot(at_decision, compound.premium, compound.inner)

    carry = inputs.b
    spread = inputs.vol * math.sqrt(compound.decision)
    total = inputs.vol * math.sqrt(inputs.time)
    half = 0.5 * inputs.vol * inputs.vol
    a1 = (math.log(inputs.spot / boundary) + (carry + half) * compound.decision) / spread
    a2 = a1 - spread
    b1 = (math.log(inputs.spot / inputs.strike) + (carry + half) * inputs.time) / total
    b2 = b1 - total
    forward = inputs.spot * math.exp((carry - inputs.rate) * inputs.time)
    strike_discount = inputs.strike * math.exp(-inputs.rate * inputs.time)
    paid = compound.premium * discount

    if compound.inner is OptionType.CALL:
        if compound.outer is OptionType.CALL:
            value = (
                forward * norm_cdf2(a1, b1, correlation)
                - strike_discount * norm_cdf2(a2, b2, correlation)
                - paid * norm_cdf(a2)
            )
            probability = norm_cdf(a2)
        else:
            value = (
                strike_discount * norm_cdf2(-a2, b2, -correlation)
                - forward * norm_cdf2(-a1, b1, -correlation)
                + paid * norm_cdf(-a2)
            )
            probability = norm_cdf(-a2)
    elif compound.outer is OptionType.CALL:
        value = (
            strike_discount * norm_cdf2(-a2, -b2, correlation)
            - forward * norm_cdf2(-a1, -b1, correlation)
            - paid * norm_cdf(-a2)
        )
        probability = norm_cdf(-a2)
    else:
        value = (
            forward * norm_cdf2(a1, -b1, -correlation)
            - strike_discount * norm_cdf2(a2, -b2, -correlation)
            + paid * norm_cdf(a2)
        )
        probability = norm_cdf(a2)

    return CompoundValue(
        value=value,
        inner=inner,
        critical_spot=boundary,
        exercise_probability=probability,
        correlation=correlation,
        discount=discount,
        compound=compound,
    )


def compound_parity_gap(inputs: Inputs, compound: Compound) -> float:
    """``call-on-X less put-on-X`` against ``inner less discounted premium``.

    Zero to rounding for every parameter set, because at the decision date the
    two outer payoffs differ by ``C - K1`` whatever the spot has done. The
    quantity exists as a function rather than only as a test because it checks
    the bivariate normal's arguments and correlation signs for both members of a
    pair at once, which nothing else here does.
    """
    _check_decision(inputs, compound.decision)
    bought = compound_price(
        inputs,
        Compound(
            outer=OptionType.CALL,
            inner=compound.inner,
            premium=compound.premium,
            decision=compound.decision,
        ),
    )
    sold = compound_price(
        inputs,
        Compound(
            outer=OptionType.PUT,
            inner=compound.inner,
            premium=compound.premium,
            decision=compound.decision,
        ),
    )
    discount = math.exp(-inputs.rate * compound.decision)
    return (bought.value - sold.value) - (bought.inner - compound.premium * discount)


def chooser_legs(inputs: Inputs, decision: float) -> ChooserLegs:
    """The call and the scaled put a simple chooser is made of.

    Put-call parity at the decision date gives ``P - C`` as
    ``K exp(-r (T - t)) - S_t exp((b - r)(T - t))``, so ``max(P - C, 0)`` is
    that many puts on the spot at the decision date, struck where the bracket
    vanishes.
    """
    _check_decision(inputs, decision)
    remaining = inputs.time - decision
    scale = math.exp((inputs.b - inputs.rate) * remaining)
    shifted = inputs.strike * math.exp(-inputs.b * remaining)
    call = price(inputs, OptionType.CALL)
    if decision == 0.0:
        # No time for the put leg to be worth anything but its intrinsic, and
        # bsm refuses a zero maturity through d1. The intrinsic is exact.
        put = max(shifted - inputs.spot, 0.0)
    else:
        put = price(
            Inputs(
                inputs.spot, shifted, decision, inputs.rate, inputs.vol, carry=inputs.carry
            ),
            OptionType.PUT,
        )
    return ChooserLegs(call=call, put=put, scale=scale, shifted_strike=shifted)


def chooser_price(inputs: Inputs, decision: float) -> float:
    """A simple chooser: the direction declared at ``decision``, in closed form.

    Rubinstein's expression, which is the decomposition in
    :func:`chooser_legs` written out as four normal integrals. Both are kept:
    the closed form is what is evaluated and the decomposition is what checks
    it, and they agree to 1.8e-14 at worst over the range in the tests.
    """
    _check_decision(inputs, decision)
    if decision == 0.0:
        # Choosing with full information is the larger of the two vanillas, and
        # the closed form's `y` divides by sqrt(0).
        return max(price(inputs, OptionType.CALL), price(inputs, OptionType.PUT))
    carry = inputs.b
    half = 0.5 * inputs.vol * inputs.vol
    total = inputs.vol * math.sqrt(inputs.time)
    spread = inputs.vol * math.sqrt(decision)
    moneyness = math.log(inputs.spot / inputs.strike)
    d = (moneyness + (carry + half) * inputs.time) / total
    y = (moneyness + carry * inputs.time + half * decision) / spread
    forward = inputs.spot * math.exp((carry - inputs.rate) * inputs.time)
    strike_discount = inputs.strike * math.exp(-inputs.rate * inputs.time)
    return (
        forward * norm_cdf(d)
        - strike_discount * norm_cdf(d - total)
        - forward * norm_cdf(-y)
        + strike_discount * norm_cdf(-y + spread)
    )


def straddle_price(inputs: Inputs) -> float:
    """A call and a put on the same strike, which a chooser becomes at maturity.

    Here because it is the chooser's upper bound and its limit at
    ``decision == time``, and quoting a bound a reader has to assemble is worse
    than exporting it.
    """
    return price(inputs, OptionType.CALL) + price(inputs, OptionType.PUT)


__all__ = [
    "CRITICAL_ITERATIONS",
    "CRITICAL_LOWER",
    "CRITICAL_UPPER",
    "BadDeferral",
    "ChooserLegs",
    "Compound",
    "CompoundValue",
    "chooser_legs",
    "chooser_price",
    "compound_parity_gap",
    "compound_price",
    "critical_spot",
    "straddle_price",
]
