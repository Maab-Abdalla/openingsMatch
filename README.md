# OpeningsMatch

**A constraint-aware hardware recommender for door specification.**

A prototype exploring how ML recommendation can be applied safely in a domain
where a wrong suggestion is not a bad user experience — it is a fire-safety violation.

> **This is a prototype built on synthetic data.** The catalogue is generated
> (`src/generate_catalog.py`); products, brands, and prices are invented, modelled on
> standard architectural hardware categories. The compliance rules are illustrative of
> the *approach*, not a certified fire-door schedule.

---

## The problem

Specifying door hardware is a high-volume, high-consequence, manual task. A single
commercial project can involve hundreds of openings, each needing a complete hardware
set — lock, closer, hinges, exit device, seals — selected from a catalogue spanning
dozens of brands, under fire-code and accessibility constraints.

It is repetitive enough to be automatable, and consequential enough that naive
automation is dangerous.

## Why a standard recommender is the wrong tool

The obvious approach — collaborative filtering, *"specifiers who chose this lock also
chose this closer"* — fails here, and it fails unsafely.

Door hardware carries **hard compatibility and life-safety constraints**:

| Constraint | Consequence of violating it |
|---|---|
| A fire-rated door must be self-closing | The opening can be propped open in a fire |
| A fire door needs intumescent seals | Smoke and flame spread through the perimeter gap |
| Every component on a fire door must be UL-listed | The assembly's fire rating is void |
| An egress door needs panic hardware | People cannot escape under crowd pressure |
| A handed door needs handed hardware | It physically does not fit |

A statistically popular but code-non-compliant recommendation is not a mildly
suboptimal suggestion. **It is a building-code violation that a model cannot be
allowed to make.**

## The architecture

Two layers, in a deliberate order.

```
Door specification
        │
        ▼
┌─────────────────────────────────┐
│  LAYER 1 — HARD CONSTRAINTS     │   Rules engine.
│  Filters the catalogue to what  │   Non-negotiable.
│  is PERMISSIBLE on this door.   │   Runs FIRST.
└─────────────────────────────────┘
        │  only compliant candidates survive
        ▼
┌─────────────────────────────────┐
│  LAYER 2 — SOFT RANKING         │   TF-IDF + cosine similarity.
│  Ranks the survivors by fit to  │   Ranks. Never overrides.
│  the door's requirement profile.│
└─────────────────────────────────┘
        │
        ▼
Justified hardware set
```

**The ordering is the whole design.** The similarity model never sees a non-compliant
candidate, so it can never recommend one. The ML optimises *preference*; the rules
enforce *permissibility*. Preference is never allowed to outrank safety.

This is tested, not asserted — see `test_fire_door_never_gets_non_fire_rated_hardware`.

## Explainability is a requirement, not a feature

A spec writer must defend every choice to a building inspector. So every
recommendation carries its justification:

```
[REQUIRED] CLOSER   Meridian Heavy Duty Closer Fire Rated   ($295.15)
  score=0.582  confidence=strong
  - REQUIRED: fire-rated doors must be self-closing so the opening
    cannot be left propped open in a fire.
  - Fire-rated (UL-listed) component — compliant with fire door assembly.
  - Heavy-duty rating suits a high-traffic opening.
```

### Honest confidence

When a rule mandates a category but text similarity finds **no** signal, the item is
flagged `rule-only` rather than being dressed up as a confident match:

```
[REQUIRED] ACCESS_CONTROL   Meridian Electronic Strike   ($259.12)
  score=0.0  confidence=rule-only
```

The item is there because a rule demanded it — not because the model was sure. Hiding
that distinction would be the more comfortable design and the wrong one.

### Failing loudly

If the catalogue genuinely cannot satisfy a specification, the system **warns** rather
than silently returning an incomplete set. Silent omission is the dangerous failure
mode: a spec that is missing a required closer looks exactly like a spec that never
needed one.

## A bug worth documenting

The security filter was originally applied to **every** category. Because seals carry
`security_level=1`, a high-security fire door filtered out *all* seals — silently
dropping a legally-required fire component from the set.

The fix: seals, hinges, and closers are not security-bearing components. A fire seal's
job is to block smoke, not to resist attack.

```python
SECURITY_BEARING = {"lock", "exit_device", "access_control"}
```

This is exactly the failure class the constraint layer exists to prevent, and it got
past me anyway — which is why `test_fire_door_always_gets_a_seal` now exists as a
regression test. In a life-safety domain, "the model usually gets it right" is not a
standard. The constraint layer has to be provably inviolable, and proof means tests.

## Running it

```bash
pip install -r requirements.txt

python src/generate_catalog.py     # build the synthetic catalogue
python src/recommender.py          # CLI demo: 3 contrasting doors
python -m pytest src/ -v           # 12 tests, incl. the safety properties

python app.py                      # web UI at localhost:5000
```

## Tests

The suite exists to prove the safety property, not to check the maths.

```
test_fire_door_never_gets_non_fire_rated_hardware   ← the critical one
test_fire_door_always_gets_a_closer
test_fire_door_always_gets_a_seal                   ← regression, see above
test_egress_door_always_gets_an_exit_device
test_high_security_door_rejects_weak_locks
test_security_filter_does_not_apply_to_seals
test_unhanded_door_rejects_handed_hardware
test_every_recommendation_is_justified
test_rule_only_recommendations_are_flagged_honestly
test_impossible_spec_warns_rather_than_silently_dropping
...
12 passed
```

## Honest limitations

- **The catalogue is synthetic.** Real product data would change the ranking quality
  substantially; it would not change the architecture.
- **TF-IDF is lexical, not semantic.** It matches word overlap, not meaning — it has no
  concept of what "fire-rated" *means*. Sentence embeddings over real product
  descriptions would retrieve better, and would be the natural next step.
- **The rules here are illustrative.** Real fire-door compliance depends on jurisdiction
  (NFPA 80, BS 476, EN 1634), rating duration, and assembly certification. The point of
  the prototype is the *architecture* — a rules layer gating an ML layer — not the
  specific rule set.
- **No interaction data.** With real specification history, collaborative filtering could
  be layered in as a third signal — still gated by the same constraint layer.

## Stack

Python · scikit-learn (TF-IDF, cosine similarity) · pandas · Flask · pytest
