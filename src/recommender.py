"""
OpeningsMatch - a constraint-aware hardware recommender for door specification.

WHY NOT A PLAIN RECOMMENDER?
----------------------------
A standard "customers who bought X also bought Y" recommender is unsafe in this
domain. Door hardware has HARD compatibility and life-safety constraints:

  * A fire-rated door legally REQUIRES a self-closing device.
  * A fire-rated door REQUIRES fire-rated (UL-listed) components throughout.
  * An egress route REQUIRES an exit device (panic hardware).
  * A handed door needs handed hardware.

A statistically popular but code-non-compliant suggestion is not a mildly bad
recommendation - it is a fire-safety violation.

So this engine is a HYBRID with two layers:

  Layer 1 (HARD CONSTRAINTS) - a rules engine that filters the catalogue down
      to only what is *permissible* for this door. Non-negotiable. Rules win.

  Layer 2 (SOFT RANKING) - TF-IDF + cosine similarity ranks the surviving
      candidates by how well they match the door's requirement profile.

The constraint layer runs FIRST. The ML never gets the chance to suggest
something illegal. Every recommendation carries a human-readable justification.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


# ---------------------------------------------------------------------------
# The door being specified
# ---------------------------------------------------------------------------
@dataclass
class Door:
    fire_rated: bool = False
    is_egress: bool = False           # on an escape route?
    handed: bool = True               # needs left/right handing?
    security_level: int = 2           # 1 (low) .. 5 (high)
    location: str = "interior"        # interior | exterior | stairwell
    traffic: str = "standard"         # standard | high
    acoustic: bool = False            # needs sound insulation?

    def profile_text(self) -> str:
        """Build the text query that TF-IDF will match against descriptions.

        This is the bridge from structured door attributes to the text space
        the product descriptions live in.
        """
        terms = []
        if self.fire_rated:
            terms += ["fire", "rated", "ul", "listed", "self", "closing"]
        if self.is_egress:
            terms += ["egress", "exit", "panic", "emergency"]
        if self.security_level >= 4:
            terms += ["high", "security", "heavy", "duty", "commercial"]
        elif self.security_level <= 1:
            terms += ["residential", "interior", "light", "duty"]
        if self.traffic == "high":
            terms += ["heavy", "duty", "high", "traffic", "commercial"]
        if self.location == "exterior":
            terms += ["exterior", "weather", "water", "resistant"]
        if self.location == "stairwell":
            terms += ["stairwell", "commercial", "self", "latching"]
        if self.acoustic:
            terms += ["acoustic", "sound", "insulation"]
        return " ".join(terms) or "standard commercial door hardware"


# ---------------------------------------------------------------------------
# A single recommendation, with its reasoning
# ---------------------------------------------------------------------------
@dataclass
class Recommendation:
    product_id: str
    name: str
    category: str
    brand: str
    price: float
    score: float
    required: bool                       # is this category legally mandatory?
    confidence: str = "moderate"         # strong | moderate | weak | rule-only
    reasons: List[str] = field(default_factory=list)


class OpeningsMatch:
    def __init__(self, catalog: Optional[pd.DataFrame] = None):
        if catalog is not None:
            self.catalog = catalog
        else:
            csv = DATA / "catalog.csv"
            # Defensive: if the catalogue was never generated (e.g. a fresh
            # deploy that didn't run the generator), build it on the fly rather
            # than crashing the container on startup.
            if not csv.exists():
                from generate_catalog import build_catalog
                DATA.mkdir(exist_ok=True)
                build_catalog().to_csv(csv, index=False)
            self.catalog = pd.read_csv(csv)
        # Fit TF-IDF once over all product descriptions.
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.catalog["description"])

    # -----------------------------------------------------------------------
    # LAYER 1 - HARD CONSTRAINTS
    # -----------------------------------------------------------------------
    def required_categories(self, door: Door) -> Dict[str, str]:
        """Which hardware categories are MANDATORY for this door, and why.

        These are life-safety and functional requirements, not preferences.
        """
        req: Dict[str, str] = {
            "lock": "Every door opening requires a locking or latching device.",
            "hinge": "Every door opening requires hanging hardware.",
        }
        if door.fire_rated:
            req["closer"] = (
                "REQUIRED: fire-rated doors must be self-closing so the "
                "opening cannot be left propped open in a fire."
            )
            req["seal"] = (
                "REQUIRED: fire-rated doors need intumescent seals to block "
                "smoke and flame spread through the perimeter gap."
            )
        if door.is_egress:
            req["exit_device"] = (
                "REQUIRED: doors on an egress route must open under pressure "
                "without prior knowledge of the hardware (panic hardware)."
            )
        if door.security_level >= 4:
            req["access_control"] = (
                "REQUIRED at security level 4+: controlled credential-based entry."
            )
        return req

    # Only these categories BEAR security. A seal or a hinge is not a security
    # component - a fire seal's job is to block smoke, not to resist attack.
    # Applying a security floor to every category is a bug: it silently deletes
    # legally-required fire seals from high-security fire doors.
    SECURITY_BEARING = {"lock", "exit_device", "access_control"}

    def filter_permissible(self, door: Door, category: str) -> pd.DataFrame:
        """Reduce the catalogue to products that are ALLOWED on this door.

        This runs BEFORE any similarity scoring. The ML cannot override it.
        """
        df = self.catalog[self.catalog["category"] == category].copy()

        # CONSTRAINT 1: a fire door needs fire-rated components throughout.
        if door.fire_rated:
            df = df[df["fire_rated"]]

        # CONSTRAINT 2: security-bearing hardware must meet the door's security
        # requirement. A level-4 door cannot take a level-1 passage latch.
        # Deliberately NOT applied to seals/hinges/closers - see SECURITY_BEARING.
        if category in self.SECURITY_BEARING:
            df = df[df["security_level"] >= door.security_level]

        # CONSTRAINT 3: an unhanded door cannot take handed-only hardware.
        if not door.handed:
            df = df[~df["handed"]]

        return df

    # -----------------------------------------------------------------------
    # LAYER 2 - SOFT RANKING (only over what survived Layer 1)
    # -----------------------------------------------------------------------
    def rank(self, door: Door, candidates: pd.DataFrame) -> pd.DataFrame:
        """Score permissible candidates by TF-IDF cosine similarity to the
        door's requirement profile."""
        if candidates.empty:
            return candidates

        query_vec = self.vectorizer.transform([door.profile_text()])
        cand_vecs = self.matrix[candidates.index]
        scores = cosine_similarity(query_vec, cand_vecs).flatten()

        out = candidates.copy()
        out["score"] = scores
        return out.sort_values("score", ascending=False)

    # -----------------------------------------------------------------------
    # ORCHESTRATION
    # -----------------------------------------------------------------------
    def _explain(self, door: Door, row, required_reason: Optional[str]) -> List[str]:
        """Human-readable justification for a single recommendation.

        Explainability is not decoration here. A spec writer must be able to
        defend every choice to a building inspector.
        """
        reasons = []
        if required_reason:
            reasons.append(required_reason)
        if door.fire_rated and row["fire_rated"]:
            reasons.append("Fire-rated (UL-listed) component - compliant with fire door assembly.")
        if door.traffic == "high" and "heavy" in row["description"]:
            reasons.append("Heavy-duty rating suits a high-traffic opening.")
        if door.security_level >= 4 and row["security_level"] >= 4:
            reasons.append(f"Security level {row['security_level']} meets the level-{door.security_level} requirement.")
        if door.location == "exterior" and "weather" in row["description"]:
            reasons.append("Weather-resistant - suitable for an exterior opening.")
        if door.acoustic and "acoustic" in row["description"]:
            reasons.append("Provides acoustic insulation as requested.")
        if not reasons:
            reasons.append("Best text-similarity match to this door's requirement profile.")
        return reasons

    @staticmethod
    def _confidence(score: float, required: bool) -> str:
        """Be honest about WHY an item is in the set.

        A score of 0.0 means TF-IDF found no textual signal - the item is here
        purely because a hard rule demanded the category. That is a legitimate
        outcome, but the spec writer must SEE it rather than mistake it for a
        confident match.
        """
        if score == 0.0 and required:
            return "rule-only"      # mandated by code, no similarity signal
        if score >= 0.40:
            return "strong"
        if score >= 0.20:
            return "moderate"
        return "weak"

    def recommend_set(self, door: Door, per_category: int = 1) -> Dict:
        """Produce a complete, compliant hardware set for a door.

        Returns the recommended set plus an audit trail of the constraints applied.
        """
        required = self.required_categories(door)
        results: List[Recommendation] = []
        warnings: List[str] = []

        # Recommend for every required category, plus any optional extras
        # that the door's attributes call for.
        categories = list(required.keys())
        if door.acoustic and "seal" not in categories:
            categories.append("seal")
        if door.location == "exterior" and "seal" not in categories:
            categories.append("seal")

        for cat in categories:
            permissible = self.filter_permissible(door, cat)

            if permissible.empty:
                warnings.append(
                    f"No compliant '{cat}' found in the catalogue for this door "
                    f"specification. This opening cannot be completed - escalate."
                )
                continue

            ranked = self.rank(door, permissible).head(per_category)
            for _, row in ranked.iterrows():
                score = round(float(row["score"]), 3)
                is_req = cat in required
                results.append(Recommendation(
                    product_id=row["product_id"],
                    name=row["name"],
                    category=cat,
                    brand=row["brand"],
                    price=float(row["price"]),
                    score=score,
                    required=is_req,
                    confidence=self._confidence(score, is_req),
                    reasons=self._explain(door, row, required.get(cat)),
                ))

        return {
            "door": door,
            "recommendations": results,
            "warnings": warnings,
            "total_price": round(sum(r.price for r in results), 2),
            "constraints_applied": required,
        }


if __name__ == "__main__":
    engine = OpeningsMatch()

    print("=" * 72)
    print("CASE 1: Fire-rated stairwell door on an egress route (high security)")
    print("=" * 72)
    door = Door(fire_rated=True, is_egress=True, security_level=4,
                location="stairwell", traffic="high")
    result = engine.recommend_set(door)
    def show(result):
        for r in result["recommendations"]:
            flag = "[REQUIRED]" if r.required else "[optional]"
            print(f"\n{flag} {r.category.upper():<15} {r.name}  (${r.price})")
            print(f"     score={r.score}  confidence={r.confidence}")
            for reason in r.reasons:
                print(f"     - {reason}")
        for w in result["warnings"]:
            print(f"\n  !! WARNING: {w}")
        print(f"\nTOTAL: ${result['total_price']}")

    show(result)

    print("\n" + "=" * 72)
    print("CASE 2: Simple interior residential door (low security, not fire-rated)")
    print("=" * 72)
    door2 = Door(fire_rated=False, is_egress=False, security_level=1,
                 location="interior", traffic="standard")
    show(engine.recommend_set(door2))

    print("\n" + "=" * 72)
    print("CASE 3: THE SAFETY TEST - can the ML ever override a fire code rule?")
    print("=" * 72)
    fire = Door(fire_rated=True, security_level=2)
    permissible = engine.filter_permissible(fire, "closer")
    all_closers = engine.catalog[engine.catalog["category"] == "closer"]
    print(f"Closers in catalogue:        {len(all_closers)}")
    print(f"Permissible on a fire door:  {len(permissible)}")
    print(f"Non-fire-rated closers excluded: "
          f"{len(all_closers) - len(permissible)}")
    print("\nEvery surviving candidate is fire-rated:",
          bool(permissible['fire_rated'].all()))
    print("The similarity layer only ever sees compliant candidates.")
