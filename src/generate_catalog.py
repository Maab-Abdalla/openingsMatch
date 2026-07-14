"""
Generate a synthetic door-hardware catalogue.

IMPORTANT: This is SYNTHETIC data, generated for prototyping. It is modelled on
real door-hardware product categories and industry terminology, but the products,
brands, and prices are invented. No proprietary or real catalogue data is used.

Categories reflect standard architectural door hardware:
locks, closers, hinges, exit devices, seals, and access control.
"""

import pandas as pd
import itertools
import random
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Generic brand names - invented, not real manufacturers
BRANDS = ["Northwind", "Bastion", "Corvus", "Meridian", "Ironvale", "Kestrel"]

FINISHES = ["satin stainless", "polished brass", "matte black", "galvanised steel"]

# ---------------------------------------------------------------------------
# Product templates: (category, base_name, keywords, fire_rated, min_security,
#                     handed, price_low, price_high)
# ---------------------------------------------------------------------------
TEMPLATES = [
    # --- LOCKS ---
    ("lock", "Mortise Lockset",
     "mortise lockset heavy duty commercial cylinder keyed entry high traffic",
     True, 3, True, 180, 340),
    ("lock", "Cylindrical Lever Lock",
     "cylindrical lever lock commercial keyed entry standard duty",
     True, 2, True, 90, 170),
    ("lock", "Tubular Passage Latch",
     "tubular passage latch residential interior non locking lightweight",
     False, 1, False, 25, 55),
    ("lock", "Deadbolt Single Cylinder",
     "deadbolt single cylinder security auxiliary lock keyed",
     False, 3, False, 60, 120),
    ("lock", "Mortise Lock Fire Rated",
     "mortise lock fire rated ul listed self latching commercial stairwell",
     True, 4, True, 240, 420),

    # --- CLOSERS (fire doors legally require these) ---
    ("closer", "Surface Door Closer",
     "surface mounted door closer adjustable spring commercial self closing",
     True, 2, True, 110, 210),
    ("closer", "Concealed Overhead Closer",
     "concealed overhead door closer aesthetic hidden commercial self closing",
     True, 3, True, 260, 480),
    ("closer", "Heavy Duty Closer Fire Rated",
     "heavy duty door closer fire rated ul listed self closing high traffic",
     True, 3, True, 190, 360),
    ("closer", "Light Duty Residential Closer",
     "light duty residential door closer interior low traffic spring",
     False, 1, False, 45, 90),

    # --- HINGES ---
    ("hinge", "Ball Bearing Butt Hinge",
     "ball bearing butt hinge heavy duty steel commercial high traffic",
     True, 2, True, 20, 55),
    ("hinge", "Continuous Geared Hinge",
     "continuous geared hinge aluminium full length heavy duty high traffic",
     True, 3, True, 140, 290),
    ("hinge", "Plain Bearing Hinge",
     "plain bearing hinge residential interior light duty standard",
     False, 1, True, 8, 22),
    ("hinge", "Spring Hinge Fire Rated",
     "spring hinge fire rated self closing ul listed steel",
     True, 2, True, 30, 70),

    # --- EXIT DEVICES (required on egress routes) ---
    ("exit_device", "Rim Exit Device",
     "rim exit device panic bar egress emergency commercial push",
     True, 3, True, 320, 620),
    ("exit_device", "Mortise Exit Device",
     "mortise exit device panic hardware egress heavy duty commercial",
     True, 4, True, 420, 780),
    ("exit_device", "Vertical Rod Exit Device",
     "vertical rod exit device panic double door egress concealed",
     True, 4, True, 480, 890),

    # --- SEALS (fire doors need intumescent seals) ---
    ("seal", "Intumescent Fire Seal",
     "intumescent fire seal smoke expanding strip fire rated perimeter",
     True, 1, False, 15, 40),
    ("seal", "Acoustic Perimeter Seal",
     "acoustic perimeter seal sound insulation adhesive gasket",
     False, 1, False, 20, 50),
    ("seal", "Weather Threshold Seal",
     "weather threshold seal exterior draught water resistant aluminium",
     False, 1, False, 25, 60),

    # --- ACCESS CONTROL ---
    ("access_control", "Electronic Strike",
     "electronic strike electric release access control fail secure",
     True, 4, False, 130, 280),
    ("access_control", "Card Reader Lock",
     "card reader lock rfid electronic access control credential smart",
     True, 4, True, 340, 650),
    ("access_control", "Magnetic Lock",
     "magnetic lock maglock electromagnetic access control fail safe",
     False, 3, False, 160, 320),
]


def build_catalog() -> pd.DataFrame:
    rows = []
    pid = 1
    for (cat, base, kw, fire, sec, handed, lo, hi) in TEMPLATES:
        # Create 2 brand/finish variants of each template so the catalogue
        # has realistic near-duplicates that the recommender must rank.
        for brand, finish in itertools.islice(
            zip(random.sample(BRANDS, 2), random.sample(FINISHES, 2)), 2
        ):
            rows.append({
                "product_id": f"P{pid:03d}",
                "name": f"{brand} {base}",
                "category": cat,
                "brand": brand,
                "finish": finish,
                # description = the text the recommender vectorises
                "description": f"{kw} {finish}",
                "fire_rated": fire,
                "security_level": sec,          # 1 (low) .. 5 (high)
                "handed": handed,               # does it need left/right handing?
                "price": round(random.uniform(lo, hi), 2),
            })
            pid += 1
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_catalog()
    DATA.mkdir(exist_ok=True)
    df.to_csv(DATA / "catalog.csv", index=False)
    print(f"Generated {len(df)} products across {df['category'].nunique()} categories")
    print(df.groupby("category").size().to_string())
