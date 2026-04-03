"""
Synthetic task generation for ContextStress benchmark.

Generates 400 task instances (100 per family T1-T4) using synthetic facts
from a controlled vocabulary. All entities, relationships, and attributes
are constructed to eliminate parametric knowledge confounds.
"""

import json
import random
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ── Controlled Vocabulary ────────────────────────────────────────────────

FIRST_NAMES = [
    "Vorath", "Kestine", "Oluvian", "Rendahl", "Pavique", "Jessamin",
    "Thandor", "Lirenne", "Colvane", "Mythra", "Delquin", "Surenna",
    "Baxthor", "Fenilly", "Gravish", "Helquist", "Iridon", "Jarquelle",
    "Kymber", "Lorthane", "Moriven", "Nelvaine", "Orquist", "Prendel",
    "Quelthis", "Rovaine", "Selvaran", "Trevique", "Uldarine", "Vexholm",
]

LAST_NAMES = [
    "Wrenthall", "Blackmere", "Stormfeld", "Quinvale", "Drakhelm",
    "Ashgrove", "Ironlace", "Thornwick", "Goldmantle", "Silverbrook",
    "Brightmoor", "Deepwell", "Frostwick", "Greenvale", "Hawkridge",
    "Kingsward", "Longmere", "Northcross", "Oakvane", "Redfield",
    "Southgate", "Westholm", "Highcastle", "Lowmere", "Midvale",
    "Eastbrook", "Stonepath", "Cloudpeak", "Riverdale", "Moorwick",
]

ORGS = [
    "Nexvara Institute", "Coltrane Foundation", "Meridian Collective",
    "Arbelos Corporation", "Synthetik Labs", "Vanguard Consortium",
    "Prismatic Alliance", "Heliograph Trust", "Catalon Industries",
    "Zenthera Group", "Orvalis Network", "Spectral Dynamics",
    "Luminos Federation", "Arcturus Holdings", "Parallax Systems",
    "Tessera Ventures", "Quorum Analytics", "Ecliptic Research",
    "Nomadic Sciences", "Bastion Technologies", "Veritas Council",
    "Epoch Enterprises", "Frontier Mechanics", "Helion Partners",
    "Ignition Works", "Jovian Labs", "Kinetic Solutions",
    "Lattice Dynamics", "Manifold Corp", "Nexus Protocols",
]

PLACES = [
    "Thornhaven", "Azurmere", "Cascadel", "Dawnport", "Evernight",
    "Feldspar Ridge", "Glimwick", "Havenreach", "Ironvale", "Jaspertown",
    "Kingspire", "Luminara", "Moontide", "Nethervale", "Obsidian Bluff",
    "Pinecrest", "Quartzhollow", "Ravenspire", "Silverdale", "Tidefall",
]

LANGUAGES = [
    "Vorenthi", "Caltrusian", "Meridese", "Arbelic", "Synthari",
    "Prismaic", "Heliosi", "Cataloni", "Zentheran", "Orvali",
]

YEARS = list(range(1847, 2019))

COMPOUNDS = [f"Compound {chr(65+i)}-{j}" for i in range(26) for j in range(1, 6)]

PROGRAMS = [
    "Kepler-III", "Argon Initiative", "Helix Protocol", "Zenith Project",
    "Nova Cascade", "Titan Mapping", "Stellar Drift", "Ion Harvest",
    "Quantum Lattice", "Solar Veil", "Dark Current", "Phase Bridge",
    "Echo Traverse", "Nebula Core", "Proton Bloom", "Vertex Dawn",
    "Cipher Wave", "Axiom Gate", "Nimbus Array", "Corona Study",
]

TRANSIT_SYSTEMS = [
    "Metro Arcanis", "Skybridge Loop", "Subterrane Express",
    "Coastal Maglev", "Highland Monorail", "Valley Hyperloop",
    "Urban Gondola", "River Shuttle", "Canyon Tram", "Plains Rail",
]


@dataclass
class SignalPassage:
    """A passage containing a single fact needed for the task."""
    content: str
    fact: str
    step: int  # Which reasoning step this supports (1-indexed)


@dataclass
class TaskInstance:
    """A single benchmark task instance."""
    id: str
    family: str  # T1, T2, T3, T4
    depth: int  # 1, 2, 3, or 4
    query: str
    answer: str
    reasoning_chain: list[str]
    signal_passages: list[dict]
    aliases: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _seeded_rng(seed: int) -> random.Random:
    return random.Random(seed)


def _make_passage(fact: str, rng: random.Random, padding_sentences: int = 5) -> str:
    """Wrap a fact in a realistic-looking passage with padding context."""
    fillers = [
        "This was documented in the official records of the period.",
        "Several independent sources have confirmed this information.",
        "The details were later published in a comprehensive report.",
        "Historians have noted the significance of this development.",
        "Further analysis revealed additional context surrounding this.",
        "Contemporary accounts describe the circumstances in detail.",
        "The implications of this were widely discussed at the time.",
        "Archival materials provide extensive documentation on this matter.",
        "This finding was subsequently verified through multiple channels.",
        "The broader context illuminates the importance of this detail.",
    ]
    before = rng.sample(fillers, min(padding_sentences // 2, len(fillers)))
    after = rng.sample(
        [f for f in fillers if f not in before],
        min(padding_sentences - len(before), len(fillers) - len(before)),
    )
    return " ".join(before) + " " + fact + " " + " ".join(after)


def generate_t1(instance_id: int, rng: random.Random) -> TaskInstance:
    """Generate a single-hop retrieval task (d=1)."""
    person = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    org = rng.choice(ORGS)
    year = rng.choice(YEARS)
    event = rng.choice([
        f"the {person} Accord was signed",
        f"{org} was officially established",
        f"the treaty between {person} and {org} was ratified",
        f"{person} published the foundational report on {rng.choice(PROGRAMS)}",
    ])
    fact = f"In {year}, {event}."
    query_variants = [
        f"In what year was {event.replace(f'In {year}, ', '')}?",
        f"When did {event.replace(f'In {year}, ', '')} occur?",
    ]
    query = rng.choice(query_variants)
    passage = _make_passage(fact, rng)

    return TaskInstance(
        id=f"T1_{instance_id:03d}",
        family="T1",
        depth=1,
        query=query,
        answer=str(year),
        reasoning_chain=[f"The passage states: '{fact}' → Answer: {year}"],
        signal_passages=[{"content": passage, "fact": fact, "step": 1}],
        aliases=[str(year)],
    )


def generate_t2(instance_id: int, rng: random.Random) -> TaskInstance:
    """Generate a two-hop reasoning task (d=2)."""
    person = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    org = rng.choice(ORGS)
    program = rng.choice(PROGRAMS)
    place = rng.choice(PLACES)

    # Fact 1: org funded program
    fact1 = f"{org} provided primary funding for the {program} satellite program."
    # Fact 2: person directed org
    fact2 = f"{person} served as the director of {org} from {rng.choice(YEARS)} to {rng.choice(YEARS)}."

    query = f"Who directed the organization that funded the {program} satellite program?"
    answer = person

    passage1 = _make_passage(fact1, rng)
    passage2 = _make_passage(fact2, rng)

    return TaskInstance(
        id=f"T2_{instance_id:03d}",
        family="T2",
        depth=2,
        query=query,
        answer=answer,
        reasoning_chain=[
            f"Step 1: {org} funded {program}. (from passage)",
            f"Step 2: {person} directed {org}. (from passage) → Answer: {person}",
        ],
        signal_passages=[
            {"content": passage1, "fact": fact1, "step": 1},
            {"content": passage2, "fact": fact2, "step": 2},
        ],
        aliases=[person, person.split()[-1]],
    )


def generate_t3(instance_id: int, rng: random.Random) -> TaskInstance:
    """Generate a four-hop reasoning task (d=4)."""
    person = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    language = rng.choice(LANGUAGES)
    place = rng.choice(PLACES)
    org1 = rng.choice(ORGS)
    orgs_remaining = [o for o in ORGS if o != org1]
    org2 = rng.choice(orgs_remaining)
    compound = rng.choice(COMPOUNDS)

    # Chain: language ← person ← designed HQ of org1 ← acquired org2 ← held patent on compound
    fact1 = f"{org2} held the original patent for {compound}."
    fact2 = f"{org1} acquired {org2} in a landmark merger in {rng.choice(YEARS)}."
    fact3 = f"The headquarters of {org1} in {place} was designed by architect {person}."
    fact4 = f"{person} was a native speaker of {language}, having grown up in the {language}-speaking region."

    query = (
        f"What is the native language of the architect who designed the headquarters "
        f"of the corporation that acquired the patent holder of {compound}?"
    )
    answer = language

    return TaskInstance(
        id=f"T3_{instance_id:03d}",
        family="T3",
        depth=4,
        query=query,
        answer=answer,
        reasoning_chain=[
            f"Step 1: {org2} held the patent for {compound}.",
            f"Step 2: {org1} acquired {org2}.",
            f"Step 3: {person} designed the HQ of {org1}.",
            f"Step 4: {person} spoke {language}. → Answer: {language}",
        ],
        signal_passages=[
            {"content": _make_passage(fact1, rng), "fact": fact1, "step": 1},
            {"content": _make_passage(fact2, rng), "fact": fact2, "step": 2},
            {"content": _make_passage(fact3, rng), "fact": fact3, "step": 3},
            {"content": _make_passage(fact4, rng), "fact": fact4, "step": 4},
        ],
        aliases=[language],
    )


def generate_t4(instance_id: int, rng: random.Random) -> TaskInstance:
    """Generate a comparative reasoning task (d=3)."""
    systems = rng.sample(TRANSIT_SYSTEMS, 3)
    costs = [rng.randint(500, 5000) for _ in range(3)]  # millions
    riderships = [rng.randint(10000, 500000) for _ in range(3)]  # daily
    ratios = [r / c for r, c in zip(riderships, costs)]
    best_idx = ratios.index(max(ratios))
    answer = systems[best_idx]

    facts = []
    passages = []
    for i, (sys, cost, rider) in enumerate(zip(systems, costs, riderships)):
        fact = (
            f"The {sys} project has a projected construction cost of "
            f"${cost} million and an estimated daily ridership of {rider:,} passengers."
        )
        facts.append(fact)
        passages.append({
            "content": _make_passage(fact, rng),
            "fact": fact,
            "step": i + 1,
        })

    query = (
        f"Which of the three proposed transit systems — {systems[0]}, "
        f"{systems[1]}, or {systems[2]} — has the highest projected "
        f"ridership per dollar of construction cost?"
    )

    return TaskInstance(
        id=f"T4_{instance_id:03d}",
        family="T4",
        depth=3,
        query=query,
        answer=answer,
        reasoning_chain=[
            f"Step 1: Gather data for all three systems.",
            f"Step 2: Compute ratios: "
            + ", ".join(
                f"{s}: {r:,}/{c}M = {r/c:.1f}"
                for s, r, c in zip(systems, riderships, costs)
            ),
            f"Step 3: {answer} has the highest ratio. → Answer: {answer}",
        ],
        signal_passages=passages,
        aliases=[answer],
    )


GENERATORS = {"T1": generate_t1, "T2": generate_t2, "T3": generate_t3, "T4": generate_t4}


def generate_all_tasks(
    seed: int = 42,
    instances_per_family: int = 100,
    output_dir: Optional[Path] = None,
) -> list[TaskInstance]:
    """Generate the full benchmark task set (400 instances by default)."""
    rng = _seeded_rng(seed)
    tasks = []

    for family, gen_fn in GENERATORS.items():
        for i in range(instances_per_family):
            # Each instance gets a deterministic sub-seed
            sub_seed = int(
                hashlib.md5(f"{seed}_{family}_{i}".encode()).hexdigest()[:8], 16
            )
            instance_rng = _seeded_rng(sub_seed)
            task = gen_fn(i, instance_rng)
            tasks.append(task)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for family in GENERATORS:
            family_tasks = [t for t in tasks if t.family == family]
            path = output_dir / f"{family.lower()}_tasks.json"
            with open(path, "w") as f:
                json.dump([t.to_dict() for t in family_tasks], f, indent=2)

        # Also write combined file
        with open(output_dir / "all_tasks.json", "w") as f:
            json.dump([t.to_dict() for t in tasks], f, indent=2)

    return tasks


if __name__ == "__main__":
    tasks = generate_all_tasks(output_dir=Path(__file__).parent)
    print(f"Generated {len(tasks)} tasks:")
    for family in ["T1", "T2", "T3", "T4"]:
        count = sum(1 for t in tasks if t.family == family)
        print(f"  {family}: {count} instances (d={tasks[[t.family for t in tasks].index(family)].depth})")
