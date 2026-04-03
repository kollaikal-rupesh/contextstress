"""
Noise corpus generation for ContextStress benchmark.

Generates 10,000 passages organized into 20 topical clusters.
Each passage is ~500 tokens of topically coherent but task-irrelevant text.
"""

import json
import random
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


CLUSTER_TOPICS = [
    "marine biology and oceanography",
    "medieval European history",
    "quantum computing research",
    "agricultural economics",
    "classical music composition",
    "volcanic geology and tectonics",
    "urban transportation planning",
    "infectious disease epidemiology",
    "Renaissance art and architecture",
    "cryptocurrency and blockchain",
    "atmospheric chemistry",
    "cognitive neuroscience",
    "textile manufacturing processes",
    "space exploration missions",
    "labor economics and policy",
    "ancient Mesopotamian civilization",
    "renewable energy systems",
    "pharmaceutical drug development",
    "documentary filmmaking",
    "sports analytics and statistics",
]

# Template sentences per cluster (used to generate varied passages)
_TEMPLATES = {i: [
    f"Research in {topic} has advanced significantly in recent decades.",
    f"Scholars studying {topic} have identified several key patterns.",
    f"The field of {topic} draws on multiple disciplinary perspectives.",
    f"Funding for {topic} has fluctuated with changing policy priorities.",
    f"New methodologies in {topic} are transforming established approaches.",
    f"International collaboration in {topic} has yielded important results.",
    f"The historical development of {topic} reveals complex trajectories.",
    f"Debates within {topic} often center on foundational assumptions.",
    f"Practical applications of {topic} continue to expand in scope.",
    f"Training the next generation of experts in {topic} remains critical.",
    f"Data collection methods in {topic} vary across institutions.",
    f"Recent publications on {topic} have challenged conventional wisdom.",
    f"The intersection of {topic} with technology opens new possibilities.",
    f"Public understanding of {topic} has improved through outreach efforts.",
    f"Ethical considerations in {topic} are receiving increased attention.",
    f"Regional variations in {topic} reflect diverse environmental contexts.",
    f"Long-term studies in {topic} provide valuable longitudinal data.",
    f"Cross-cultural perspectives on {topic} enrich the overall discourse.",
    f"The economic impact of {topic} is difficult to quantify precisely.",
    f"Future directions in {topic} will likely involve interdisciplinary work.",
] for i, topic in enumerate(CLUSTER_TOPICS)}


@dataclass
class NoisePassage:
    """A noise passage from the corpus."""
    id: str
    cluster_id: int
    cluster_topic: str
    content: str
    approx_tokens: int


def _seeded_rng(seed: int) -> random.Random:
    return random.Random(seed)


def _generate_passage(cluster_id: int, passage_idx: int, rng: random.Random) -> str:
    """Generate a ~500-token noise passage for a given cluster."""
    templates = _TEMPLATES[cluster_id]
    topic = CLUSTER_TOPICS[cluster_id]

    # Build passage from shuffled template sentences + elaborations
    selected = rng.sample(templates, min(8, len(templates)))
    elaborations = [
        f"This aspect of {topic} warrants further investigation.",
        f"Multiple studies have corroborated these findings about {topic}.",
        f"The relationship between these factors in {topic} is nuanced.",
        f"Experts disagree on the relative importance of various {topic} components.",
        f"Additional context from related fields helps illuminate {topic} dynamics.",
        f"Quantitative analysis of {topic} data reveals interesting trends.",
        f"Case studies in {topic} demonstrate the diversity of outcomes.",
        f"Theoretical frameworks for understanding {topic} continue to evolve.",
        f"The policy implications of {topic} research are significant.",
        f"Comparative analyses across regions shed light on {topic} variations.",
        f"Primary sources on {topic} are scattered across multiple archives.",
        f"Peer review in {topic} maintains standards of scientific rigor.",
    ]
    extra = rng.sample(elaborations, min(6, len(elaborations)))

    sentences = selected + extra
    rng.shuffle(sentences)

    # Aim for roughly 500 tokens (~375 words at 1.33 tokens/word)
    passage = " ".join(sentences)

    # Pad if too short
    while len(passage.split()) < 350:
        filler = rng.choice([
            f"Furthermore, the study of {topic} benefits from new technologies.",
            f"In conclusion, {topic} remains a vibrant area of inquiry.",
            f"These developments in {topic} have far-reaching implications.",
            f"The evidence base for {topic} continues to grow steadily.",
            f"Methodological rigor in {topic} is essential for valid conclusions.",
        ])
        passage += " " + filler

    return passage


def generate_noise_corpus(
    seed: int = 42,
    num_passages: int = 10000,
    num_clusters: int = 20,
    output_dir: Optional[Path] = None,
) -> list[NoisePassage]:
    """Generate the full noise corpus."""
    rng = _seeded_rng(seed)
    passages_per_cluster = num_passages // num_clusters
    corpus = []

    for cluster_id in range(num_clusters):
        for j in range(passages_per_cluster):
            sub_seed = int(
                hashlib.md5(f"{seed}_noise_{cluster_id}_{j}".encode()).hexdigest()[:8], 16
            )
            p_rng = _seeded_rng(sub_seed)
            content = _generate_passage(cluster_id, j, p_rng)

            passage = NoisePassage(
                id=f"noise_{cluster_id:02d}_{j:04d}",
                cluster_id=cluster_id,
                cluster_topic=CLUSTER_TOPICS[cluster_id],
                content=content,
                approx_tokens=len(content.split()) * 4 // 3,  # rough estimate
            )
            corpus.append(passage)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write per-cluster files
        for cluster_id in range(num_clusters):
            cluster_passages = [p for p in corpus if p.cluster_id == cluster_id]
            path = output_dir / f"cluster_{cluster_id:02d}.json"
            with open(path, "w") as f:
                json.dump([asdict(p) for p in cluster_passages], f, indent=2)

        # Write manifest
        manifest = {
            "num_passages": len(corpus),
            "num_clusters": num_clusters,
            "passages_per_cluster": passages_per_cluster,
            "cluster_topics": {i: t for i, t in enumerate(CLUSTER_TOPICS)},
            "seed": seed,
        }
        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    return corpus


if __name__ == "__main__":
    corpus = generate_noise_corpus(output_dir=Path(__file__).parent)
    print(f"Generated {len(corpus)} noise passages across {len(CLUSTER_TOPICS)} clusters")
