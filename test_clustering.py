"""Clustering regression tests. Run: python test_clustering.py"""
from datetime import datetime, timezone
import main as m

NOW = datetime.now(timezone.utc)


def it(title, source="BBC World", weight=3.0):
    h = str(abs(hash(title)) % 99999)
    return {"title": title, "url": "https://x.test/" + h, "canonical": "x.test/" + h,
            "source": source, "category": "world", "weight": weight,
            "published": NOW, "summary": "", "points": 0}


BATCH = [
    # same story, four outlets -> must merge
    it("Chile hit by magnitude 6.2 earthquake near Valparaiso", "BBC World"),
    it("Magnitude 6.2 earthquake strikes central Chile", "Reuters (via Google News)"),
    it("Strong earthquake shakes Chile, no tsunami warning issued", "r/worldnews", 1.5),
    it("Chile earthquake: buildings damaged in Valparaiso region", "BBC Tech", 2.5),
    # different earthquake, same wording pattern -> must NOT merge
    it("Magnitude 5.8 earthquake strikes northern Japan", "Reuters (via Google News)"),
    it("Japan earthquake prompts brief evacuation order", "r/news", 1.5),
    # ADVERSARIAL: same entity, unrelated story -> must NOT merge
    it("Chile announces new lithium mining investment programme", "BBC World"),
    # same story, heavy paraphrase -> should merge
    it("EU fines Meta over targeted advertising practices", "BBC Tech", 2.5),
    it("Meta hit with EU penalty in advertising case", "TechCrunch", 2.5),
    # ADVERSARIAL: same entity, unrelated story -> must NOT merge
    it("Meta unveils updated virtual reality headset", "TechCrunch", 2.5),
    # two distinct OpenAI stories -> must NOT merge (tests camelCase entity)
    it("OpenAI launches new reasoning model for enterprise", "OpenAI", 4.0),
    it("OpenAI announces changes to developer pricing tiers", "OpenAI", 4.0),
    it("Rivals respond to OpenAI reasoning model launch", "TechCrunch", 2.5),
    # entity-free duplicates -> ungated path
    it("Ceasefire agreed in long-running border conflict", "BBC World"),
    it("Ceasefire agreed after border conflict talks conclude", "r/worldnews", 1.5),
    # filler, all distinct
    it("Global markets close higher on rate cut expectations", "BBC World"),
    it("Researchers report progress on protein folding models", "arXiv cs.AI", 1.0),
    it("Startup raises funding round for logistics software", "TechCrunch", 2.5),
    it("Health officials warn of rising flu cases this winter", "BBC World"),
    it("Wildfires force evacuations across southern Europe", "BBC World"),
    it("Central bank holds interest rates steady", "BBC World"),
    it("New undersea cable project links Baltic states", "BBC Tech", 2.5),
    it("Scientists describe new deep sea species", "BBC World"),
    it("Airline cancels flights after system outage", "BBC World"),
    it("Report finds gaps in cybersecurity readiness", "TechCrunch", 2.5),
]

FAILURES = []


def check(name, cond):
    print(("  PASS  " if cond else "  FAIL  ") + name)
    if not cond:
        FAILURES.append(name)


def members(clusters, *words):
    """Clusters whose lead title or any member matches all given words."""
    out = []
    for c in clusters:
        blob = " ".join(t.lower() for t in c["member_titles"])
        if all(w.lower() in blob for w in words):
            out.append(c)
    return out


def run():
    clusters = m.cluster(BATCH)
    print(f"{len(BATCH)} headlines -> {len(clusters)} clusters\n")
    for c in sorted(clusters, key=lambda x: -len(x["sources"])):
        mark = "*" if len(c["sources"]) > 1 else " "
        print(f" {mark} [{len(c['sources'])}] {c['title'][:60]}")
        for t in c["member_titles"][1:]:
            print(f"       + {t[:56]}")
    print()

    quake = [c for c in clusters if "earthquake" in " ".join(c["member_titles"]).lower()
             and "chile" in " ".join(c["member_titles"]).lower()]
    check("4 Chile quake headlines in one cluster",
          len(quake) == 1 and len(quake[0]["sources"]) == 4)
    check("Japan quake separate from Chile quake",
          len(members(clusters, "japan")) >= 1
          and all("chile" not in " ".join(c["member_titles"]).lower()
                  for c in members(clusters, "japan")))
    check("Chile mining story NOT merged into Chile quake",
          any("lithium" in " ".join(c["member_titles"]).lower()
              and "earthquake" not in " ".join(c["member_titles"]).lower()
              for c in clusters))
    check("EU/Meta fine paraphrase merged",
          any(len(c["sources"]) == 2 and "penalty" in " ".join(c["member_titles"]).lower()
              for c in clusters))
    check("Meta headset NOT merged into Meta fine",
          any("headset" in " ".join(c["member_titles"]).lower()
              and "penalty" not in " ".join(c["member_titles"]).lower()
              for c in clusters))
    check("OpenAI pricing story stays apart from model launch",
          any("pricing" in " ".join(c["member_titles"]).lower()
              and "reasoning" not in " ".join(c["member_titles"]).lower()
              for c in clusters))
    check("entity-free duplicates merged via ungated path",
          any(len(c["sources"]) == 2 and "ceasefire" in c["title"].lower()
              for c in clusters))

    print(f"\n{len(FAILURES)} failure(s): {FAILURES}" if FAILURES else "\nAll checks passed.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(run())
