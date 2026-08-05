#!/usr/bin/env python3
"""
News + AI digest -> Telegram.

Pipeline:
  fetch feeds (parallel) -> filter by time -> dedupe -> cluster near-duplicate
  headlines across outlets -> score by cross-outlet coverage -> summarise the
  top N with an LLM -> send to Telegram -> persist seen-URLs state.

Links always come from the feed itself, never from the model.

Run locally with:  python main.py --dry-run
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import feedparser
import requests

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

LOOKBACK_HOURS = 8          # slightly wider than the 6h cron, so nothing slips
MAX_CANDIDATES = 30         # clusters sent to the LLM
MAX_DIGEST_ITEMS = 10       # stories in the final message
SEEN_RETENTION_DAYS = 14

MODEL = "deepseek/deepseek-v4-flash"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

STATE_PATH = Path(__file__).parent / "state" / "seen.json"

UA = "Mozilla/5.0 (compatible; personal-news-digest/1.0; +https://github.com)"

# category: world | ai | tech  (the fallback bucket if the LLM call fails)
# weight:   baseline trust/importance of the source itself
FEEDS = [
    # --- World news -------------------------------------------------------
    {"name": "BBC World", "category": "world", "weight": 3.0,
     "url": "https://feeds.bbci.co.uk/news/world/rss.xml"},
    {"name": "BBC Tech", "category": "tech", "weight": 2.5,
     "url": "https://feeds.bbci.co.uk/news/technology/rss.xml"},
    {"name": "Reuters (via Google News)", "category": "world", "weight": 3.0,
     "url": "https://news.google.com/rss/search?q=when:1d+source:Reuters&hl=en&gl=US&ceid=US:en"},

    # --- AI labs ----------------------------------------------------------
    {"name": "OpenAI", "category": "ai", "weight": 4.0,
     "url": "https://openai.com/news/rss.xml"},
    {"name": "Google DeepMind", "category": "ai", "weight": 4.0,
     "url": "https://deepmind.google/blog/rss.xml"},
    # Anthropic publishes no official feed; community mirror + news fallback.
    {"name": "Anthropic", "category": "ai", "weight": 4.0,
     "url": "https://rsshub.bestblogs.dev/anthropic/news"},
    {"name": "Anthropic (via Google News)", "category": "ai", "weight": 2.0,
     "url": "https://news.google.com/rss/search?q=when:1d+Anthropic+Claude&hl=en&gl=US&ceid=US:en"},
    {"name": "Meta Engineering", "category": "ai", "weight": 3.0,
     "url": "https://engineering.fb.com/feed/"},

    # --- Research ---------------------------------------------------------
    {"name": "arXiv cs.AI", "category": "ai", "weight": 1.0,
     "url": "https://rss.arxiv.org/rss/cs.AI"},

    # --- Tech press -------------------------------------------------------
    {"name": "TechCrunch", "category": "tech", "weight": 2.5,
     "url": "https://techcrunch.com/feed/"},

    # --- Popularity signals (low weight: they rank stories, not break them) --
    {"name": "Hacker News", "category": "tech", "weight": 2.0,
     "url": "https://hnrss.org/frontpage?points=200"},
    {"name": "r/worldnews", "category": "world", "weight": 1.5,
     "url": "https://old.reddit.com/r/worldnews/top/.rss?t=day"},
    {"name": "r/news", "category": "world", "weight": 1.5,
     "url": "https://old.reddit.com/r/news/top/.rss?t=day"},
    {"name": "r/technology", "category": "tech", "weight": 1.5,
     "url": "https://old.reddit.com/r/technology/top/.rss?t=day"},
]

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "with", "from", "by", "as", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "he", "she", "they", "his",
    "her", "their", "we", "you", "i", "not", "no", "new", "says", "said",
    "after", "over", "into", "amid", "how", "why", "what", "who", "will",
}

TRACKING_PARAMS = re.compile(
    r"^(utm_|fbclid|gclid|oc$|ved$|at_|ns_|ito$|ref$)")


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def canonical_url(url: str) -> str:
    """Strip tracking params and fragments so the same story dedupes cleanly."""
    try:
        p = urlparse(url)
        q = [(k, v) for k, v in parse_qsl(p.query)
             if not TRACKING_PARAMS.match(k)]
        return urlunparse((p.scheme, p.netloc.lower().removeprefix("www."),
                           p.path.rstrip("/"), "", urlencode(q), ""))
    except Exception:
        return url


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def clean_title(title: str, source: str) -> str:
    """Google News appends ' - Publisher' to every headline."""
    if "Google News" in source:
        title = re.sub(r"\s+-\s+[^-]{2,40}$", "", title)
    return title.strip()


def title_tokens(title: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", title.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def entity_tokens(title: str) -> set[str]:
    """
    Tokens capitalised mid-headline: a cheap proper-noun detector. The first
    word is skipped because every headline capitalises it.

    Headlines in Title Case (common on arXiv) capitalise everything, so the
    signal is meaningless there - detect that and return nothing rather than
    flooding the vocabulary with ordinary words.
    """
    words = re.findall(r"\S+", title)
    if len(words) < 3:
        return set()
    out = set()

    # Internal capitals (OpenAI, iPhone, eBay, DeepMind) are unambiguous brand
    # names regardless of position, so take those from the whole headline.
    for w in words:
        if re.search(r"[a-z][A-Z]", w):
            tok = re.sub(r"[^A-Za-z0-9]", "", w).lower()
            if len(tok) > 2:
                out.add(tok)

    # Otherwise rely on mid-headline capitalisation. Title Case headlines
    # (common on arXiv) capitalise everything, so that signal is discarded.
    rest = words[1:]
    caps = [w for w in rest if w[:1].isupper()]
    if len(caps) / len(rest) <= 0.6:
        for w in caps:
            tok = re.sub(r"[^A-Za-z0-9]", "", w).lower()
            if len(tok) > 2 and tok not in STOPWORDS:
                out.add(tok)
    return out


def build_lexicon(items: list[dict]) -> tuple[dict[str, float], set[str]]:
    """
    Per-batch IDF weights plus a vocabulary of likely proper nouns.

    IDF is what stops generic words carrying a match. In a batch of headlines
    'earthquake' might appear eight times and 'Chile' three, so Chile ends up
    worth far more - which is exactly the distinction needed to keep two
    unrelated earthquakes apart.

    The entity vocabulary is built batch-wide, so a name only ever seen
    headline-initial in one item ('Chile hit by...') is still recognised as an
    entity there, because another headline mentioned it mid-sentence.
    """
    docs = [title_tokens(i["title"]) for i in items]
    n = len(docs)
    df: Counter[str] = Counter()
    for d in docs:
        df.update(d)
    # Smoothed IDF: never zero, so a token shared by everything still counts
    # slightly rather than vanishing.
    idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}

    vocab: set[str] = set()
    for item in items:
        vocab |= entity_tokens(item["title"])
    return idf, vocab


def tfidf_cosine(a: set[str], b: set[str], idf: dict[str, float]) -> float:
    """Cosine similarity over IDF-weighted binary term vectors."""
    shared = a & b
    if not shared:
        return 0.0
    num = sum(idf.get(t, 1.0) ** 2 for t in shared)
    na = math.sqrt(sum(idf.get(t, 1.0) ** 2 for t in a))
    nb = math.sqrt(sum(idf.get(t, 1.0) ** 2 for t in b))
    if not na or not nb:
        return 0.0
    return num / (na * nb)


# Two headlines merge if they share a proper noun AND their IDF-weighted
# vectors are close.
#
# The gated bar is deliberately low. Measurement showed the entity gate, not
# the cosine, is what separates genuinely different stories: two unrelated
# earthquakes score 0.27 but share no entity, while four outlets covering one
# earthquake score as low as 0.19 and all share "Chile". Demanding a high
# cosine on top of the gate therefore destroys recall without adding safety.
#
# When neither headline yields an entity the gate cannot apply, so a much
# stricter similarity bar substitutes for it.
COSINE_GATED = 0.20
COSINE_UNGATED = 0.58


def similar(a: set[str], b: set[str], idf: dict[str, float],
            a_ent: set[str], b_ent: set[str]) -> bool:
    if not a or not b:
        return False
    cos = tfidf_cosine(a, b, idf)
    if a_ent and b_ent:
        # Both headlines name something. If they name different things, they
        # are different stories however similar the phrasing.
        if not (a_ent & b_ent):
            return False
        return cos >= COSINE_GATED
    return cos >= COSINE_UNGATED


def entry_datetime(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        st = entry.get(key)
        if st:
            try:
                return datetime.fromtimestamp(time.mktime(st), tz=timezone.utc)
            except Exception:
                continue
    return None


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

def fetch_feed(feed: dict) -> list[dict]:
    """Fetch and normalise one feed. Never raises - a dead feed yields []."""
    try:
        resp = requests.get(feed["url"], timeout=25,
                            headers={"User-Agent": UA, "Accept": "application/rss+xml, application/xml, */*"})
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
    except Exception as exc:
        log(f"  FAIL {feed['name']}: {type(exc).__name__}: {exc}")
        return []

    if not parsed.entries:
        log(f"  EMPTY {feed['name']} (feed returned no entries)")
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    items = []
    for entry in parsed.entries:
        link = entry.get("link")
        title = entry.get("title")
        if not link or not title:
            continue

        published = entry_datetime(entry)
        # Feeds without dates (some mirrors) are kept; state dedupe protects us.
        if published and published < cutoff:
            continue

        summary = strip_html(entry.get("summary", ""))[:400]
        points = 0
        m = re.search(r"Points:\s*(\d+)", entry.get("summary", ""))
        if m:
            points = int(m.group(1))

        items.append({
            "title": clean_title(title, feed["name"]),
            "url": link,
            "canonical": canonical_url(link),
            "source": feed["name"],
            "category": feed["category"],
            "weight": feed["weight"],
            "published": published,
            "summary": summary,
            "points": points,
        })

    log(f"  ok   {feed['name']}: {len(items)} recent / {len(parsed.entries)} total")
    return items


def fetch_all() -> list[dict]:
    log(f"Fetching {len(FEEDS)} feeds...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(fetch_feed, FEEDS))
    items = [item for batch in results for item in batch]
    log(f"Collected {len(items)} items total")
    return items


# --------------------------------------------------------------------------
# Dedupe, cluster, score
# --------------------------------------------------------------------------

def load_seen() -> dict[str, str]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            log("state file unreadable, starting fresh")
    return {}


def save_seen(seen: dict[str, str]) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=SEEN_RETENTION_DAYS)
    pruned = {}
    for url, ts in seen.items():
        try:
            if datetime.fromisoformat(ts) >= cutoff:
                pruned[url] = ts
        except Exception:
            continue
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(pruned, indent=0, sort_keys=True))
    log(f"state: {len(pruned)} urls retained")


def cluster(items: list[dict]) -> list[dict]:
    """Group near-identical headlines. Cross-outlet coverage = importance."""
    idf, entity_vocab = build_lexicon(items)

    clusters: list[dict] = []
    for item in items:
        tokens = title_tokens(item["title"])
        # An item's entities are its own capitalised words, plus any of its
        # tokens the batch has seen capitalised elsewhere.
        ents = (entity_tokens(item["title"]) | (tokens & entity_vocab))
        placed = False
        for c in clusters:
            # Compare against each member separately. Merging token sets would
            # dilute the similarity score as a cluster grows, so the third
            # outlet to cover a story would fail to match.
            if any(similar(tokens, t, idf, ents, e)
                   for t, e in zip(c["token_sets"], c["entity_sets"])):
                c["items"].append(item)
                c["token_sets"].append(tokens)
                c["entity_sets"].append(ents)
                placed = True
                break
        if not placed:
            clusters.append({"token_sets": [tokens], "entity_sets": [ents],
                             "items": [item]})

    out = []
    for idx, c in enumerate(clusters):
        # Prefer the highest-weight source as the canonical link for the story.
        members = sorted(c["items"], key=lambda i: -i["weight"])
        lead = members[0]
        sources = sorted({m["source"] for m in members})
        points = max((m["points"] for m in members), default=0)

        score = lead["weight"]
        score += 3.0 * (len(sources) - 1)          # cross-outlet coverage
        score += min(points / 150.0, 3.0)          # HN votes, capped
        if lead["published"]:
            age_h = (datetime.now(timezone.utc) -
                     lead["published"]).total_seconds() / 3600
            score += max(0.0, 2.0 - age_h / 4.0)   # mild recency bonus

        out.append({
            "id": idx,
            "title": lead["title"],
            "url": lead["url"],
            "category": lead["category"],
            "summary": lead["summary"],
            "sources": sources,
            "member_titles": [mm["title"] for mm in members],
            "points": points,
            "score": round(score, 2),
            "all_urls": [m["canonical"] for m in members],
        })

    out.sort(key=lambda c: -c["score"])
    return out


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a news editor writing a concise digest in English.

You receive candidate stories as JSON. Each has an "id", a headline, a raw feed
snippet, and the outlets that covered it.

Select the most consequential stories, up to the limit given. Prefer:
  - stories covered by several independent outlets
  - substantive AI/technology developments over incremental product news
  - genuine world events over opinion, sport, and celebrity items
Drop anything trivial, duplicated, or purely promotional. Returning fewer items
than the limit is correct when the material is thin.

Keep every summary to 1-2 sentences, under 300 characters. Never invent facts
beyond what the snippet supports. If a snippet is too thin to summarise, write a
summary that only restates the headline's claim. Never output URLs."""

# Schema-enforced output. The model cannot emit prose, markdown fences, or
# malformed JSON, which is what a plain "respond with JSON" instruction kept
# producing. OpenRouter requires the schema root to be an object, hence the
# "stories" wrapper.
RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "digest",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "stories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "headline": {"type": "string"},
                            "summary": {"type": "string"},
                            "category": {"type": "string",
                                         "enum": ["world", "ai", "tech"]},
                        },
                        "required": ["id", "headline", "summary", "category"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["stories"],
            "additionalProperties": False,
        },
    },
}


def salvage_objects(text: str) -> list[dict]:
    """
    Pull every complete JSON object out of a possibly-truncated response.

    If generation is cut off mid-object, json.loads fails on the whole payload
    and nine good stories are lost along with the tenth partial one. Decoding
    object-by-object keeps whatever finished.
    """
    decoder = json.JSONDecoder()
    out, i = [], 0
    while i < len(text):
        start = text.find("{", i)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
            if isinstance(obj, dict):
                out.append(obj)
            i = end
        except ValueError:
            i = start + 1
    return out


def summarise(candidates: list[dict], api_key: str) -> list[dict] | None:
    payload = [{
        "id": c["id"],
        "headline": c["title"],
        "snippet": c["summary"][:300],
        "outlets": c["sources"],
    } for c in candidates]

    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                f"Limit: {MAX_DIGEST_ITEMS} stories.\n\n{json.dumps(payload, ensure_ascii=False)}"},
        ],
        "temperature": 0.3,
        # Generous ceiling: reasoning-capable models spend part of this budget
        # on hidden thinking tokens before emitting any JSON, so a tight limit
        # truncates the visible answer.
        "max_tokens": 8000,
        "response_format": RESPONSE_SCHEMA,
    }

    for attempt in range(3):
        try:
            resp = requests.post(
                OPENROUTER_URL, timeout=180,
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json",
                         "X-Title": "news-digest"},
                json=body,
            )
            if resp.status_code != 200:
                log(f"LLM HTTP {resp.status_code}: {resp.text[:300]}")
                resp.raise_for_status()

            choice = resp.json()["choices"][0]
            finish = choice.get("finish_reason")
            text = (choice["message"].get("content") or "").strip()
            usage = resp.json().get("usage", {})
            log(f"LLM finish_reason={finish} chars={len(text)} usage={usage}")

            if finish == "length":
                log("  response hit the token ceiling - salvaging what completed")

            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()

            stories = None
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    stories = data.get("stories")
                elif isinstance(data, list):
                    stories = data
            except json.JSONDecodeError:
                salvaged = [o for o in salvage_objects(text) if "id" in o]
                if salvaged:
                    log(f"  recovered {len(salvaged)} complete objects from malformed JSON")
                    stories = salvaged

            if isinstance(stories, list) and stories:
                return stories
            if isinstance(stories, list):
                log("LLM returned an empty list")
                return []
            log("  could not parse a story list from the response")
            raise ValueError("unparseable response")

        except Exception as exc:
            log(f"LLM attempt {attempt + 1} failed: {type(exc).__name__}: {exc}")
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
    return None


# --------------------------------------------------------------------------
# Telegram
# --------------------------------------------------------------------------

EMOJI = {"world": "\U0001F30D", "ai": "\U0001F9E0", "tech": "\U0001F4BB"}
LABEL = {"world": "WORLD", "ai": "AI", "tech": "TECH"}


def build_message(selected: list[dict], by_id: dict[int, dict]) -> str:
    header = f"<b>Digest</b> \u00b7 {datetime.now(timezone.utc):%d %b %H:%M} UTC\n"
    blocks = []
    for cat in ("world", "ai", "tech"):
        rows = [s for s in selected if s.get("category") == cat]
        if not rows:
            continue
        lines = [f"\n{EMOJI[cat]} <b>{LABEL[cat]}</b>"]
        for s in rows:
            cluster_item = by_id.get(s["id"])
            if not cluster_item:
                continue
            url = html.escape(cluster_item["url"], quote=True)
            title = html.escape(s.get("headline") or cluster_item["title"])
            summary = html.escape(s.get("summary", ""))
            outlets = ", ".join(cluster_item["sources"][:3])
            if len(cluster_item["sources"]) > 3:
                outlets += f" +{len(cluster_item['sources']) - 3}"
            extra = f" \u00b7 {cluster_item['points']}pts" if cluster_item["points"] else ""
            lines.append(f'\n\u2022 <a href="{url}">{title}</a>\n'
                         f'{summary}\n'
                         f'<i>{html.escape(outlets)}{extra}</i>')
        blocks.append("\n".join(lines))
    return header + "\n".join(blocks)


def chunk(text: str, limit: int = 3900) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for para in text.split("\n\n"):
        if len(current) + len(para) + 2 > limit:
            parts.append(current.strip())
            current = para
        else:
            current += "\n\n" + para
    if current.strip():
        parts.append(current.strip())
    return parts


def send_telegram(text: str, token: str, chat_id: str) -> bool:
    ok = True
    for part in chunk(text):
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage", timeout=30,
                json={"chat_id": chat_id, "text": part, "parse_mode": "HTML",
                      "disable_web_page_preview": True},
            )
            if resp.status_code != 200:
                log(f"telegram error {resp.status_code}: {resp.text[:300]}")
                ok = False
            time.sleep(0.5)
        except Exception as exc:
            log(f"telegram send failed: {exc}")
            ok = False
    return ok


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the digest instead of sending it; state is not written")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID")

    if not args.dry_run and not all([api_key, tg_token, tg_chat]):
        log("ERROR: set OPENROUTER_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")
        return 1

    items = fetch_all()
    if not items:
        log("no items fetched at all - check network or feed URLs")
        return 0

    seen = load_seen()
    fresh = [i for i in items if i["canonical"] not in seen]
    log(f"{len(fresh)} unseen items ({len(items) - len(fresh)} already sent)")
    if not fresh:
        log("nothing new, exiting quietly")
        return 0

    clusters = cluster(fresh)
    candidates = clusters[:MAX_CANDIDATES]
    log(f"{len(clusters)} clusters, sending top {len(candidates)} to {MODEL}")

    by_id = {c["id"]: c for c in candidates}
    selected = summarise(candidates, api_key) if api_key else None

    if selected is None:
        # Model unreachable: degrade to raw headlines rather than sending nothing.
        log("falling back to unsummarised headlines")
        selected = [{"id": c["id"], "headline": c["title"],
                     "summary": c["summary"][:200],
                     "category": c["category"]}
                    for c in candidates[:MAX_DIGEST_ITEMS]]

    selected = [s for s in selected if isinstance(
        s, dict) and s.get("id") in by_id]
    if not selected:
        log("model selected nothing worth sending")
        return 0

    message = build_message(selected, by_id)

    if args.dry_run:
        print("\n" + "=" * 70)
        print(message)
        print("=" * 70)
        return 0

    if send_telegram(message, tg_token, tg_chat):
        for s in selected:
            for url in by_id[s["id"]]["all_urls"]:
                seen[url] = datetime.now(timezone.utc).isoformat()
        # Mark every candidate seen, not just the sent ones, so rejected
        # stories don't get re-evaluated on every subsequent run.
        for c in candidates:
            for url in c["all_urls"]:
                seen.setdefault(url, datetime.now(timezone.utc).isoformat())
        save_seen(seen)
        log(f"sent {len(selected)} stories")
        return 0

    log("send failed; state left untouched so the next run retries")
    return 1


if __name__ == "__main__":
    sys.exit(main())
