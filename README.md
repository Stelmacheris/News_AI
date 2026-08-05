# News + AI digest → Telegram

Pulls ~14 RSS feeds every 6 hours, clusters the same story across outlets, ranks
by how many independent outlets picked it up, summarises the top 10 with
DeepSeek V4 Flash, and sends one Telegram message.

## Setup

### 1. Telegram bot

1. Message [@BotFather](https://t.me/BotFather) → `/newbot` → copy the token.
2. Send your new bot any message (it can't message you first).
3. Get your chat ID:
   ```
   curl "https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates"
   ```
   Read `result[0].message.chat.id` — a number, negative for groups.

### 2. OpenRouter

Create a key at [openrouter.ai/keys](https://openrouter.ai/keys) and **add a few
dollars of credit**. DeepSeek has no free tier on OpenRouter, so a $0 balance
returns 402 errors. Expected spend is about **$0.15/month**: ~8k input tokens per
run × 4 runs/day at $0.09/M input, $0.18/M output.

To use a free model instead, change `MODEL` in `main.py` to `"openrouter/free"`
(auto-router across free models) — quality drops, but cost is zero.

### 3. Repo secrets

Settings → Secrets and variables → Actions → New repository secret:

| Secret | Value |
|---|---|
| `OPENROUTER_API_KEY` | `sk-or-v1-...` |
| `TELEGRAM_BOT_TOKEN` | `1234567:AA...` |
| `TELEGRAM_CHAT_ID` | `123456789` |

### 4. Run it

Push, then Actions tab → `news-digest` → **Run workflow** to test immediately.
The schedule takes over after that.

## Local testing

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-v1-...
python main.py --dry-run     # prints to stdout, sends nothing, writes no state
```

`--dry-run` also works without an API key — it falls back to raw headlines, which
is a fast way to check the feeds are alive.

## How ranking works

There is no "most read" API, so importance is inferred:

- **Cross-outlet coverage** is the main signal. If BBC, Reuters, and two
  subreddits all ran a story, it clusters into one entry scoring `+3` per extra
  outlet. This is what pushes real news above filler.

  Clustering is IDF-weighted cosine over headline tokens, behind an **entity
  gate**: two headlines must share a proper noun before similarity is even
  considered. This is what keeps "magnitude 6.2 earthquake strikes Chile" and
  "magnitude 5.8 earthquake strikes Japan" apart - they score 0.27 similarity
  but share no entity. Proper nouns are detected by mid-headline capitalisation
  (pooled across the whole batch, so a name is recognised even where it appears
  headline-initial) plus internal capitals for brands like OpenAI or DeepMind.
  Title Case headlines are excluded from the vocabulary since they capitalise
  everything. When neither headline yields an entity, a much stricter
  similarity bar (0.58 vs 0.20) substitutes for the gate.
- **Hacker News points** add up to `+3`, parsed from the hnrss description.
- **Source weight** gives labs (OpenAI, DeepMind, Anthropic = 4.0) a head start
  over aggregators (Reddit = 1.5).
- **Recency** adds a small decaying bonus.

Reddit and HN are deliberately low-weight: they're there to *rank* stories other
feeds broke, not to break stories themselves.

The LLM only ever sees headlines and feed snippets — never article bodies. It
returns story IDs, and links are looked up from the feed data, so the model
cannot hallucinate a URL.

Run `python test_clustering.py` after changing any threshold in `main.py` - it
covers the adversarial cases (same entity/different story, same phrasing
pattern/different event) that are easy to break while tuning for recall.

## Editing feeds

`FEEDS` at the top of `main.py`. Each entry needs `name`, `url`, `category`
(`world` / `ai` / `tech`), and `weight`. A feed that 404s logs `FAIL` and is
skipped — it can't break the run.

Feed URLs rot constantly. Check the Actions log occasionally for `FAIL` or
`EMPTY` lines. Known fragile ones:

- **Anthropic** publishes no official RSS. The configured
  `rsshub.bestblogs.dev` mirror is community-run and may vanish; the Google News
  fallback query covers you if it does.
- **Reddit** sometimes 403s datacenter IPs, so it may fail intermittently from
  GitHub runners. Low weight means losing it costs little.
- **Reuters** has no official feed; the Google News proxy is doing the work.

## Caveats

- GitHub disables scheduled workflows after **60 days of repo inactivity**, and
  bot commits may not reset that timer. If the digest goes quiet, push any commit
  and re-enable the workflow.
- Public repos get unlimited Actions minutes; private repos use ~2 min/run,
  about 240 min of the 2000/month free allowance.
- `state/seen.json` is committed after each successful send. If a send fails,
  state is left untouched so the next run retries the same stories.
