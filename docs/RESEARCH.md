# Research Log: Data and Features

Data sources, feature ideas, findings, and the experiment log for this project. Add findings with a date, source, and verdict. Coverage years below are starting points to verify, not facts.

**Free sources only.** Anything paid is out of scope unless Tanner changes this rule.

## Data sources

### nflverse (via nflreadpy): primary, free
| Loader | What it has | Use for | Verify |
|---|---|---|---|
| `load_schedules` | Results, spread, total, moneylines (full from 2010), spread odds, rest days, starting QB IDs and names, roof, surface, temp, wind, coaches, referee, stadium | Targets, market baseline, situational and weather features | Confirmed 2026-10-21: moneylines are partial in 2006 to 2009 and full from 2010 on |
| `load_pbp` | Play-by-play with EPA, WPA, success, air yards, CPOE | Efficiency features (the biggest upgrade) | File size; load one season at a time |
| `load_injuries` | Weekly injury reports with practice and game status | Availability features | Start year; is the final pre-game status included? |
| `load_depth_charts` | Weekly depth charts | Identifying starters, especially offensive line and QB | Timing: is the chart published before games? |
| `load_snap_counts` | Snaps by player and game | Player importance weights (from past games only) | Never use the same game's snaps |
| `load_nextgen_stats` | NGS passing, rushing, receiving | Possible QB and receiver features | Start year (about 2016) |
| `load_ftn_charting` | Charting data (play action, blitz, and so on) | Possibly later | Start year (about 2022), too short for long backtests |
| `load_players`, `load_rosters_weekly` | Player IDs and positions | Joins | |

### Other sources
| Source | Use for | Notes |
|---|---|---|
| The Odds API, free tier | Current-week lines for the dashboard | Free tier has a monthly request cap; historical odds are a paid feature, so not usable for backtests |
| Open-Meteo | Forecast weather for upcoming games | Free, no key; needs stadium coordinates |
| Stadium coordinates | Travel distance, time zones, weather lookups | Small hand-built CSV keyed on nflverse `stadium_id` |
| Historical opening lines | Line movement, closing line value | Free options look thin. Candidates to check: free historical spreadsheets that hobby sites publish, and Kaggle datasets. If none are reliable, skip closing line value and say so in the write-up |

## Feature ideas

Each feature is marked with its leakage risk.

### Efficiency (from play-by-play)
- Offense and defense EPA per play, split pass and rush. Low risk if built from prior games only.
- Success rate, early-down EPA, and EPA excluding garbage time (win probability between 10% and 90%).
- Opponent adjustment: a ridge regression of play EPA on offense and defense team effects, fit on prior games only.
- Exponentially decayed history across seasons, with the half-life tuned on validation. Replaces the 3-game rolling window.

### Quarterback
- The starting QB's own decayed EPA per dropback, so injuries and benchings show up.
- Flag for a QB change versus the last game; backup QB indicator.
- Risk: the starting QB ID in schedules is known at kickoff, so it is fine for betting near kickoff, but not for lines bet early in the week.

### Injuries
- Count or weight of starters ruled Out or Doubtful, by position group (QB, offensive line, CB, edge rusher, WR).
- Risk: high. Use only the final pre-game report status, never participation from that game.

### Situational
- Rest differential, short week (Thursday), coming off a bye, international game.
- Travel distance and time zones crossed; west-coast team playing a 1pm ET game.
- Divisional game, primetime game.
- Home field advantage over time (it has shrunk since about 2020).

### Environment
- Dome or retractable roof, temperature, wind above about 15 mph, precipitation. Mostly affects totals and passing, so test the effect on margin rather than assuming it.

### Priors
- Preseason rating: last season's rating regressed toward the mean, possibly blended with season win-total markets.
- Coaching or coordinator change flag.

## Open research questions
- Is there a free, reliable source of historical opening lines, so we can measure closing line value? If not, closing line value is out of scope.
- How much does opponent-adjusted EPA beat raw EPA in walk-forward tests?
- Do injuries add anything once the starting QB is known?
- Best decay half-life within a season and across seasons.

## Findings
(Add dated entries here.)

## Experiment log

One entry per experiment, newest last: date, what changed, metrics, verdict, commit hash. Results within about two standard errors of zero are recorded as no evidence of edge. Each entry notes how many strategies have been tested so far.

(No experiments yet.)
