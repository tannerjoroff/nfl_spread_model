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

### 2026-10-22: first nflverse snapshot (2009 to 2025)
Source: nflreadpy 0.1.5, polars 2.0.0. Snapshot of `load_schedules` (seasons 1999 to 2025, filtered to exclude the live 2026 season), `load_pbp` and `load_team_stats` (weekly) for 2009 to 2025. 35 files, 219 MB, hashes in `data/manifest.json`.

- **Games:** 267 per season through 2019, 269 in 2020, 284 to 285 from 2021 (17-game season). 2022 has 271 regular-season games because the canceled Bills at Bengals game is not in the schedule. Every game has a result.
- **Play-by-play:** every season has exactly as many distinct `game_id`s as the schedule, 46,500 to 49,900 plays per season, and 372 columns in every season.
- **Team stats:** exactly 2 rows per game in every season, 138 columns, and `game_id` is present (the column whose addition broke the old notebook).
- **Spreads and totals:** present for 100% of games, 2006 to 2025.
- **Moneylines:** partial before 2010 (2006: 82%, 2007: 99.6%, 2008: 73%, 2009: 95%). From 2010 on, every game has both moneylines except one: `2017_04_CHI_GB`, which has a spread (7.5) but no moneylines. Verdict: confirms "full from 2010" with that single exception, which the moneyline backtests must drop and count.
- **Starting QB IDs:** present for 100% of games.
- **Weather:** `temp` and `wind` are missing by design for dome, closed-roof, and open-roof games (open-roof games have none recorded). For outdoor games, coverage is 94% to 99% in most seasons, but only 54% in 2022 and 79% in 2023. Verdict: weather features need a backfill for those seasons (candidate: the free Open-Meteo historical archive, using stadium coordinates) or must treat missing weather explicitly. Do not drop those games silently.

### 2026-10-23: snapshot extended back to 2006; test seasons now 2012 to 2024
Change: data now starts in 2006 (was 2009), and walk-forward test seasons start in 2012 (was 2015). The first split trains on 2006 to 2010, validates on 2011, and tests on 2012. Reason: three more test seasons (13 instead of 10) means roughly 30% more test games and standard errors about 12% smaller, which matters when the question is whether any edge is distinguishable from noise. Moneylines are complete from 2010, so every test season still has full moneyline coverage. Snapshot is now 41 files, 256 MB.

Coverage of the new seasons, compared with later ones (2025 not inspected):

| Season | Games in pbp / schedule | EPA present (pass and run plays) | Air yards present (passes) | CPOE present (passes) | League EPA per play |
|---|---|---|---|---|---|
| 2006 | 267 / 267 | 100% | 93.0% | 86.5% | -0.040 |
| 2007 | 267 / 267 | 100% | 93.4% | 90.5% | -0.028 |
| 2008 | 267 / 267 | 100% | 93.8% | 92.3% | -0.013 |
| 2010 | 267 / 267 | 100% | 93.6% | 92.4% | -0.016 |
| 2015 | 267 / 267 | 100% | 93.5% | 92.6% | -0.014 |
| 2024 | 285 / 285 | 100% | 92.5% | 88.7% | +0.010 |

Team stats have exactly 2 rows per game in 2006 to 2008, and play-by-play has the same 372 columns as every later season.

Verdict: 2006 to 2008 are usable for EPA and success rate features. CPOE is somewhat thinner in 2006. League-wide EPA per play drifts upward over time (offenses have improved), so efficiency features should be measured relative to the league average for that season, or the model will mistake era for team strength.

## Experiment log

One entry per experiment, newest last: date, what changed, metrics, verdict, commit hash. Results within about two standard errors of zero are recorded as no evidence of edge. Each entry notes how many strategies have been tested so far.

### 2026-10-23: market baselines, test seasons 2012 to 2024
What: score the betting market itself as a predictor, with no model. Predicted home margin = `spread_line`; home win probability = moneylines de-vigged multiplicatively. Regular season only, 3,391 games. Ties (12) are dropped from win probability metrics. One game (`2017_04_CHI_GB`) has no moneylines and is left out of the win probability and favorite-betting numbers. Full per-season table: `results/market_baselines.csv`.

Pooled over 2012 to 2024:

| Metric | Market | Reference point |
|---|---|---|
| Margin MAE | 10.02 points | |
| Margin RMSE | 13.00 points | |
| Log loss | 0.6094 | 0.693 for always saying 50% |
| Brier score | 0.2109 | 0.25 for always saying 50% |
| Favorite win rate | 66.7% (3,359 decided games) | |
| Bet every moneyline favorite, ROI per unit staked | -2.3% (SE 1.2%) | |

Single seasons vary a lot: log loss ranges from 0.587 (2017) to 0.652 (2015), and betting every favorite ranges from -12.0% (2021) to +6.6% (2013), with a per-season SE of about 4.4%. A single season of favorite betting can look profitable by chance alone.

Verdict: these are the numbers any model has to beat. The favorite-betting loss is about 1.9 SE below zero, which is not conclusive on its own, but it is what the bookmaker's margin predicts: betting without information should lose roughly the vig. The pooled figures are close to the earlier 2010 to 2025 reference (66.5%, 0.609, 0.211, -2.7%), as expected from overlapping seasons; 2025 was not used here.

Standard errors treat bets as independent; week-level and game-level bootstrap intervals come next. Strategies tested so far: 1 (bet every favorite, a reference point, not a candidate strategy). Commit: the commit that adds this entry.
