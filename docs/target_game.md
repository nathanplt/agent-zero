# Target Game: Money Clicker Incremental

## Overview

- **Game URL**: https://www.roblox.com/games/18408132742/Money-Clicker-Incremental
- **Game ID**: 18408132742
- **Genre**: Incremental / Clicker
- **Estimated completion time**: 8–20 hours (depends on active vs idle play)
- **Selection date**: 2025

## Candidate Games Evaluated

At least three candidate games were evaluated against the selection criteria (publicly available, pure incremental, reasonable scope, stable, clear UI).

### 1. Money Clicker Incremental (SELECTED)

- **URL**: https://www.roblox.com/games/18408132742/Money-Clicker-Incremental
- **Rationale**: Large player base (26M+ visits), high rating, 545+ upgrades with clear progression. Pure click/upgrade loop. Public and stable. Resource display uses standard abbreviated numbers (K, M, B) suitable for OCR.
- **Fit**: Meets all five criteria. Strong automation potential.

### 2. Tapping Legends X

- **URL**: https://www.roblox.com/games/8750997647/Tapping-Legends-X
- **Rationale**: Very popular (27M+ players), rebirth/rubies/pets add complexity. Multiple worlds and eggs increase UI states and decision space.
- **Fit**: More complex (pets, eggs, worlds). Good for a second target; scope larger than ideal for first automation.

### 3. Clicker Simulator X

- **URL**: https://www.roblox.com/games/120575147399256/Clicker-Simulator-X
- **Rationale**: Classic clicker simulator style. Simpler than Tapping Legends but less documented than Money Clicker Incremental.
- **Fit**: Viable alternative; Money Clicker chosen for better documentation and clearer upgrade structure.

## Core Loop

1. **Click** to earn money (primary resource).
2. **Buy upgrades** to increase money per click and passive income.
3. **Buy auto-clickers** (or equivalent) for idle progress.
4. **Prestige / Rebirth** at a threshold to gain permanent multipliers.
5. Repeat until win condition or target completion.

## UI Elements

| Element            | Location   | Detection Method                    |
|--------------------|------------|-------------------------------------|
| Money counter      | Top / Top-left | OCR, format e.g. "1.5M", "2.3B" |
| Upgrades list      | Right panel / sidebar | UI detection, buttons with cost text |
| Upgrade button     | Per row in list | Template / button detection, OCR cost |
| Prestige / Rebirth | Modal or top button | Button + confirmation dialog |
| Main click area    | Center     | Large clickable region               |
| Settings / Menu   | Top-right  | Icon or hamburger menu               |

Detection strategy: use existing OCR (2.2) for numeric resources and costs; UI detection (2.3) for buttons and panels; LLM vision (2.4) when layout or new screens are ambiguous.

## Screenshots

Place real screenshots from the target game in `tests/fixtures/game/` for testing and training. Required coverage:

- `main_screen.png` – Default game view (money, upgrades visible).
- `upgrade_menu.png` – Upgrade panel or list in focus.
- `prestige_dialog.png` – Prestige/rebirth confirmation (if applicable).

Additional screens (optional but recommended): settings menu, different prestige tiers, error/loading states. Plan calls for 50+ screens covering all UI states; add them as the game is played and annotated.

## Win Condition

For automation purposes, **completion** is defined as one or both of:

- **Reach a target prestige/rebirth level** (e.g. prestige 10 or a documented “max” tier).
- **Max out primary upgrades** (all purchasable upgrades bought at least once).

If the game has no formal end, define a **stopping condition** (e.g. “prestige 5 and 1B total money”) and document it here after validation.

## Game-Specific Config

See `configs/game_config.yaml` for:

- Game URL and ID
- Resource names and display formats
- UI region hints (e.g. counter area, upgrade panel)
- Prestige thresholds and win-condition parameters
