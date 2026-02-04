# Game fixtures

This directory holds screenshots from the target Roblox incremental game for testing and training.

## Placeholder files

The repository includes minimal placeholder PNGs (`main_screen.png`, `upgrade_menu.png`, `prestige_dialog.png`) so that tests requiring these paths pass without real screenshots.

## Adding real screenshots

For full vision/OCR and UI detection testing, replace placeholders with real screenshots from the target game (see `docs/target_game.md` and `configs/game_config.yaml`). Capture 50+ screens covering:

- Main game view
- Upgrade menu / panel
- Prestige or rebirth dialog
- Settings and other UI states

Save them under the names listed in `configs/game_config.yaml` under `fixtures.required_screenshots`, or add additional files as needed.
