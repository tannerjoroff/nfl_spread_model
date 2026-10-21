import pytest


@pytest.mark.skip(reason="no features until Phase 2")
def test_changing_game_x_does_not_change_its_features():
    """Change the outcome and stats of game X; the features for game X must not change."""
