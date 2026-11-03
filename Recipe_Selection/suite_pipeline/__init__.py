"""Configuration-driven, end-to-end PORTAL suite construction."""

from .discovery import RecipeRecord, discover_recipes

__all__ = ["RecipeRecord", "discover_recipes"]
