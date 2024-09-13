from datetime import datetime
from pydantic import BaseModel
from typing import List
from uuid import UUID


class Player(BaseModel):
    name: str | None = None
    games_played: int
    minutes_played: float
    points: int
    fieldgoals_made: int
    fieldgoals_attempts: int
    fieldgoals_percents: float
    threepoints_made: int
    threepoints_attempts: int   
    threepoints_percents: float
    freethrow_made: int
    freethrow_attempts: int
    freethrow_percentage: float
    offensive_rebounds: int
    defensive_rebounds: int
    rebounds: int
    assists: int
    steals: int
    blocks: int
    turnovers: int
    stayed_over_5y: bool | None = None
