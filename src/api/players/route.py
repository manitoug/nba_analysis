import joblib as jl
import pandas as pd
import numpy as np

from fastapi import APIRouter
from .schema import Player


player_router = APIRouter(prefix="/players", tags=["players"])

@player_router.post("/predict")
async def predict_player(player: Player):
    pipeline = jl.load("./data/trained_nba.pkl")
    #'FG%', '3P Made', 'FTA', 'OREB', 'AST', 'BLK']

    X = np.array([player.fieldgoals_percents, player.threepoints_made,
                      player.freethrow_attempts, player.offensive_rebounds,
                      player.assists, player.blocks]).reshape((1, -1))
    predicted = pipeline.predict(X)[0]
    return {"target_5yrs":True if predicted>0 else False}
