from fastapi import FastAPI
from contextlib import asynccontextmanager
import joblib as jl

from players.route import player_router
from pipelines.datascience_pipelines import LinearSVCPipeline    


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline = LinearSVCPipeline(input_file='./data/nba_logreg.csv',
                                 features=['FG%', '3P Made', 'FTA', 'OREB', 'AST', 'BLK'],
                                 target="TARGET_5Yrs")
    pipeline.load_data()
    pipeline.create_model()
    pipeline.fit()
    jl.dump(pipeline, "./data/trained_nba.pkl")
    yield   

app = FastAPI(title="NBA API", lifespan=lifespan)
app.include_router(player_router)

@app.get("/")
def read_root():
    return {"msg": "Welcome to the root"}   
