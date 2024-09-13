# Analysis on young NBA talents

Given a dataset `nba_logreg.csv` containing stats about beginner players in NBA, we would like to train a binary classifier to assess the player ability to stand in the league for the long run (> 5 years). This model objective is to counsel investors seeking to find a good player investment. 

The dataset comprise the following features:

| Name         | Description                              |
|--------------|------------------------------------------|
| Name         | Name                                     |
| GP           | Games Played                             |
| MIN          | Minutes Played                           |
| PTS          | Points Per Game                          |
| FGM          | Field Goals Made                         |
| FGA          | Field Goal Attempts                      |
| FG%          | Field Goal Percent                       |
| 3P Made      | 3 Point Made                             |
| 3PA          | 3 Point Attempts                         |
| 3P%          | 3 Point Percent                          |
| FTM          | Free Throw Made                          |
| FTA          | Free Throw Attempts                      |
| FT%          | Free Throw Percent                       |
| OREB         | Offensive Rebounds                       |
| DREB         | Defensive Rebounds                       |
| REB          | Rebounds                                 |
| AST          | Assists                                  |
| STL          | Steals                                   |
| BLK          | Blocks                                   |
| TOV          | Turnovers                                |
| TARGET_5Yrs  | Outcome: 1 if career length >= 5 yrs, 0 if < 5 yrs |

*Table 1 – Feature description*



## Question 1 - Exploration and Analysis

Find the analysis in `exploration/exploration.ipynb`

## Question 2 - Setting in production with FastAPI  

1. Install the dependencies using `poetry install`
2. Go the poetry shell using `poetry shell`
3. start the server with 
``` bash
    fastapi dev src/api/api.py
```
4. Predict the result for a player using the following command:
   ``` bash
   curl -X POST http://127.0.0.1:8000/players/predict\
    -H "Content-Type: application/json" \
    -d '{
      "games_played": 60,
      "minutes_played": 17.0,
      "points": 7,
      "fieldgoals_made": 3,
      "fieldgoals_attempts": 8,   
      "fieldgoals_percents": 37.5,
      "threepoints_made": 1,
      "threepoints_attempts": 4,
      "threepoints_percents": 25.0,
      "freethrow_made": 0,
      "freethrow_attempts": 1,
      "freethrow_percentage": 0.0,
      "offensive_rebounds": 1,
      "defensive_rebounds": 3,
      "rebounds": 4,
      "assists": 1,
      "steals": 0,
      "blocks": 0,
      "turnovers": 2
    }'
    > {"target_5yrs":false}
    ```

> ⚠️ If you do not have poetry installed, create a python virtual environment and install the dependencies using `pip install -r requirements.txt`

### Improvments to be made

Due to a lack of time, I wasn't able to include all the features I wanted in my API, here are a few example of improvments to do:

* Get rid of the pkl model file and create a Model in the database for the classifier
* Create a dockerfile and contenerize the application
* Add some unit testing