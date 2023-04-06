import os
import sys
from typing import List

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class User(BaseModel):
    user_id: int
    time: int
    popular_streamers: List


def process_data(path_from: str, time_now: int = 6147):
    """Function process

    Parameters
    ----------
    path_from : str
        path to read data
    time_now : int
        time to filter data

    Returns
    -------
    data: pandas.DataFrame
        dataframe after proccessing
    """
    df = pd.read_csv(
        path_from,
        names=["uid", "session_id", "streamer_name", "time_start", "time_end"]
    )
    df = df[(df["time_start"] < time_now) & (df["time_end"] > time_now)]
    return df


def recomend_popularity(data: pd.DataFrame):
    """Recomend Popularity

    Parameters
    ----------
    data : pd.DataFrame

    Returns
    -------
    popular_streamers: List
    """
    data = data.groupby(['streamer_name']) \
        .count() \
        .reset_index() \
        .sort_values(['session_id'], ascending=False)
    return data['streamer_name'].tolist()


@app.get("/popular/user/{user_id}")
async def get_popularity(user_id: int, time: int = 6147):
    """Fast Api Web Application

    Parameters
    ----------
    user_id : int
        user id
    time : int, optional
        time, by default 6147

    Returns
    -------
    user: json
        user informations
    """
    path = os.path.join(sys.path[0], os.environ["data_path"])
    df = process_data(path, time)
    popular_streamers = recomend_popularity(df)
    user = User(user_id=user_id, time=time,
                popular_streamers=popular_streamers)
    return user


def main() -> None:
    """Run application"""
    uvicorn.run("solution:app", host="localhost")


if __name__ == "__main__":
    main()
    # df = process_data("100k_a.csv", 6147)
    # popular_streamers = recomend_popularity(df)
    # user = User(user_id=123, time=6147,
    #             popular_streamers=popular_streamers)
    # print(user)
