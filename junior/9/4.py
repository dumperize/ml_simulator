from typing import Dict
from fastapi import FastAPI
import uvicorn
import numpy as np

app = FastAPI()


class Click:
    """Class click"""

    def __init__(self, click_id: int):
        """Constructor"""
        self.click_id = click_id
        self.is_conversion = False
        self.reward = 0

    def set_offer(self, offer_id: int):
        """Set offer"""
        self.offer_id = offer_id

    def set_rewards(self, reward: int):
        """Set rewards"""
        self.is_conversion = reward > 0
        self.reward = reward


class ClicksDict:
    """Class click dict"""

    clicks: Dict[int, Click] = {}

    def get_click(self, click_id: int):
        """Get click"""
        return self.clicks[click_id]

    def set_click(self, click: Click):
        """Set click"""
        self.clicks[click.click_id] = click

    def get_clicks_by_offer_id(self, offer_id: int):
        """Get clicks by offer_id"""
        return list(filter(lambda x: x.offer_id == offer_id, self.clicks.values()))

    def get_stat_by_offer(self, offer_id: int):
        """Get stat"""
        offer_clicks = self.get_clicks_by_offer_id(offer_id)

        conversions = len(
            list(filter(lambda x: x.is_conversion, offer_clicks)))
        rewards = sum(map(lambda x: x.reward, offer_clicks))
        count_clicks = len(offer_clicks)

        cr = 0 if count_clicks == 0 else conversions / count_clicks
        rpc = 0 if count_clicks == 0 else rewards / count_clicks

        return {
            "offer_id": offer_id,
            "clicks": count_clicks,
            "conversions": conversions,
            "reward": rewards,
            "cr": cr,
            "rpc": rpc,
        }

    def clear(self):
        """Clear dict"""
        return self.clicks.clear()

    def get_optim_offer(self, offers_ids: [int]):
        """Get optima of offers"""
        ucbs = []
        for offer_id in offers_ids:
            epsilon = 0.5
            offer_clicks = self.get_clicks_by_offer_id(offer_id)
            if len(offer_clicks) == 0:
                ucbs.append({'offer_id': offer_id, 'ucb': 9999999})
            else:
                q = np.mean([x.reward for x in offer_clicks])
                s = np.sqrt(epsilon * np.log(len(self.clicks)) /
                            len(offer_clicks))
                ucbs.append({'offer_id': offer_id, 'ucb': q+s})
        return max(ucbs, key=lambda x: x['ucb'])['offer_id']


clicksDict = ClicksDict()


@app.on_event("startup")
def startup_event():
    """Clean dict"""
    clicksDict.clear()


@app.put("/feedback/")
def feedback(click_id: int, reward: float) -> dict:
    """Get feedback for particular click"""
    # Response body consists of click ID
    # and accepted click status (True/False)
    click = clicksDict.get_click(click_id=click_id)
    click.set_rewards(reward)
    return {
        "click_id": click.click_id,
        "offer_id": click.offer_id,
        "is_conversion": click.is_conversion,
        "reward": click.reward,
    }


@app.get("/offer_ids/{offer_id}/stats/")
def stats(offer_id: int) -> dict:
    """Return offer's statistics"""
    return clicksDict.get_stat_by_offer(offer_id)


@app.get("/sample/")
def sample(click_id: int, offer_ids: str) -> dict:
    """Sample random offer"""
    # Parse offer IDs
    offers_ids = [int(offer) for offer in offer_ids.split(",")]

    click = Click(click_id)

    offer_id = clicksDict.get_optim_offer(offers_ids)
    click.set_offer(offer_id)

    clicksDict.set_click(click)
    return {
        "click_id": click.click_id,
        "offer_id": click.offer_id,
    }


def main() -> None:
    """Run application"""
    uvicorn.run("4:app", host="localhost")


if __name__ == "__main__":
    main()
