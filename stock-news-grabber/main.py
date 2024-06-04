import json

from newsapi import NewsApiClient
from dotenv import dotenv_values
from datetime import datetime, timedelta

cfg = dotenv_values(".env")
newsapi = NewsApiClient(api_key=cfg.get("NEWSAPI_API_KEY", None))


def get_past_two_weeks_date_from_today():
    today = datetime.today()
    past_two_weeks_date = today - timedelta(days=14)
    return past_two_weeks_date.strftime("%Y-%m-%d")


def get_today_date():
    today = datetime.today()
    return today.strftime("%Y-%m-%d")


def store_json_to_file(json_response):
    with open("news.json", "w") as file:
        json.dump(json_response, file, indent=4)


def get_all_articles():
    all_articles = newsapi.get_everything(
        q=cfg.get("SEARCH_QUERY", ""),
        from_param=get_past_two_weeks_date_from_today(),
        to=get_today_date(),
        language="en",
        sort_by="relevancy",
        page=1,
        page_size=100,
    )
    store_json_to_file(all_articles)


if __name__ == "__main__":
    get_all_articles()
