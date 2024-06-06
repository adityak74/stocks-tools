### Stock News Grabber

Usage:

1. The example file is available in the `.env.example` file.
2. Copy and create `.env` file and add your newsapi api key (if you do not have one get here: https://newsapi.org/account) to the `.env` file and search query string.
3. Run the following command to start the application.
4. This should create a `news.json` file with the results.


```bash
make run
```

_*NOTE:*_ The application will run the newsapi to get the news related to the query as a json file with links to articles.
