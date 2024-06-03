### Stock Announcer

Usage:

1. The example file is available in the `.env.example` file.
2. Copy and create `.env` file and add your robin hood credentials to the `.env` file and symbol of the stock you want to track separated by a comma. 
3. Run the following command to start the application.
4. To override the deplay time, update the `SLEEP_TIME` variable in the `.env` file.


```bash
make run
```

_*NOTE:*_ The application will run every 5 seconds and will announce the stock price.