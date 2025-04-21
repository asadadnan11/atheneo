@echo off
echo Starting Morning Sports Analysis Routine...

:: Set the time (adjust as needed)
set START_TIME=05:30
set END_TIME=06:30

:: Create timestamp for today's date
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set TODAY=%datetime:~0,8%

:: Create directory for today's data
mkdir data\%TODAY%

:: Step 1: Run Reddit Harvester (collect overnight data)
echo [%TIME%] Starting Reddit Harvester (estimated time: 20-30 minutes)...
python reddit_harvester.py --timeframe overnight --output data\%TODAY%\reddit_data.json
if %ERRORLEVEL% NEQ 0 (
    echo Error in Reddit Harvester. Check logs for details.
    exit /b 1
)
echo [%TIME%] Reddit Harvester completed successfully.

:: Step 2: Run GPT Signal Matcher
echo [%TIME%] Starting GPT Signal Matcher (estimated time: 5-10 minutes)...
python gpt_signal_matcher.py --input data\%TODAY%\reddit_data.json --output data\%TODAY%\signals.json
if %ERRORLEVEL% NEQ 0 (
    echo Error in GPT Signal Matcher. Check logs for details.
    exit /b 1
)
echo [%TIME%] GPT Signal Matcher completed successfully.

:: Step 3: Generate Tweet Summaries
echo [%TIME%] Generating Tweet Summaries (estimated time: 2-5 minutes)...
python tweetify_summaries.py --input data\%TODAY%\signals.json --output data\%TODAY%\tweets.json
if %ERRORLEVEL% NEQ 0 (
    echo Error in Tweet Summaries. Check logs for details.
    exit /b 1
)
echo [%TIME%] Tweet Summaries generated successfully.

:: Step 4: Send Results
echo [%TIME%] Sending Results...
python send_results.py --input data\%TODAY%\tweets.json
if %ERRORLEVEL% NEQ 0 (
    echo Error sending results. Check logs for details.
    exit /b 1
)

echo [%TIME%] Morning Analysis Complete!
echo Total runtime: %START_TIME% to %TIME% 