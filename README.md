# FINA4350 NLP Midterm Project - Trump Tweets and Market Volatility

## Overview
This project examines how Trump tweets during his presidency affect market volatility (measured by VIX).

## Sourcing
Trump tweets were downloaded from https://www.thetrumparchive.com/ while the VIX values were scraped using yfinance API.

## Cleaning
Trump tweets were cleaned of "RT" symbol, links, usernames, and hashtags (#). Also removed all deleted tweets.

## Sentiment Analysis
VADER SIA was used to do sentiment analysis on Trump tweets

## Report
It turns out that the sentiment of Trump's tweets is overall negatively correlated with market volatility. Basically this means as the sentiment value of Trump's tweet increases (becomes positive), then market volatility decreases. This makes sense as since the President is more positive, this usually indicates a stronger economy and market. However, starting from April 2020, it became significantly postively correlated, indicating that Trump's tweets were not significant in affecting market volatility. 
