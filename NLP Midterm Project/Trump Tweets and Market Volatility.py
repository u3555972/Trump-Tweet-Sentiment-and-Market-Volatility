#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 14:31:51 2022

@author: Chan, Cheuk Hang (Hang)
UID: 3035559725

Midterm Project: How did President Trump's tweets affect market volatility?

Trump's account has been suspended since Jan 8, 2021, thus,
Tweets provided by: https://www.thetrumparchive.com/
VIX is used as a proxy for market volatility
"""

# Import libraries needed to run program
import pandas as pd
import os
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import time

def yahooFinanceVolatility():
    
    '''
    The function downloads the VIX to measure market volatility in the US
    from Trump's inauguration until the suspension of his twitter account. It
    outputs a csv file containing the VIX daily values. I assume that all data
    here will be type float (beside the column names)
    '''
    
    # Trump's presidency started in 2017-01-20, but his twitter account
    # was suspended in 2021-01-08, hence not the full presidency tenure
    # of VIX was downloaded. Weekly data is collected here to view whole term
    yvix_df = yf.download('^VIX', start = '2017-01-21', end = '2021-01-12',
                         interval = '1wk', progress = False) 
    
    # Outputs a csv file to my preferred folder
    yvix_df.to_csv(path + os.sep + "pres_trump_vix.csv")

def cleanDF(df):
    '''
    This function removes all "isDeleted" tweets and removes any symbols like
    "RT", "@", "...", etc. and links (https:)
    '''
    
    # Only takes non-deleted tweets and after inauguration, inclusive
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= '2017-01-21')]
    
    # Reset index to make the data easier to process
    df = df[df['isDeleted'] == 'f'].reset_index()
    
    # Loop through all rows
    for text in range(len(df['text'])):
        
        # Split the text into a list to remove certain words
        text_lst = df['text'].iat[text].split()
        
        # Loop through all words to remove certain words that may cause
        # confusion for sentiment analysis
        for word in list(text_lst):
            if 'RT' in word:
                text_lst.remove(word)
            elif '@' in word:
                text_lst.remove(word)
            elif '…' in word:
                text_lst.remove(word)
            elif '...' in word:
                text_lst.remove(word)
            elif 'http' in word:
                text_lst.remove(word)
            elif '#' in word:
                text_lst.remove(word)
            elif '&amp;' in word:
                text_lst.remove(word)
        
        # Join the words together to create a string for sentiment analysis
        df.at[text, 'text'] = ' '.join(text_lst)
    
    
    # Any empty cells after cleaning will be removed
    df = df.drop(df[df['text'] == ''].index)
    
    # Sort by date
    df = df.sort_values(by = 'date')
    
    # Reset index again to organize the index numbers
    df.reset_index(inplace = True)
    
    # A new column called 'index' and 'level_0' is created after resetting the 
    # index, thus it is dropped to avoid confusion
    df = df.drop(columns = ['index', 'level_0'])
        
    return df
        
def convertDaysHours(date_t_df, date_v_df):
    '''
    This function converts the date to pandas datetime format and also changes
    the day into the NEXT day of vix should the tweet fall after regular hours.
    8:30 am - 3:15 pm. If tweet comes after 3:15 pm, it will be counted towards
    the next day. It also converts any weekend tweets to count towards the
    next week. I assume that everything in "date" is a string that can 
    easily converted to a datetime object.
    '''
    
    # Convert Date column to a datetime object
    date_v_df['Date'] = pd.to_datetime(date_v_df['Date'])
    
    # Create a new column to calculate percentage change
    date_v_df['Weekly Vix Return (Adj Close %)'] = date_v_df['Adj Close'].pct_change()*100
    
    # Initialize empty list to append converted date based on the rules above
    converted_date = []
    
    # Loop through date_t_df to convert dates to deal with market closing hours
    for i in range(len(date_t_df)):
        
        # Based on Vix market closing hours and minutes (15:15)
        if date_t_df['date'].iat[i].hour > 15:
            if date_t_df['date'].iat[i].minute > 15:
                
                # Convert to the next day after closing hours
                converted_date.append(date_t_df['date'].iat[i] + pd.DateOffset(days = 1))
                
            else:
                
                # Append naturally if before 15 minutes
                converted_date.append(date_t_df['date'].iat[i])
        else:
            
            # Append naturally if its before 15 hour (3 pm)
            converted_date.append(date_t_df['date'].iat[i])
    
    # Convert the list into a Series in the DataFrame
    date_t_df['converted_date'] = converted_date
    
    # Convert to datetime object for future manipulation
    date_t_df['converted_date'] = pd.to_datetime(date_t_df['converted_date'].dt.date)
    
    # Loop through date_t_df to convert dates on Sat or Sun to the next
    # week (considering weekend tweets may affect volatility when market opens)
    for j in range(len(date_t_df)):
        
        # If 5 (Saturday), add 2 days to convert to week of Monday
        if date_t_df['converted_date'].iat[j].dayofweek == 5:
            date_t_df['converted_date'].iat[j] = date_t_df['converted_date'].iat[j] + pd.DateOffset(days = 2)
        
        # If 6 (Sunday), add 1 day to convert to week of Monday
        elif date_t_df['converted_date'].iat[j].dayofweek == 6:
            date_t_df['converted_date'].iat[j] = date_t_df['converted_date'].iat[j] + pd.DateOffset(days = 1)
    
    return date_t_df, date_v_df

def sentimentAnalysis(cleaned_df):
    '''
    This function applys VADER SIA to the tweets and creates a new column
    showing only the compound score to later compare sentiment to volatility
    '''
    # Initialize sentiment analysis tool
    sia = SentimentIntensityAnalyzer()
    
    # Empty list initialized to append compound sentiment score
    compound_sentiment_lst = []
    
    # Looping through the text column
    for row in range(len(cleaned_df['text'])):
        
        # Assign contents to text variable to be analyzed and append to list
        text = cleaned_df['text'].iat[row]
        ss = sia.polarity_scores(text)
        compound_sentiment_lst.append(ss['compound'])
        
    # Convert the list into a Series to be in the DataFrame   
    cleaned_df['Compound Sentiment'] = compound_sentiment_lst
    
    return cleaned_df


def matchVolatility(trump_df, vix_df):
    '''
    This function takes in the trump tweets and the vix to match the tweet
    week to then gain the weekly sentiment and vix average.
    '''
    
    # Intialize list to append the average weekly sentiment score
    avg_week_sentiment = []
    
    # Initialize new DataFrame for the correlated DataFrame
    corr_df = pd.DataFrame()
    
    # Copy the dates because it is already weekly in vix_df
    corr_df['Week of Date'] = vix_df['Date']
    
    # Loop through vix_df to find the correlating dates in order to find the
    # average weekly sentiment score within that week and append to a list
    for day_i in range(len(vix_df)):
        
        # Within the boundaries of two given days (Mon-Fri), append the 
        # average value of the compound sentiment score to list
        avg_week_sentiment.append((trump_df['Compound Sentiment'][(trump_df['converted_date'] >= vix_df['Date'].iat[day_i]) 
                                & (trump_df['converted_date'] <= (vix_df['Date'].iat[day_i] + pd.DateOffset(days = 4)))]).mean())
        
    # Convert average weekly setniment score list to Series in DataFrame
    corr_df['Avg Weekly Compound Sentiment'] = avg_week_sentiment
    
    # Copy in Adj Close and Vix Return (%) values to plot later on
    corr_df['Vix Adj Close'] = vix_df['Adj Close']
    corr_df['Vix Return (%)'] = vix_df['Weekly Vix Return (Adj Close %)']
    
    # Output the CSV after matching Vix and Tweets
    corr_df.to_csv(path + os.sep + 'Correlated Vix and Trump Tweets.csv', index = False)
    
    return corr_df

def graphVixSentiment(corr_df):
    '''
    This function graphs both the Vix values and the sentiment scores with
    two y-axes for visualization.
    '''
    
    # Drops any row with empty data
    corr_df.dropna(inplace = True)
    
    # Initialize graph
    fig, ax = plt.subplots()
    
    # Initialize use of dual lines and y-axes
    ax_2 = ax.twinx()
    
    # Plot Vix and Compound Sentiment Score lines with labeling and legend
    vix_line, = ax.plot(corr_df['Week of Date'], corr_df['Vix Adj Close'], 'b', label = 'Vix')
    comp_sia, = ax_2.plot(corr_df['Week of Date'], corr_df['Avg Weekly Compound Sentiment'], 'r', label = 'SIA Score')
    ax.legend(handles = [vix_line, comp_sia], loc = 'lower right')
    
    # Label x and y axis and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Vix (Adj Close)', color = 'b')
    ax_2.set_ylabel('Avg Compound Sentiment Score', color = 'r')
    plt.title('VIX and Average Compound Sentiment Score')
    
    # Show and save graph
    plt.savefig(path + os.sep + 'VIX and Average Compound Sentiment Score.png')
    plt.show()
    
    
def graphRegression(corr_df):
    
    # Drops any row with empty data
    corr_df.dropna(inplace = True)
    
    # Assign x and y values for regression
    x = corr_df['Avg Weekly Compound Sentiment']
    y = corr_df['Vix Adj Close']
    
    # Use numpy to get coeffiecient(B1) and intercept (B0)
    a, b = np.polyfit(x, y, deg = 1)
    
    # Plotting of regression line with formula, labels, and title
    plt.plot(x, a*x + b, label = '{a:.2f}x + {b:.2f}'.format(a = a, b = b))
    plt.xlabel('Compound Sentiment Score')
    plt.ylabel('VIX')
    plt.title('Overall Regression')
    plt.legend(loc = 'lower left')
    
    # Showand save graph
    plt.savefig(path + os.sep + 'Overall Regression.png')
    plt.show()
    
    # Print regression formula for overall timeframe
    print('The Overall Regression Formula is: '\
          'VIX = {a:.2f}(Sentiment Score) + {b:.2f}'.format(a = a, b = b))
    
    print('')
    
    # Only done after discovering highly positive correlation after VIX Peak
    # Conduct regression after April 2020
    apr2020_corr = corr_df[corr_df['Week of Date'] >= '2020-04-1']
    
    # Assign x and y values for regression
    x = apr2020_corr['Avg Weekly Compound Sentiment']
    y = apr2020_corr['Vix Adj Close']
    
    # Use numpy to get coeffiecient(B1) and intercept (B0)
    a, b = np.polyfit(x, y, deg = 1)
    
    # Plotting of regression line with formula, labels, and title
    plt.plot(x, a*x + b, label = '{a:.2f}x + {b:.2f}'.format(a = a, b = b))
    plt.xlabel('Compound Sentiment Score')
    plt.ylabel('VIX')
    plt.title('Starting April 2020 Regression')
    plt.legend(loc = 'lower right')
    
    # Show and save graph
    plt.savefig(path + os.sep + 'After April 2020 Regression.png')
    plt.show()
    
    
    # Print regression formula starting April 2020
    print('Starting April 2020, the Regression Formula is: '\
          'VIX = {a:.2f}(Sentiment Score) + {b:.2f}'.format(a = a, b = b))
    
    print('')
    
def main():    
    
    '''
    This function is the main part of the program which calls on other
    functions as well as read and output files
    '''
    
    # Track the start time
    start_time = time.time()
    
    ## To prevent any errors, use of try and except
    
    # Read the vix csv file
    try:
        og_vix = pd.read_csv(path + os.sep + "pres_trump_vix.csv") 
        
    # If there is no csv file, the file will be downloaded automatically
    except:
        print("VIX csv file not found, downloading now")
        yahooFinanceVolatility()
        og_vix = pd.read_csv(path + os.sep + "pres_trump_vix.csv") 
    
    # Read the trump tweets csv file
    try:
        og_trump = pd.read_csv(path + os.sep + "Trump Tweets 2009.csv") 
        
    # If there is none, it prompts an error message to check for the file
    except:
        print("Trump Tweets csv file not found, please check again")
    
    # Ensure that the original DataFrame will not be altered
    vix_df = og_vix.copy()
    trump_df = og_trump.copy()
    
    # Cleans the trump tweets
    trump_df = cleanDF(trump_df)
    
    # Converts the dates to match the VIX weeks
    trump_df, vix_df = convertDaysHours(trump_df, vix_df)
    
    # Conduct sentiment analysis
    trump_df = sentimentAnalysis(trump_df)
    
    # Match the VIX values to the trump tweets' sentiment scores
    corr_df = matchVolatility(trump_df, vix_df)
    
    # Graph both VIX and sentiment scores
    graphVixSentiment(corr_df)
    
    # Graph regression
    graphRegression(corr_df)
    
    # Print the overall time the program runs for
    print('The time it took to run the program:', time.time() - start_time, 'seconds')


# Path to read and save my files. Global variable since used very often to 
# read or write files
path = r'/Users/XFlazer/Documents/HKU/FBE/Finance/Natural Language Processing/NLP Midterm Project'

# Calls main program
main()