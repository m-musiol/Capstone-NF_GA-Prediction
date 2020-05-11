![Logo](pics/logo.png)

---
# Capstone Project @neuefische
This is my capston project from the twelve week Data Science Bootcamp at neuefische in Hamburg and should show what I learned in the last three months. I am new in Data Science and had almost no previous knowledge in programming and mathematical models, so there was a lot to learn... For my project I have selected the following kaggle competition.

---
## Google-Analytics-Customer-Revenue-Prediction
https://www.kaggle.com/c/ga-customer-revenue-prediction


### Introduction
For many businesses only a small percentage of customers produce most of the revenue. As such, marketing teams are challenged to make appropriate investments in promotional strategies.

In this competition, you’re challenged to analyze a **Google Merchandise Store** customer dataset to predict revenue per customer. Hopefully, the outcome will be more actionable operational changes and a better use of marketing budgets for those companies who choose to use data analysis on top of GA data.


### What am I predicting?
We are predicting the natural log of the sum of all transactions per user. Once the data is updated, as noted above, this will be for all users in test_v2.csv for December 1st, 2018 to January 31st, 2019. For every user in the test set.
Note that the dataset does NOT contain data for December 1st 2018 to January 31st 2019. You must identify the unique fullVisitorIds in the provided test_v2.csv and make predictions for them for those unseen months.


### What files do I need?
You will need to download train_v2.csv (24 GB) and test_v2.csv (7 GB). These contain the data necessary to make predictions for each fullVisitorId listed in sample_submission_v2.csv.

Download the data here: https://www.kaggle.com/c/ga-customer-revenue-prediction/data


### File Descriptions
- train_v2.csv - the updated training set - contains user transactions from August 1st 2016 to April 30th 2018.
- test_v2.csv - the updated test set - contains user transactions from May 1st 2018 to October 15th 2018.
- sample_submission_v2.csv - a updated sample submission file in the correct format. Contains all fullVisitorIds in test_v2.csv. Your submission's PredictedLogRevenue column should make forward-looking predictions for each of these fullVisitorIds for the timeframe of December 1st 2018 to January 31st 2019. Review "What am I predicting?" above for details.

---
## My three biggest challenges
- **Handle the data size**

    I was using a MacBook Pro with 8GB of memory and only 35GB of free space on the hard drive. The first big hurdle was to load the data into the Jupyter notebook. 
    
- **Understand the meaning of 148 different features**

    I know a lot about advertising, but I have never seen Google Analytics data before. The most difficult thing was to find out under which conditions missing values occur. 
    
- **To find the needle in the haystack**

    The task was to find from a list of almost 300,000 IDs those who would spend money again within a defined period. Only 30-40 belong to the target group.
    
---    
## Which files are in this repositorie? 
- **GA-Customer-Revenue-Prediction_presentation**

    A short non-technical presentation about my solution to the task and what additional information I could draw from the data set so far. 
    
- **GA-Customer-Revenue-Prediction_model**
    
    The complete process from data import, preparation, feature engineering, modelling and export the result. 
    
- **Model_finding**
    
    The different models I tested before I decided to use LightGBM.
    
- **EDA_and_Visualization**

    The visualizations I used in the presentation.
