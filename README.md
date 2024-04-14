# Prediction of Star Rating Outcomes of Edinburgh Airbnbs Through Review Analysis

Hello, my name is Charlie Liao, and this is my Capstone Project for the Brainstation Data Science Bootcamp. In this project, I utilized Machine Learning algorithms to analyze review texts and predict star rating outcomes for Edinburgh Airbnb listings.

## Project Background

Airbnb has revolutionized the hospitality industry by offering unique accommodations worldwide. Edinburgh, a city known for its rich history and vibrant culture, serves as the backdrop for this analysis. By leveraging Edinburgh Airbnb listings and guest reviews data, we aim to uncover insights that can benefit both Airbnb hosts and guests. In this project, we aim to answer the following questions: How can Airbnb hosts improve their rentals through analysis of guest reviews? What are the key factors that determine an outstanding Airbnb listing? 

## Methodology
Our approach revolves around NLP (Natural Language Processing) techniques and supervised learning classification models. NLP helps us preprocess and analyze textual data, while classification models like Logistic Regression and Decision Trees enable us to predict if an Airbnb listing belongs to the class of 'Outstanding'. We're also exploring the integration of deep learning methods attempting to further enhance model performances.

## Dataset
Our dataset comprises two datasets: Listing data and review data sourced from Inside Airbnb. These datasets are processed separately initially and then aggregated in two ways: Listings data aggregated on each unique review (Reviews without Collapsed), and Review data collapsed and aggregated on each unique listing (Reviews Collapsed). This dual approach allows for comprehensive analysis and interpretation of the data in varous perspective.

## Notebooks
This project consists of a total of 7 notebooks, each serving a specific purpose:

- **1_Data_Cleaning**: Primary data cleaning process is conducted in this notebook, where null values, irrelevant columns, and duplicated columns are removed for future analysis.
- **2_EDA**: Advanced exploratory data analysis is performed in this notebook. We define our problem statement into a machine learning suitable modeling question: Predicting if a listing belongs to the outstanding class **(Overall Rating Score > 4.8)** or the less-outstanding class **(Overall Rating Score < 4.8)**. We then address fundamental questions Airbnb hosts might ask: How can I improve my rating score to rank among the outstanding group? This approach can potentially impact future bookings and revenue.
- **3_Data_Pre_Processing**: Secondary data cleaning and pre-processing steps are conducted in this notebook. We obtain two types of review datasets as defined above. Additionally, we remove all non-English reviews from the dataset for ease of future modeling.
- **4_Modelling_Reviews_Uncollapsed**: Initial modeling stage is performed in this notebook, where several helper functions, including a customized tokenizer and column transformer, are defined. Two GridSearches are conducted, and we eventually obtain a well-trained Decision Tree classifier with a testing score of 94%.
- **5_Modelling_Collapsed_Reviews**: Secondary modeling stage is performed in this notebook. We apply the previous approach to this form of dataset aiming for better model interpretability.
- **6_Modelling_Subratings**: Alongside modeling analysis, predicting different types of sub-ratings becomes necessary and interesting for analysis. We perform a small grid search on the uncondensed review dataset for four different sub-ratings as target variables and obtain corresponding results.
- **7_Findings**: All modeling results are presented and evaluated in this notebook. Through feature extraction, we answer the pre-defined modeling question and provide actionable insights.

For access to the dataset and data dictionary, please visit [here](https://drive.google.com/drive/folders/1KuRosCHxTRygPDLzHlt5agvEWF9eAcai?usp=sharing).