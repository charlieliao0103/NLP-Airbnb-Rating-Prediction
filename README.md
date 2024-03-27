# Prediction of Star Rating Outcomes of Edinburgh Airbnbs Through Review Analysi

Hi, my name is Charlie Liao and this is my Capstone Project for the Brainstation Data Science Bootcamp. In this project, I used Machine Leaning algorithms to analyze review texts and predict star rating outcomes for Edinburgh Airbnb listings.

### Project Background

Airbnb has revolutionized the hospitality industry by offering unique accommodations worldwide. Edinburgh, a city known for its rich history and vibrant culture, serves as the backdrop for this analysis. By leveraging Edinburgh Airbnb listings and guest reviews data, we aim to uncover insights that can benefit both Airbnb hosts and guests.

### Methodology
Our approach revolves around NLP (Natural Language Processing) techniques and supervised learning classification models. NLP helps us preprocess and analyze textual data, while classification models like Logistic Regression and Decision Trees enable us to predict high rating scores for Airbnb listings. We're also exploring the integration of deep learning methods attempting to further enhance model performance.

### Dataset
Our dataset comprises two distinct datasets: Listing data and review data sourced from Inside Airbnb. These datasets are processed separately initially and then aggregated in two ways: Listings data aggregated on each unique review (Reviews without Collapsed), and Review data collapsed and aggregated on each unique listing (Reviews Collapsed). This dual approach allows for comprehensive analysis and interpretation of the data in varous perspective.

### Findings
In our initial analysis using the Reviews Without Collapsed data, we achieved promising results. Our best-performing model boasts an 94% accuracy on unseen data, outperforming the baseline by 40%. Additionally, we've identified key words in guest reviews as well as listing features that serve as potential indicators for high overall ratings, providing actionable insights for Airbnb hosts. 

Indicating words include:
- lovely
- welcome
- wonderful
- touch
- clean

Top Listing Features include:
- host_is_superhost
- number_of_reviews
- number_of_reviews_ltm (number of reviews last 12 months)

For access to the dataset and data dictionary, please visit [here](https://drive.google.com/drive/folders/1KuRosCHxTRygPDLzHlt5agvEWF9eAcai?usp=sharing).