# Main Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import seaborn as sns

# Scipy Library for sparse  matrix
from scipy.sparse import csr_matrix

# NLP Libraries
import nltk
import re
import string
import html
import contractions
import langid
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from langid.langid import LanguageIdentifier

# Download from nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Feature Extraction Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Dummy Classifer 
from sklearn.dummy import DummyClassifier

# Modelling Libraries
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Evaluation Libraries
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import joblib
import pickle as pk







# Create a function to design customised color map
def custom_colormap(color, n=10):
    """
    Create a custom colormap with varying tints of the same color based on percentage.
    
    Parameters:
    color: str, the base color in hexadecimal format.
    n    : int, number of colors in the colormap.
    
    Returns:
    cmap : LinearSegmentedColormap, customised colormap object.
    """
    base_color = np.array(cl.to_rgb(color))
    colors = [base_color * (1 - i/n) + np.array([1, 1, 1]) * (i/n) for i in range(n)]
    colors = colors[::-1]
    
    return LinearSegmentedColormap.from_list('custom_colormap', colors, N=n)

#--------------------------------------------------------------------------------------------------------------------#

# Function to detect language
def detect_language(text):
    try:
        return detect(text) == 'en'
    except:
        return False  # Return False if language detection fails
    
#--------------------------------------------------------------------------------------------------------------------#

# Define customized tokenizer
def customized_tokenizer(sentence):
    
    # Remove HTML tags and entities
    sentence = html.unescape(sentence)
    sentence = re.sub(r'<[^>]+>', '', sentence)
    
    # Remove HTML white spaces \r<br/> and <br/>
    sentence = re.sub(r'(\r<br/>)|(<br/>)', ' ', sentence)
    
    # Remove punctuations
    sentence = re.sub(r'[^\w\s]', '', sentence)
    
    # Lowercase text
    sentence = sentence.lower()
    
    # Remove whitespaces
    sentence = re.sub(r'\s+', ' ', sentence)
    
    # Remove emails
    sentence = re.sub(r'\S*@\S*\s?', '', sentence)
    
    # Remove emojis
    sentence = sentence.encode('ascii', 'ignore').decode('ascii')
    
    # Remove special characters
    sentence = re.sub(r'[^A-Za-z\s]', '', sentence)
    
    # Remove numbers
    sentence = re.sub(r'[0-9]+', '', sentence)
    
    # Remove weblinks
    sentence = re.sub(r'http\S+', '', sentence)
    
    # Expand contractions
    sentence = contractions.fix(sentence)
    
    # Remove non-English text characters
    if langid.classify(sentence)[0] != 'en':
        sentence = ''
    
    # Remove English stopwords
    eng_stop_words=stopwords.words('english')
    # Append EDA insights driven stop words
    eng_stop_words.extend(['apartment','flat','edinburgh','could', 'would', 'x', 'caroline', 'stay']) 
    stop_words = set(eng_stop_words)
    tokens = sentence.split()
    sentence = ' '.join([word for word in tokens if word.lower() not in stop_words])
    
    # Perform text stemming
    lemmatizer = WordNetLemmatizer()
    sentence = ' '.join([lemmatizer.lemmatize(word, pos = 'v') for word in sentence.split()])
    
    # Tokenize cleaned sentence
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    
    return tokens

#--------------------------------------------------------------------------------------------------------------------#

# Define function that can perform downsampling on training data and make it perfectly balanced
def downsample_train(X_train, y_train, target, proportion):
    '''
    Function only works on target variable as binary column. 
    
    Returns downsampled X_train and y_train which
    has downsampled to specified proportion of original
    data, also makes sure y_train is balanced
    
    PARAMETERS:
    - X_train: dataframe, splitted feature training data
    - y_train: dataframe, splitted farget training data
    - target: str, target variable name
    - proportion: float, specified downsample proportion of total training data
    
    RETURNS:
    - X_train_sample: Downsampled X_train
    - y_train_sample: Downsampled y_train
    
    '''
    # Specify number of rows for each class 
    num_rows_each = round(proportion*X_train.shape[0]/2)
    
    # Temporarily concatenate X_train and y_train back to dataframe format
    df_train = pd.concat([X_train, y_train], axis=1)
    
    # Split classes
    df_pos = df_train[df_train[target] == 1]
    df_neg = df_train[df_train[target] == 0]
    
    # Resplit X and y for both two classes
    X_pos, y_pos = df_pos.drop(target, axis=1), df_pos[target]
    X_neg, y_neg = df_neg.drop(target, axis=1), df_neg[target]
    
    # Select randomized samples from each class
    X_pos_sample, y_pos_sample = resample(X_pos, y_pos, random_state=123, n_samples = num_rows_each, replace=False, stratify=y_pos)
    X_neg_sample, y_neg_sample = resample(X_neg, y_neg, random_state=123, n_samples = num_rows_each, replace=False, stratify=y_neg)
    
    # Concatenate back downsampled data
    X_train_sample = pd.concat([X_pos_sample, X_neg_sample], axis=0)
    y_train_sample = pd.concat([y_pos_sample, y_neg_sample], axis=0)
    
    return X_train_sample, y_train_sample

#--------------------------------------------------------------------------------------------------------------------#

def define_col_trans(input_text, vectorizer):
    '''
    Returns a ColumnTransformer which first performs a 
    passthrough on the numeric columns, then applies
    a vectorizer on the `text` column
    
    PARAMETERS:
    - input_text: str, to name the vectorizer tuple
    - vectorizer: Sklearn text vectorizer
    
    RETURNS:
    - col_trans: sklearn ColumnTransformer
    
    '''
    
    col_trans = ColumnTransformer([
        ('numeric', 'passthrough', numeric_columns), # numerical_columns defined above
        (input_text, vectorizer, 'comments') # 'comments' as review text feature column
    ])
    
    return col_trans

#--------------------------------------------------------------------------------------------------------------------#

def convert_to_array(sparse_matrix):
    '''
    Converts sparse matrix to dense array
    
    PARAMETERS:
    - sparse_matrix: scipy.sparse.csr_matrix or numpy array
    
    RETURNS:
    - If sparse_matrix is not a scipy.sparse.csr_matrix,
      sparse_matrix is returned. Else, returns the dense array
      form of sparse_matrix.
    
    '''
    
    if type(sparse_matrix) == csr_matrix:
    
        return sparse_matrix.toarray()
    
    else:
        return sparse_matrix
    
#--------------------------------------------------------------------------------------------------------------------#

# Define plotting function for extracting key features
def extract_key_words_plot(grid, ct, n_words, ct_prefix):
    '''
    Returns a barplot that shows the top words with
    highest coefficients in the selected best model
    in the fitted GridSearch
    
    PARAMETERS:
    - grid: Fitted resulted Grid
    - ct: Column Transformer
    - n_words: int, Number of words that will be shown in the plot
    - ct_prefix: str, The prefix of the word features
    
    RETURNS:
    - two barplots that show the top indicating words and listing features for the best model selected
    
    '''  
    # Extract the best model from grid search result
    best_model= grid.best_estimator_
    
    # Extract feature coefficients out of best model in the resulted grid
    coefficients = best_model.named_steps['model'].feature_importances_
    
    # Extract feature names from column transformer
    feature_names = ct.get_feature_names_out()
    
    # Generate column indicate if the feature is numerical or word related
    if_word=[]
    for i in range(len(feature_names)):
        if feature_names[i][0] == 'n':
            if_word.append(0)
        else:
            if_word.append(1)
    
    # create dataframe contains feature names with corresponding coefficients resulted from best model
    df_features = pd.DataFrame({'coefficients':coefficients, 'feature_names': feature_names, 'if_word': if_word})     
    
    # Split into numerical feature dataframe and word feature daraframe
    df_word = df_features[df_features['if_word']==1]
    df_num =  df_features[df_features['if_word']==0]
    
    # Order the features based on coefficients and remove prefix
    df_word = df_word.sort_values('coefficients', ascending=False).reset_index(drop=True).loc[:n_words-1]
    df_num = df_num.sort_values('coefficients', ascending=False).reset_index(drop=True).loc[:n_words-5]
    
    # Plot separate plots for word features and numerical features on two subplots
    
    plt.subplots(1, 2, figsize=(30,10))  # one row, two columns 
    
    # Word Features plot
    plt.subplot(1,2, 1)  # slot 1
    plt.barh(width=df_word['coefficients'], y=df_word['feature_names'].str.replace(ct_prefix+'__', ''), color='#FF5A5F')
    plt.title(f'Top {n_words} Indicating words', fontsize=25)
    plt.ylabel('Word Features', fontsize=18)
    plt.xlabel('Model Coefficients', fontsize=15)
    plt.xticks(rotation=45)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=20)
    
    # Numerical Features plot
    plt.subplot(1,2, 2)  # slot 2
    plt.barh(width=df_num['coefficients'], y=df_num['feature_names'].str.replace('numeric__', ''), color='#FF5A5F')
    plt.title(f'Top {n_words-4} Indicating Listing Features', fontsize=25)
    plt.ylabel('Listing Features', fontsize=18)
    plt.xlabel('Model Coefficients', fontsize=15)
    plt.xticks(rotation=45)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=20)
    ax=plt.gca()
    ax.yaxis.set_label_position("right")
    
    plt.tight_layout()
    # plt.savefig('feature_extraction.jpg', dpi =300)
    plt.show()

#--------------------------------------------------------------------------------------------------------------------#

# Define plotting function for extracting key features Logistic Regressions
def extract_key_words_plot_log(grid, ct, n_words, ct_prefix):
    '''
    Returns a barplot that shows the top words with
    highest coefficients in the selected best model
    in the fitted GridSearch
    
    PARAMETERS:
    - grid: Fitted resulted Grid
    - ct: Column Transformer
    - n_words: int, Number of words that will be shown in the plot
    - ct_prefix: str, The prefix of the word features
    
    RETURNS:
    - two barplots that show the top indicating words and listing features for the best model selected
    
    '''  
    # Extract the best model from grid search result
    best_model= grid.best_estimator_
    
    # Extract feature coefficients out of best model in the resulted grid
    coefficients = best_model.named_steps['model'].coef_[0]
    
    # Extract feature names from column transformer
    feature_names = ct.get_feature_names_out()
    
    # Generate column indicate if the feature is numerical or word related
    if_word=[]
    for i in range(len(feature_names)):
        if feature_names[i][0] == 'n':
            if_word.append(0)
        else:
            if_word.append(1)
    
    # create dataframe contains feature names with corresponding coefficients resulted from best model
    df_features = pd.DataFrame({'coefficients':coefficients, 'feature_names': feature_names, 'if_word': if_word})     
    
    # Split into numerical feature dataframe and word feature daraframe
    df_word = df_features[df_features['if_word']==1]
    df_num =  df_features[df_features['if_word']==0]
    
    # Order the features based on coefficients and remove prefix
    df_word_high = df_word.sort_values('coefficients', ascending=False).reset_index(drop=True).loc[:n_words-1]
    df_word_low = df_word.sort_values('coefficients', ascending=True).reset_index(drop=True).loc[:n_words-1]
    df_num_high = df_num.sort_values('coefficients', ascending=False).reset_index(drop=True).loc[:n_words-5]
    df_num_low = df_num.sort_values('coefficients', ascending=True).reset_index(drop=True).loc[:n_words-5]

    # Plot separate plots for word features and numerical features on two subplots
    
    plt.subplots(1, 2, figsize=(30,10))  # one row, two columns 
    
    # Word Features plot Top Indicators
    plt.subplot(1, 2, 1)  # slot 1
    plt.barh(width=df_word_high['coefficients'], y=df_word_high['feature_names'].str.replace(ct_prefix+'__', ''), color='#FF5A5F')
    plt.title(f'Top {n_words} Indicating phrases', fontsize=25)
    plt.ylabel('Word Features', fontsize=18)
    plt.xlabel('Model Coefficients', fontsize=15)
    plt.xticks(rotation=45)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=20)
    
    # Word Features plot Bottom Indicators
    plt.subplot(1, 2, 2)  # slot 1
    plt.barh(width=df_word_low['coefficients'], y=df_word_low['feature_names'].str.replace(ct_prefix+'__', ''), color='grey')
    plt.title(f'Bottom {n_words} Indicating phrases', fontsize=25)
    plt.ylabel('Word Features', fontsize=18)
    plt.xlabel('Model Coefficients', fontsize=15)
    plt.xticks(rotation=45)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=20)
    ax1=plt.gca()
    ax1.yaxis.set_label_position("right")

    # Numerical Features plot Top Indicators
    #plt.subplot(2, 2, 3)  # slot 2
    #plt.barh(width=df_num_high['coefficients'], y=df_num_high['feature_names'].str.replace('numeric__', ''), color='#FF5A5F')
    #plt.title(f'Top {n_words-4} Indicating Listing Features', fontsize=25)
    #plt.ylabel('Listing Features', fontsize=18)
    #plt.xlabel('Model Coefficients', fontsize=15)
    #plt.xticks(rotation=45)
    #plt.xticks(fontsize=15)
    #plt.yticks(fontsize=20)
    
    # Numerical Features plot Bottom Indicators
    #plt.subplot(2, 2, 4)  # slot 2
    #plt.barh(width=df_num_low['coefficients'], y=df_num_low['feature_names'].str.replace('numeric__', ''), color='grey')
    #plt.title(f'Bottom {n_words-4} Indicating Listing Features', fontsize=25)
    #plt.ylabel('Listing Features', fontsize=18)
    #plt.xlabel('Model Coefficients', fontsize=15)
    #plt.xticks(rotation=45)
    #plt.xticks(fontsize=15)
    #plt.yticks(fontsize=20)
    #ax2=plt.gca()
    #ax2.yaxis.set_label_position("right")


    plt.tight_layout()
    plt.savefig('grid4_features.jpg', dpi =300)
    plt.show()

#--------------------------------------------------------------------------------------------------------------------#

# Define plotting function for extracting key features Logistic Regressions
def extract_key_words_plot_subrating(grid, ct, n_words, ct_prefix, subrating_names):
    '''
    Returns a barplot that shows the top words with
    highest coefficients in the selected best model
    in the fitted GridSearch
    
    PARAMETERS:
    - grid: Fitted resulted Grid
    - ct: Column Transformer
    - n_words: int, Number of words that will be shown in the plot
    - ct_prefix: str, The prefix of the word features
    
    RETURNS:
    - two barplots that show the top indicating words and listing features for the best model selected
    
    '''  
    # Extract the best model from grid search result
    best_model= grid.best_estimator_
    
    # Extract feature coefficients out of best model in the resulted grid
    coefficients = best_model.named_steps['model'].feature_importances_
    
    # Extract feature names from column transformer
    feature_names = ct.get_feature_names_out()
    
    # Generate column indicate if the feature is numerical or word related
    if_word=[]
    for i in range(len(feature_names)):
        if feature_names[i][0] == 'n':
            if_word.append(0)
        else:
            if_word.append(1)
    
    # create dataframe contains feature names with corresponding coefficients resulted from best model
    df_features = pd.DataFrame({'coefficients':coefficients, 'feature_names': feature_names, 'if_word': if_word})     
    
    # Split into numerical feature dataframe and word feature daraframe
    df_word = df_features[df_features['if_word']==1]
    df_num =  df_features[df_features['if_word']==0]
    
    # Order the features based on coefficients and remove prefix
    df_word_high = df_word.sort_values('coefficients', ascending=False).reset_index(drop=True).loc[:n_words-1]
    df_num_high = df_num.sort_values('coefficients', ascending=False).reset_index(drop=True).loc[:n_words-5]

    # Plot separate plots for word features and numerical features on two subplots
    
    plt.subplots(1, 2, figsize=(30,10))  # one row, two columns 
    
    # Word Features plot Top Indicators
    plt.subplot(1, 2, 1)  # slot 1
    plt.barh(width=df_word_high['coefficients'], y=df_word_high['feature_names'].str.replace(ct_prefix+'__', ''), color='#FF5A5F')
    plt.title(f'Top {n_words} Indicating phrases for {subrating_names}', fontsize=25)
    plt.ylabel('Word Features', fontsize=18)
    plt.xlabel('Model Coefficients', fontsize=15)
    plt.xticks(rotation=45)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=20)

    # Numerical Features plot Top Indicators
    plt.subplot(1, 2, 2)  # slot 2
    plt.barh(width=df_num_high['coefficients'], y=df_num_high['feature_names'].str.replace('numeric__', ''), color='#FF5A5F')
    plt.title(f'Top {n_words-4} Indicating Listing Features for {subrating_names}', fontsize=25)
    plt.ylabel('Listing Features', fontsize=18)
    plt.xlabel('Model Coefficients', fontsize=15)
    plt.xticks(rotation=45)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=20)
    ax=plt.gca()
    ax.yaxis.set_label_position("right")


    plt.tight_layout()
    plt.savefig('location_result.jpg', dpi =300)
    plt.show()