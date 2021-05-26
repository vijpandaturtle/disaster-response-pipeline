#Import required packages 
import sys
import nltk
import numpy as np
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sqlalchemy import create_engine
import re
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
import pickle

def load_data(database_filepath):
    '''Loading data from the SQL database'''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Disaster', con=engine)
    X = df['message']
    Y = df[df.columns[5:]]
    added = pd.get_dummies(df[['related','genre']])
    y = pd.concat([Y, added], axis=1)
    return X,y

def tokenize(text):
    ''''Tokenize, remove punctuation and lemmatize text data'''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    #Remove punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
    #Lemmatize 
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_token = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_token)
    return clean_tokens

def build_model():
    '''Building the model pipeline'''
    multioutputclf = MultiOutputClassifier(RandomForestClassifier())
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', multioutputclf)
    ])
    parameters = {'clf__estimator__max_depth': [10, 50, None],
              'clf__estimator__min_samples_leaf':[2, 5, 10]}
    model = GridSearchCV(pipeline, parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''Fitting and evaluating the model'''
    y_pred = model.predict(X_test)
    class_report = classification_report(y_test, y_pred, target_names=category_names)
    print(class_report)


def save_model(model, model_filepath):
    '''Saving the model to a .pkl file'''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()