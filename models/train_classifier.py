import sys
import pandas as pd
import numpy as np
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import re
import pickle


from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """Load the filepath and return the data"""
    name = 'sqlite:///' + database_filepath
    engine = create_engine(name)
    df = pd.read_sql_table('df',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    """tokenize and transform input text to output cleaned text"""
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """Starting Verb Extractor class
    
    It extracts the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """
    Build Model function
    
    It output is a Scikit ML Pipeline that processes text messages
    and applies a classifier.
    """
    model = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    return model



def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function
    
    It applies the ML pipeline to a test set and prints out
    model performance
    
    Arguments:
        model - Scikit ML Pipeline
        X_test - test features
        Y_test - test labels
        category_names - label names (multi-output)
    """
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns = Y_test.columns)
    for column in Y_test.columns:
        print('Testing result of the model tuned with AdaBoost and a Custom Transformer')
        print('------------------------------------------------------\n')
        print('FEATURE NAME: {}\n'.format(column))
        print(classification_report(Y_test[column],y_pred[column]))
    

def save_model(model, model_filepath):
    """
    Save Model function
    
    It saves trained model as Pickle file to be used later.
    
    Arguments:
        model - Pipeline
        model_filepath - location of where to save .pkl file
    
    """
    filename = 'model.pkl'
    pickle.dump(model, open(filename, 'wb'))


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