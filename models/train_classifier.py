import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
import pickle
from utils import tokenize, StartingVerbExtractor, NamedEntityExtractor


def load_data(database_filepath):
    """
    Load data from database.
    Input
        database_filepath (str): database file path
    Output
        X (Series): the messages in English
        Y (DataFrame): the categories in binary values
        category_names (list): categories names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Message', con=engine)
    X = df['message']
    Y = df[df.columns[4:]]
    category_names = Y.columns.tolist()
    return X, Y, category_names


def build_model():
    """
    Build the machine learning model with Pipeline and GridSearch.
    Input
        None
    Output
        cv: machine learning model
    """
    # Build model pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor()), # this is a custom transformer
            
            ('named_entity', NamedEntityExtractor()) # this is a custom transformer
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])

    # Specify parameters for grid search
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        # 'features__text_pipeline__vect__max_df': (0.75, 1.0),
        # 'features__text_pipeline__tfidf__use_idf': (True, False),
    #     'clf__estimator__n_estimators': [50, 100, 200],
    #     'clf__estimator__min_samples_split': [2, 5, 10],
    #     'clf__estimator__max_depth': [3, 50, None],
    #     'features__transformer_weights': (
    #         {'text_pipeline': 1, 'starting_verb': 1, 'named_entity': 1},
    #         {'text_pipeline': 0.5, 'starting_verb': 1, 'named_entity': 1},
    #         {'text_pipeline': 1, 'starting_verb': 0.5, 'named_entity': 0.5},
    #     )
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='accuracy', cv=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Report accuracy, precision, recall, and F1 scores of model on testing dataset.
    Input
        model: machine learning model
        X_test (DataFrame): testing features
        Y_test (DataFrame or Series): testing labels
        category_names (list): list of category names
    Output
        None
    """
    # Make predictions based on the trained model
    Y_pred = model.predict(X_test)

    # Scores for each category
    for i, col in enumerate(category_names):
        print(f"\nCategory: {col} ---------------------------")
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))

    # Overall scores
    accuracy = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred, average='micro')
    precision = precision_score(Y_test, Y_pred, average='micro')
    recall = recall_score(Y_test, Y_pred, average='micro')
    print("\nOverall Scores: ---------------------------")
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")


def save_model(model, model_filepath):
    """
    Save the model into a pickle file.
    Input
        model: machine learning model
        model_filepath (str): the filepath to save the model
    Output
        None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


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