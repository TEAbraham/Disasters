import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
from sklearn.externals import joblib
from sqlalchemy import create_engine

import spacy
from sklearn.feature_extraction.text import CountVectorizer
from plotly.offline import plot
import random
import pickle

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Message', engine)

# load model
model = pickle.load(open("../models/classifier.pkl", 'rb'))
# model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # calculate number of messages by genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = [x.title() for x in genre_counts.index]

    # calculate number of messages by category
    category_counts = df[df.columns[4:]].sum().sort_values(ascending=False)
    category_names = [x.replace('_', ' ').title() for x in category_counts.index]
    
    # calculate word cloud
    spacy_nlp = spacy.load('en_core_web_sm')
    message_all = df['message']
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=(1,1))
    vect = vectorizer.fit_transform(message_all)
    words = vectorizer.get_feature_names()
    word_counts = vect.toarray().sum(axis=0)
    word_counts_df = pd.Series(word_counts, index=words).sort_values(ascending=False)
    query = pd.to_numeric(word_counts_df.index, errors='coerce').isna()
    word_counts_df = word_counts_df[query]
    top = word_counts_df[:100]
    words = top.index
    colors = [plotly.colors.DEFAULT_PLOTLY_COLORS[random.randrange(1, 10)] for i in range(len(words))]
    weights = top.values ** 0.8 / 9
    
    # create visuals
    graphs = [
        # Graph 1
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker=dict(
                        color='rgba(55, 128, 191, 0.7)',
                        line=dict(
                            color='rgba(55, 128, 191, 1.0)',
                            width=2,
                        )
                    )
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        # Graph 2
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts,
                    marker=dict(
                        color='rgba(50, 171, 96, 0.7)',
                        line=dict(
                            color='rgba(50, 171, 96, 1.0)',
                            width=2,
                        )
                    )
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                },
                'margin': {
                    'l': 100,
                    'r': 100,
                    't': 100,
                    'b': 200
                },
            }
        },

        # Graph 3
        {
            'data': [
                Scatter(
                    x=random.sample(range(len(words)), k=len(words)),
                    y=random.sample(range(len(words)), k=len(words)),
                    mode='text',
                    text=words,
                    marker={
                        'opacity': 0.3
                    },
                    textfont={
                        'size': weights,
                        'color': colors
                    }
                )
            ],

            'layout': {
                'xaxis': {
                    'showgrid': False, 
                    'showticklabels': False, 
                    'zeroline': False,
                    'range': [-len(words)*0.1, len(words)*1.15]
                },
                'yaxis': {
                    'showgrid': False, 
                    'showticklabels': False, 
                    'zeroline': False,
                    'range': [-len(words)*0.1, len(words)*1.15]
                },
                'title': '100 Most Frequent Words Appearing in Messages <br> Larger Text Represents Higher Frequency',
            #     'height' : 600,
            #     'width' : 1200,
            }
        },
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')
    print(query)
    print(type(query))

    # use model to predict classification for query
    classification_labels = model.predict(pd.Series([query], name='message'))[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # render the go.html
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()