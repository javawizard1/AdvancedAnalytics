import os, json, gzip 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet, stopwords

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer

import datetime
import logging
import sys
import warnings

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


warnings.filterwarnings('ignore')

TIMESTMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = '.\output_task2_' + TIMESTMP + '\\'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


logger = logging.getLogger()

########################################################################################################################
def get_wordnet_pos(word):
  tag = nltk.pos_tag([word])[0][1][0].upper()
  tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

  return tag_dict.get(tag, wordnet.NOUN)


########################################################################################################################
def lemmatizeSentence(sentence):
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(sentence)
    return ' '.join([lemmatizer.lemmatize(w) for w in words])


########################################################################################################################
def preprocessData(df):
    logger.info('Pre Processing Data')

    # Remove non-alphanumeric characters and lowercase the text
    df['review_cleaned'] = df['reviewText'].str.replace('[^a-zA-Z0-9 ]', '', regex=True).str.lower()
    logger.info('Non Alphanumerics Removed')

    # Tokenization and stopwords removal
    stop_words = set(stopwords.words('english'))
    df['review_cleaned'] = df['review_cleaned'].str.split().apply(lambda x: [word for word in x if word not in stop_words])
    logger.info('Tokenization and stopwords removal Complete')

    logger.info('Joining Words back to String')
    df['review_cleaned'] = df['review_cleaned'].apply(' '.join)
    logger.info('Joining Words back to String Complete')

    logger.info('Starting Lemmatization')

    # Lemmatization
    lemmatizer = nltk.stem.WordNetLemmatizer()
    df['lemmatized_text'] = df['review_cleaned'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in nltk.word_tokenize(x)]))
    logging.info('Lemmatization Complete')

    return df


########################################################################################################################
def categoryBarPlot(myData):
    
    # Count the occurrences of each category in 'overall'
    category_counts = myData['overall'].value_counts()

    # Create the bar plot
    plt.figure(figsize=(8, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values)

    # Setting the title and labels
    plt.title('Bar Plot of Overall Categories')
    plt.xlabel('Categories')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'categoryBarPlot.png'))

########################################################################################################################
# Load and preprocess data
def loadData(file_name):
    logger.info('Loading Data')

    data = []
    with gzip.open(file_name) as f:
        for line in f:
            data.append(json.loads(line.strip()))

    df = pd.DataFrame.from_dict(data)
    df = df.dropna()

    logger.info(f'{len(df)} rows loaded')

    categoryBarPlot(df)
    
    # Reduce the dataset so that we have an equal amount of observations 
    # for each value
    sampleSize = df['overall'].value_counts().min()
    logger.info('Computed Sample Size is ' + str(sampleSize))
    
    df_equal_overall = []
    for rating in df['overall'].unique():
        df_equal_overall.append(df[df['overall'] == rating].sample(sampleSize))

    df = pd.concat(df_equal_overall)
    return df

########################################################################################################################
def plotConfusionMatrix(cm, vals, name):
    # Calculate the percentage of the whole for each cell
    total = np.sum(cm)
    cm_percentage = (cm / total) * 100

    # Annotate with both the count and the percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            c = cm[i, j]
            p = cm_percentage[i, j]
            annot[i, j] = f'{c}\n({p:.2f}%)'

    # Plot the confusion matrix
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Greens', xticklabels=vals, yticklabels=vals)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {name}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{name}confusionMatrix.png'))

########################################################################################################################
def create_tensorflow_model(tokenizer, max_sequence_length):
    # Vocabulary size (adding 1 because index 0 is reserved and not assigned to any word)
    vocab_size = len(tokenizer.word_index) + 1

    # Choose an embedding length
    embedding_length = 166

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_length, input_length=max_sequence_length))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Adjust based on your classification task

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    return model




########################################################################################################################
########################################################################################################################
def main():
    print(__name__)


    electronics = 'reviews_Electronics_5.json.gz'
    books = 'reviews_Books_5.json.gz'
    garden = 'reviews_Patio_Lawn_and_Garden_5.json.gz'
   
    dataToLoad = garden    
  
    
    # Setup logging to the screen and a file
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the logging level
    
    # Create a file handler for writing logs to a file
    file_handler = logging.FileHandler(f'{OUTPUT_DIR}\Task2_log_{TIMESTMP}.txt')
    file_handler.setLevel(logging.INFO)  # Set the logging level for the file handler
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Create a stream handler for writing logs to sys.stdout (console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)  # Set the logging level for the stream handler
    stream_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(stream_format)
    
    # Add the handlers to the logger. The check is needed because Spyder
    # uses persistent data so we would end up with redundant loggers
    if len(logger.handlers) == 0 :
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    
    
    startTime = datetime.datetime.now()
    
    
    df = loadData(dataToLoad)
    
    # Load and preprocess the data
    df = preprocessData(df)
    
    
    # Define pipelines for different models
    nb = Pipeline([
        ('vectorize', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
        ])
    
    sgd = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(n_jobs=4)),
        ])
    
    logreg = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer()),
        # sag solver because it performs well on larger datasets
        ('clf', LogisticRegression(max_iter=500,solver='sag',n_jobs=4)),
        ])
    
    # Tokenize the text and calculate the length of each review
    df['review_length'] = df['review_cleaned'].apply(lambda x: len(x.split()))

    # Descriptive statistics for review lengths
    print(df['review_length'].describe())

    # Plotting the distribution of review lengths
    plt.figure(figsize=(10,6))
    plt.hist(df['review_length'], bins=50, alpha=0.7)
    plt.title('Distribution of Review Lengths')
    plt.xlabel('Length of Reviews')
    plt.ylabel('Number of Reviews')
    plt.savefig(os.path.join(OUTPUT_DIR, 'reviewLength.png'))

    # Determining the 90th percentile for sequence length
    max_length = np.percentile(df['review_length'], 90)
    print(f"90th Percentile Length: {max_length}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df['review_cleaned'], df['overall'], 
                                                        test_size=0.2, random_state = 44, stratify=df['overall'])
    
    vals = df['overall'].unique()
    print(vals)

    # Train and evaluate models
    for model, name in [(nb, "Naive Bayes"), (sgd, "SGD"), (logreg, "Logistic Regression")]:
        logger.info(f'Running {name} model')
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        logger.info(f"\nResults for {name}:")
        
        logger.info(classification_report(y_test, y_pred))
        logger.info(accuracy_score(y_test, y_pred))
        
        cm = confusion_matrix(y_test, y_pred)
        plotConfusionMatrix(cm,vals,name)
        logger.info(cm) 

        

    # Inside the main function, after data preprocessing
    # Tokenization
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df['review_cleaned'])
    X = tokenizer.texts_to_sequences(df['review_cleaned'])
    X = pad_sequences(X, maxlen=100)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, df['overall'], test_size=0.2, random_state=42)

    # Build the TensorFlow model
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=64, input_length=100))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Use 'softmax' if you have more than 2 classes


    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # Use 'categorical_crossentropy' for multi-class

    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=32)

    y_pred = model.predict(X_test)
    
    # Check if y_test is binary or multi-class
    if len(np.unique(y_test)) > 2:
        # For multi-class classification
        y_pred = np.argmax(y_pred, axis=1)
    else:
        # For binary classification, convert probabilities to binary class predictions
        y_pred = (y_pred > 0.5).astype(int)

    # Ensure y_test is an array of labels (not one-hot encoded if it's multi-class)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plotConfusionMatrix(cm, np.unique(y_test), "TensorFlow Model")

    logger.info(f"\nResults for TensorFlow:")
        
    logger.info(classification_report(y_test, y_pred))
    logger.info(accuracy_score(y_test, y_pred))


    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    logger.info(f"Test Accuracy: {accuracy}")


    stopTime = datetime.datetime.now()
    elapsedTime = stopTime - startTime        
    logger.info('\n Elapsed Time: ' + str(elapsedTime))
    
    file_handler.close()
    logger.removeHandler(file_handler)

if __name__ == '__main__':
    main()
