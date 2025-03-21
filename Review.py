import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#  Corrected Windows file path (Use your actual path)
file_path = r"C:\Users\Ramakrishna\Downloads\PROJECT-WORK\movie_reviews.csv"

#  Function to load dataset
def load_dataset(filepath):
    try:
        df = pd.read_csv(filepath)
        print(" Dataset loaded successfully!")
        return df
    except FileNotFoundError:
        print(f" ERROR: File not found at {filepath}")
        exit()

#  Load dataset
df = load_dataset(file_path)

#  Check dataset structure
print(df.head())

#  Check if required columns exist
if 'text' not in df.columns or 'tag' not in df.columns:
    print("❌ ERROR: Expected columns 'text' and 'tag' not found in dataset.")
    exit()

#  Text preprocessing function
def preprocess_text(text):
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    tokens = tokenizer.tokenize(str(text).lower())  # Convert to lowercase & tokenize
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

#  Apply text preprocessing
df['cleaned_review'] = df['text'].astype(str).apply(preprocess_text)

#  Split data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['cleaned_review'], df['tag'], test_size=0.2, random_state=42
)

#  Feature extraction function (Vectorization)
def vectorize_text(train_texts, test_texts, method='tfidf'):
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7)
    else:
        vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7)
    
    train_vectors = vectorizer.fit_transform(train_texts)
    test_vectors = vectorizer.transform(test_texts)
    return train_vectors, test_vectors, vectorizer

#  Vectorize the text data
train_vectors, test_vectors, vectorizer = vectorize_text(train_texts, test_texts, method='tfidf')

#  Train and evaluate model function
def train_model(train_vectors, train_labels, test_vectors, test_labels):
    model = MultinomialNB()
    model.fit(train_vectors, train_labels)
    predictions = model.predict(test_vectors)
    accuracy = accuracy_score(test_labels, predictions)
    cm = confusion_matrix(test_labels, predictions)
    return accuracy, cm

#  Train and evaluate the Naïve Bayes model
accuracy, cm = train_model(train_vectors, train_labels, test_vectors, test_labels)

#  Print final results
print(f' Model Accuracy: {accuracy * 100:.2f}%')
print(' Confusion Matrix:\n', cm)
