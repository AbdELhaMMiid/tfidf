pip install transformers

# link of colab
# https://colab.research.google.com/drive/1c432PHT98QcD3u59Mo3PRKMsnW-iMyQ_?usp=sharing 

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from transformers import pipeline
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Function to generate text using GPT-Neo model
def generate_text(prompt, model_name='EleutherAI/gpt-neo-2.7B', max_length=50):
    gen = pipeline('text-generation', model=model_name)
    generated_text = gen(prompt, max_length=max_length, do_sample=True)
    return generated_text[0]['generated_text']

# Function to preprocess text
def preprocess_text(text):
    # Cleaning data from symbols and characters
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    # cleaned_text = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', text)
    # cleaned_text = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', text)

    # Normalization: convert to lowercase
    cleaned_text = cleaned_text.lower()
    # Tokenization
    words = word_tokenize(cleaned_text)
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in lemmatized_words if word not in stop_words]
    return ' '.join(filtered_words)

# Generate text for different prompts
prompts = ["Sports are", "AI is"]
generated_texts = [generate_text(prompt, max_length=100) for prompt in prompts]

# Preprocess generated texts
preprocessed_texts = [preprocess_text(text) for text in generated_texts]


# Print preprocessed texts
print("Preprocessed texts:")
for i, text in enumerate(preprocessed_texts):
    print(f"Prompt: {prompts[i]}")
    print(text)
    print()

# Calculate TF for each word for all documents
def calculate_tf(document):
    words = document.split()
    word_count = len(words)
    tf = {}
    for word in set(words):
        tf[word] = words.count(word) / word_count
    return tf

# Print TF values
print("\nTF values:")
for i, text in enumerate(preprocessed_texts):
    print(f"TF values for Prompt: {prompts[i]}")
    print(calculate_tf(text))
    print()

# Calculate IDF for each word
def calculate_idf(documents):
    total_documents = len(documents)
    all_words = set(word for document in documents for word in document.split())
    idf = {}
    for word in all_words:
        doc_count = sum(1 for doc in documents if word in doc)
        idf[word] = np.log(total_documents +1 / (doc_count + 1)) + 1  # Adding 1 to avoid division by zero and ensure positive IDF values
    return idf

# Print IDF values
print("\nIDF values:")
print(calculate_idf(preprocessed_texts))

# Calculate TF-IDF using scikit-learn
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_texts)
feature_names = tfidf_vectorizer.get_feature_names_out()
sklearn_tfidf = [{feature_names[j]: tfidf_matrix[i, j] for j in range(len(feature_names))} for i in range(len(preprocessed_texts))]

# Print TF-IDF values from scikit-learn
print("\nTF-IDF using scikit-learn:")
for i, tfidf in enumerate(sklearn_tfidf):
    print(f"TF-IDF values for Prompt: {prompts[i]}")
    print(tfidf)
    print()

# Calculate TF-IDF from scratch
def calculate_tfidf(documents):
    tfidf = []
    tf = [calculate_tf(doc) for doc in documents]
    idf = calculate_idf(documents)
    for doc_tf in tf:
        doc_tfidf = {word: tf_value * idf[word] for word, tf_value in doc_tf.items()}
        tfidf.append(doc_tfidf)
    return tfidf

# Normalize TF-IDF
def normalize_tfidf(tfidf):
    normalized_tfidf = []
    for doc_tfidf in tfidf:
        norm = np.linalg.norm(list(doc_tfidf.values()))
        normalized_tfidf.append({word: value / norm for word, value in doc_tfidf.items()})
    return normalized_tfidf

# Calculate TF-IDF from scratch
tfidf_scratch = calculate_tfidf(preprocessed_texts)
normalized_tfidf_scratch = normalize_tfidf(tfidf_scratch)

# Print TF-IDF values from scratch
print("\nTF-IDF from scratch:")
for i, tfidf in enumerate(normalized_tfidf_scratch):
    print(f"TF-IDF values for Prompt: {prompts[i]}")
    print(tfidf)
    print()

