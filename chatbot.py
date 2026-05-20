"""
FAQ Chatbot - Task 2
Packages to install:
    pip install nltk scikit-learn numpy
Then run once to download NLTK data:
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
"""

import nltk
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Download required NLTK data (run once) ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# -----------------------------------------------
# 1. FAQ DATA (topic: Python Programming)
# -----------------------------------------------
faq_data = [
    {
        "question": "What is Python?",
        "answer": "Python is a high-level, interpreted programming language known for its simple syntax and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming."
    },
    {
        "question": "How do I install Python?",
        "answer": "You can install Python by going to https://www.python.org/downloads/ and downloading the latest version. Run the installer and make sure to check 'Add Python to PATH' during installation."
    },
    {
        "question": "What is a variable in Python?",
        "answer": "A variable in Python is a name that refers to a value stored in memory. You create a variable by assigning a value using the = operator, e.g., x = 10 or name = 'Alice'."
    },
    {
        "question": "What is a list in Python?",
        "answer": "A list is an ordered, mutable collection of items. You create a list using square brackets, e.g., my_list = [1, 2, 3]. Lists can contain items of different types."
    },
    {
        "question": "What is a dictionary in Python?",
        "answer": "A dictionary is an unordered collection of key-value pairs. You create one using curly braces, e.g., my_dict = {'name': 'Alice', 'age': 25}. Keys must be unique and immutable."
    },
    {
        "question": "How do I write a for loop in Python?",
        "answer": "A for loop iterates over a sequence. Example:\n  for i in range(5):\n      print(i)\nThis prints numbers 0 through 4."
    },
    {
        "question": "What is a function in Python?",
        "answer": "A function is a reusable block of code defined using the 'def' keyword. Example:\n  def greet(name):\n      return f'Hello, {name}!'"
    },
    {
        "question": "What is pip?",
        "answer": "pip is Python's package installer. You use it to install third-party libraries, e.g., 'pip install numpy' installs the NumPy library."
    },
    {
        "question": "What is a class in Python?",
        "answer": "A class is a blueprint for creating objects. It bundles data (attributes) and functions (methods) together. Example:\n  class Dog:\n      def __init__(self, name):\n          self.name = name"
    },
    {
        "question": "How do I handle exceptions in Python?",
        "answer": "Use try-except blocks to handle exceptions. Example:\n  try:\n      result = 10 / 0\n  except ZeroDivisionError:\n      print('Cannot divide by zero!')"
    },
    {
        "question": "What is a tuple in Python?",
        "answer": "A tuple is an ordered, immutable collection. Created with parentheses: my_tuple = (1, 2, 3). Unlike lists, tuples cannot be changed after creation."
    },
    {
        "question": "What is the difference between == and is?",
        "answer": "'==' checks if two values are equal. 'is' checks if two variables refer to the same object in memory. For example, [1,2] == [1,2] is True, but [1,2] is [1,2] is False."
    },
]


# -----------------------------------------------
# 2. TEXT PREPROCESSING
# -----------------------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    """Lowercase, remove punctuation, tokenize, remove stopwords, lemmatize."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# Preprocess all FAQ questions
faq_questions = [item["question"] for item in faq_data]
faq_answers   = [item["answer"]   for item in faq_data]
processed_questions = [preprocess(q) for q in faq_questions]


# -----------------------------------------------
# 3. TF-IDF VECTORIZER + COSINE SIMILARITY
# -----------------------------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_questions)

def get_best_match(user_query, threshold=0.15):
    """
    Preprocess the query, compute cosine similarity against all FAQs,
    and return the best matching answer (or a fallback message).
    """
    processed_query = preprocess(user_query)
    query_vec = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score < threshold:
        return (
            "Sorry, I couldn't find a relevant answer to your question. "
            "Please try rephrasing or ask something else about Python.",
            None,
            best_score
        )
    return faq_answers[best_idx], faq_questions[best_idx], best_score


# -----------------------------------------------
# 4. CHATBOT LOOP (CLI interface)
# -----------------------------------------------
def run_chatbot():
    print("=" * 55)
    print("  FAQ Chatbot — Python Programming")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 55)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'bye'):
            print("Bot: Goodbye! Happy coding!")
            break

        answer, matched_q, score = get_best_match(user_input)

        if matched_q:
            print(f"\nBot [matched: '{matched_q}' | score: {score:.2f}]:")
        else:
            print(f"\nBot [no match | score: {score:.2f}]:")
        print(f"  {answer}")


if __name__ == "__main__":
    run_chatbot()