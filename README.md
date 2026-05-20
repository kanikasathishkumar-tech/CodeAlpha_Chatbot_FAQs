# 🤖 FAQ Chatbot — Task 2

> NLP-powered chatbot that matches user questions to the most relevant FAQ answer using TF-IDF and Cosine Similarity.

---

## 📌 Project Overview

This project implements an FAQ Chatbot that uses Natural Language Processing (NLP) to understand user questions and return the best matching answer from a predefined FAQ dataset.

Built as part of the **CodeAlpha Internship — Task 2**.

---

## ✨ Features

- 📚 FAQ knowledge base with topic-specific Q&A pairs
- 🧹 Text preprocessing using NLTK (tokenization, stopword removal, lemmatization)
- 📊 TF-IDF vectorization + Cosine Similarity for question matching
- 🛡️ Confidence threshold to handle unmatched queries gracefully
- 💬 Interactive CLI chatbot loop

---

## 🗂️ Project Structure

```
CodeAlpha_Chatbot_FAQs/
│
├── chatbot.py        # Main chatbot script
├── README.md         # Project documentation
```

---

## ⚙️ Requirements

- Python 3.7+

### Install packages

```bash
pip install nltk scikit-learn numpy
```

### Download NLTK data (run once)

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
```

---

## 🚀 How to Run

```bash
python chatbot.py
```

---

## 💡 How It Works

### 1. FAQ Data Collection
A list of topic-specific question-answer pairs is defined in the script. You can easily add your own FAQs to the `faq_data` list.

### 2. Text Preprocessing (NLTK)
Each question is cleaned through a pipeline:
- Lowercasing
- Punctuation removal
- Tokenization
- Stopword removal
- Lemmatization

### 3. TF-IDF Vectorization
Converts preprocessed text into numerical vectors. Unique words get higher weights.

### 4. Cosine Similarity Matching
The user query is compared against all FAQ vectors. The FAQ with the highest similarity score above the threshold (`0.15`) is returned as the answer.

---

## 🖥️ Example

```
You: What is Python?
Bot [matched: 'What is Python?' | score: 0.87]:
  Python is a high-level, interpreted programming language...

You: how do i install it
Bot [matched: 'How do I install Python?' | score: 0.61]:
  You can install Python by going to https://www.python.org/downloads/...

You: quit
Bot: Goodbye! Happy coding!
```

---

## 🔧 Customization

### Add your own FAQs
Open `chatbot.py` and add to the `faq_data` list:

```python
{
    "question": "Your question here?",
    "answer": "Your answer here."
},
```

### Adjust confidence threshold
```python
answer, matched_q, score = get_best_match(user_input, threshold=0.15)
```
- Lower (e.g. `0.05`) → more lenient
- Higher (e.g. `0.30`) → stricter

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `Import 'nltk' could not be resolved` | Run `pip install nltk` in the VS Code terminal |
| `ModuleNotFoundError: No module named 'sklearn'` | Run `pip install scikit-learn` |
| `LookupError: punkt not found` | Run the NLTK download command above |
| `pip not recognized` | Use `pip3`, or reinstall Python with **Add to PATH** checked |

---

## 🧰 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| NLTK | Text preprocessing |
| scikit-learn | TF-IDF & cosine similarity |
| NumPy | Numerical operations |

---

*CodeAlpha Internship — Task 2: FAQ Chatbot*
