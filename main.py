import nltk
import numpy as np
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


def tokenize(sentence: str):
    return nltk.word_tokenize(sentence)


def stemming(word: str):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence: str, all_words: str):
    tokenized_sentence = [stemming(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)

    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1

    return bag


sentence = ['hello', 'how', 'are', 'you']
words = ['hi', 'hello', 'I', 'you', 'bye', 'thank', 'cool']
bag = bag_of_words(sentence, words)
# print(bag)

# words = ['organize', 'organizes', 'Organizing']
# stemmed_words = [stemming(w) for w in words]
# print(stemmed_words)
