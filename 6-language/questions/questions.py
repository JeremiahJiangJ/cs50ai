import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = {}

    for file_name in os.listdir(directory):
        with open(os.path.join(directory, file_name), encoding="UTF-8") as ofile:
            files[file_name] = ofile.read()
    
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    tokens = [word.lower() for word in nltk.word_tokenize(document)]
    punctuation = string.punctuation
    stopwords = nltk.corpus.stopwords.words("english")
    res = [token for token in tokens if (token not in punctuation and token not in stopwords)]

    return res


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    num_docs = len(documents)
    num_docs_with_word = {}
    
    for document in documents:
        for word in set(documents[document]):
            if word not in num_docs_with_word:
                num_docs_with_word[word] = 1
            else:
                num_docs_with_word[word] += 1

    idfs = {}
    for word in num_docs_with_word:
        idfs[word] = math.log(num_docs/num_docs_with_word[word])

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idfs = {}

    for file_name in files:
        score = 0
        for word in query:
            content = files[file_name]
            if word in content:
                tf = content.count(word)
                score += tf * idfs[word]
        if score != 0:
            tf_idfs[file_name] = score

    sorted_files = [file_name for file_name in sorted(tf_idfs, key=lambda x:tf_idfs[x], reverse=True)]

    return sorted_files[:n]
   
def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """

    sentence_scores = {}
    for sentence in sentences:
        score = 0
        for word in query:
            content = sentences[sentence]
            if word in content:
                score += idfs[word]
        if score != 0:
            qt_density = sum([content.count(i) for i in query]) / len(content)
            sentence_scores[sentence] = (score, qt_density)

    sorted_sentences = [sentence for sentence in sorted(sentence_scores, key=lambda x: (sentence_scores[x][0], sentence_scores[x][1]), reverse = True)]
    return sorted_sentences[:n]

if __name__ == "__main__":
    main()
