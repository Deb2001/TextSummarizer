import numpy as np
import pandas as pd
import nltk
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# nltk.download('punkt') # one time execution
# nltk.download('stopwords')

def convertToList(text):
    senlist=[]
    s=''
    for ch in text:
        if ch=='.':
            s=s+ch
            s=s.lstrip()
            senlist.append(s)
            s=''
        else:
            s=s+ch
    return senlist

def extractiveSumm(article):

    senlist=convertToList(article)
    print(senlist)
    sentences = []
    for s in senlist:
        sentences.append(sent_tokenize(s))
    sentences = [y for x in sentences for y in x]
    print()

    #Extract word vectors
    word_embeddings = {}
    f = open('glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()

    #Remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ",regex=False)

    #Make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]

    stop_words = stopwords.words('english')
    #Function to remove stopwords
    def remove_stopwords(sen):
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new
    #Remove stopwords from the sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    print(clean_sentences)

    #Extract word vectors
    word_embeddings = {}
    f = open('glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()

    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)

    #Similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
    print()
    print(sim_mat)

    #Graph From Similarity Matrix
    nx_graph = nx.from_numpy_array(sim_mat)

    print()
    print(nx_graph)
    scores = nx.pagerank(nx_graph)
    print()
    print(scores)


    print()
    summlist=[]
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(senlist)), reverse=True)
    for i in range(len(ranked_sentences)//2):
        summlist.append(ranked_sentences[i][1])
        print(ranked_sentences[i][1], end=" ")
    summary=' '.join([str(elem) for i,elem in enumerate(summlist)])
    return summary

article="Gautam Adani has surged back into the top 20 wealthiest individuals globally, propelled by a consecutive market rally that boosted the combined market value of his enterprises by 1.33 lakh crore. Currently occupying the 19th position on the Bloomberg Billionaires Index, Adani has seen his overall net worth rise by $6.5 billion, according to the latest update from Bloomberg. Nonetheless, his total net worth for the year-to-date period remains $53.8 billion lower, as reported by ET."
summary=extractiveSumm(article)
print(summary)