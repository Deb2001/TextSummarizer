import streamlit as st
# from urllib.request import urlopen 
# import json 

import numpy as np
import pandas as pd
# import nltk
# import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

from transformers import BartTokenizer, BartForConditionalGeneration
from simplet5 import SimpleT5

#**************************************************************************************************
#                       Extractive                 
#**************************************************************************************************

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

def extractiveSumm(article, progress_bar):

    global progress

    senlist=convertToList(article)
    print(senlist)
    sentences = []
    for s in senlist:
        sentences.append(sent_tokenize(s))
    sentences = [y for x in sentences for y in x]
    print()

    progress+=0.1
    progress_bar.progress(progress)

    # Extract word vectors
    word_embeddings = {}
    f = open('glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()

    #remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ",regex=False)

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]

    stop_words = stopwords.words('english')
    # function to remove stopwords
    def remove_stopwords(sen):
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new
    # remove stopwords from the sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    print(clean_sentences)

    progress+=0.1
    progress_bar.progress(progress)

    # Extract word vectors
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
    # print()
    # print(sentence_vectors)

    progress+=0.1
    progress_bar.progress(progress)

    # similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
    print()
    print(sim_mat)
    nx_graph = nx.from_numpy_array(sim_mat)

    print()
    print(nx_graph)
    scores = nx.pagerank(nx_graph)
    print()
    print(scores)

    progress+=0.1
    progress_bar.progress(progress)

    print()
    summlist=[]
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(senlist)), reverse=True)
    for i in range(len(ranked_sentences)//2):
        summlist.append(ranked_sentences[i][1])
        print(ranked_sentences[i][1], end=" ")
    summary=' '.join([str(elem) for i,elem in enumerate(summlist)])
    print(summary)

    progress+=0.1
    progress_bar.progress(progress)

    return summary


#**************************************************************************************************
#**************************************************************************************************
#                       Abstractive                 
#**************************************************************************************************

def abstractiveSummPre(article, progress_bar):

    global progress

    # Load BART model and tokenizer
    model_name = 'facebook/bart-large-cnn'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Tokenize and encode the article
    inputs = tokenizer.encode(article, return_tensors='pt', max_length=1024, truncation=True)

    progress+=0.15
    progress_bar.progress(progress)

    # Generate summary
    summary_ids = model.generate(inputs, num_beams=4, max_length=150, early_stopping=True)

    progress+=0.15
    progress_bar.progress(progress)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print()
    print(summary)

    progress+=0.2
    progress_bar.progress(progress)
    return summary

#**************************************************************************************************

def abstractiveSumm(article, progress_bar):
    global progress

    # instantiate
    model = SimpleT5()
    progress+=0.25
    progress_bar.progress(progress)

    # load trained T5 model
    model.load_model("t5","AbstractiveSummarization", use_gpu=False)

    # predict
    summlist = model.predict(article)
    summary= summlist[0]
    print(summary)
    progress+=0.25
    progress_bar.progress(progress)
    return summary

#**************************************************************************************************

progress = 0.0

def main():
    st.title("Text Summarizer")
    # Copyright information
    st.markdown(
        """
        &copy; Under partial fulfilment of requirements of PROJCS701 by Abhirup Mazumder and Debjyoti Ghosh
        """
    )
    input_text = st.text_area("Input your article", height=200)
    progress_placeholder = st.empty()
    if st.button("Summarize"):
        if input_text:
            progress_bar = st.progress(0.0)
            progress_placeholder.text("Hold tight...We are generating some cute summaries for you! 😻")
            # Extractive Summarization
            extractive_summary = extractive_summarize(input_text, progress_bar)
            progress_placeholder.text("We are almost there...  🏁       🏃‍♂️")
            # Abstractive Summarization
            abstractive_summary = abstractive_summarize(input_text, progress_bar)

            # Display summaries
            st.subheader("Extractive Summary:")
            st.write(extractive_summary)

            st.subheader("Abstractive Summary:")
            st.write(abstractive_summary)
            progress_bar.empty()
            progress_placeholder.empty()
        else:
            st.warning("Please input your article.")
    

# Function to perform extractive summarization
def extractive_summarize(text, progress_bar):
    return extractiveSumm(text, progress_bar)

# Function to perform abstractive summarization
def abstractive_summarize(text, progress_bar):
    return abstractiveSumm(text, progress_bar)

if __name__ == "__main__":
    main()


#**************************************************************************************************



    # newtext=text.replace(" ","%20")
    # print(newtext)
    # newtext=newtext.replace("‘","%27")
    # newtext=newtext.replace("'","%27")
    # newtext=newtext.replace("&","%26")
    # # newtext=newtext.replace("?","%3F")
    # newtext=newtext.replace("–","%2D")

    # newtext=newtext.replace("’","%27")
    # print(newtext)

    # newtext=newtext.replace("(","%28")
    # newtext=newtext.replace(")","%29")
    # newtext=newtext.replace("-","%2D")
    # # newtext=text.replace(" ","%20")
    # url = f"http://127.0.0.1:5000/abstractive/{newtext}"
    # response = urlopen(url) 
    # data_json = json.loads(response.read()) 
    # print(data_json) 

    # return data_json["response"]