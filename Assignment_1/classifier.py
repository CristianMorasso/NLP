import collections
import math
import random
import nltk 
import numpy as np
nltk.download('punkt')
nltk.download("europarl_raw")

from tqdm import tqdm
#nltk.download("stopwords")
from nltk.corpus import europarl_raw
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
from nltk.stem.lancaster import LancasterStemmer

from nltk.stem import WordNetLemmatizer

from nltk.metrics import ConfusionMatrix
from nltk.metrics.scores import (precision, recall)

# from nltk.tokenize import RegexpTokenizer, word_tokenize

# >>> from nltk.tokenize import word_tokenize
# >>> from nltk.probability import FreqDist
# >>> sent = 'This is an example sentence'
# >>> fdist = FreqDist()
# >>> for word in word_tokenize(sent):
# ...    fdist[word.lower()] += 1

#oggetti per BOW
st = PorterStemmer() 
#st = LancasterStemmer()
wnl = WordNetLemmatizer()
def bow(data, stopwords = 10, limit =2000):
    """
        riceve in input una lista di testi e ne crea una BOW levando le prime 10 parole piu frequenti\n
        IN:\n
        text:   list of documents\n
        stopwords:  number of words to ignore\n
        OUT:\n
        BOW:    list of topwords ordered by most frequent
    """
    #distribuzione parole piu frequenti
    dataProcessed = [0 for _ in range(len(data))]
    fdist = FreqDist()
    parole = []
    for i, (doc, l)in enumerate(tqdm(data)):
        temp = ([], l)
        
        #TOKENIZATION 
        #tokenization doc into sentences
        senToc = sent_tokenize(doc)
        for sent in senToc:
            sent = sent.lower()
            #tokenizer = RegexpTokenizer(r"[A-zÀ-ú ]+")  # r"[A-zÀ-ú ]+": rimuove numeri, caratteri speciali e tiene solo le lettere e spazi, anche accentate! 
            #tokenization into words 
            words = word_tokenize(sent) 
            
            for word in words:
                parole.append(word)
                #Stopwords
                # parole = list(parole)[stopwords:]#prendo tutte le parole tranne le prime 10

                #Stemming (valutare i vari stemmer)
                stemmed= st.stem(word)

                #lemmatization
                lemmatized= wnl.lemmatize(stemmed) 

                #counting words elaborated    
                fdist[lemmatized] += 1
                temp[0].append(lemmatized) 

        dataProcessed[i]= temp

    return list(fdist)[stopwords:limit], dataProcessed

def features_estractor1(d, tW):
    ds = set(d)
    features = {}
    for w in tW:
        features[f'contains({w})'] = (w in ds)
    return features


fids = 10
nLen = 3
h_ids = math.floor(fids/(nLen-1))

data = []
labels = []
tests = []
#prendo fids documenti per ogni lingua
en = europarl_raw.english.fileids()[:fids]
fr = europarl_raw.french.fileids()[:h_ids] 
dan = europarl_raw.danish.fileids()[:h_ids]



#creo una lista di tuple con doc e lingua
#E english N_E not english
for i in range(fids):
    data.append((europarl_raw.english.raw(en[i]), "E"))
for i in range(h_ids):
    data.append((europarl_raw.french.raw(fr[i]), "N_E"))
    data.append((europarl_raw.danish.raw(dan[i]), "N_E"))

#mischio i dati
random.shuffle(data)

#creo una lista con i vari documenti mischiati senza label
# text = []
# for i in data:
#     text.append(i[0])

#creo BOW per Naive Bayes
bowObj, dataProcessed = bow(data, stopwords=0, limit = 5000)

#feature set
# #uso Bow con 0 stopword perche devo mantenerle tutte
# pre_features = bow(text, 0)
# set_pre_features = set(pre_features)

# features = {}
# for word in bowObj:
#     features[f'contains({word})'] = (word in pre_features)

#divisione train e test sets
#featuresets = [(features, c) for (d,c) in enumerate(data)]
featuresets = [(features_estractor1(d,bowObj),l) for (d,l) in tqdm(dataProcessed)]
sep = math.floor(len(featuresets) * 0.5 )
train_set, test_set = featuresets[:sep], featuresets[sep:]

#classifier
classifier = nltk.NaiveBayesClassifier.train(train_set) 

print("Testing and Metrics: ")
refsets =  collections.defaultdict(set)
testsets = collections.defaultdict(set)

for i,(feats,label) in enumerate(test_set):
    refsets[label].add(i)
    result = classifier.classify(feats)
    testsets[result].add(i)
    labels.append(label)
    tests.append(result)
cm = ConfusionMatrix(labels, tests)

print("Accuracy:",nltk.classify.accuracy(classifier, test_set))
print( 'Precision:', precision(refsets['E'], testsets['E']) )
print( 'Recall:', recall(refsets['E'], testsets['E']) )
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))
classifier.show_most_informative_features(10)
