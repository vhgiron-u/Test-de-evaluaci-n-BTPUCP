import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

#se requiere instalado el paquete PyStemmer de Snowball
#para hallar las raices de las pal. en espanol:
import Stemmer
_stemmer = Stemmer.Stemmer('english')    #variable global para la funcion tokenize
#experimentalmente tiene mejor Accuracy


##en caso se use nltk:
#from nltk.stem.porter import *
#_stemmer = PorterStemmer()

from sklearn.model_selection import train_test_split


def quitar_puntuacion(text):
    return re.sub("[^\w\d'\s]+",'',text)
 

def simplif_tab(text):
    return re.sub("[\t]+"," ",text)

def stem_tokens(tokens, stemmer):
    stemmed = _stemmer.stemWords(tokens)
    #stemmed = [_stemmer.stem(token) for token in tokens] #para nltk
    return stemmed

def tokenize(text):
    tokens = text.split()
    stems = stem_tokens(tokens, _stemmer)
    return stems


def preprocesamiento(X_train):
    #list_msgs = [msg.decode('iso8859-15') for msg in X_train]
    #list_msgs = [msg.encode('UTF-8') for msg in list_msgs]
    token_dict = {} 

    for (i,msg) in enumerate(X_train):
        msg = msg.lower()
        msg = simplif_tab(msg) 
        msg_sin_punt = quitar_puntuacion(msg)
        
        token_dict[i] = msg_sin_punt
    return token_dict

def accuracy(y_pred, y_test):
    correct = 0
    total = len(y_test)
    for i, c in enumerate(y_test):
        if(y_pred[i] == c):
            correct += 1
    return correct/float(total)*100

def decod_veredicto(ch_veredicto):
    if(ch_veredicto == 'E'):
        s = "Electr."
    elif(ch_veredicto == 'M'):
        s = "Med."
    else:
        s = ""
    return s

def imprAcc(y_pred, y_test):
    print "Accuracy del clasificador Bayesiano: %.2f%%" %accuracy(y_pred, y_test)
    return


def reporteAcc(token_dict, y_pred, y_test, tope = 10):
    imprAcc(y_pred, y_test)
    #print "Accuracy del clasificador Bayesiano: %.2f%%" %accuracy(y_pred, y_test)
    #print
    print "\tMENSAJE\t\t\t\t\t\tPREDICCION\t\tTEMA REAL"
    for i, t in enumerate(y_test.index):
        if(i == tope):
            break
        print "\t", token_dict[i][:30]+"...", "\t\t\t", decod_veredicto(y_pred[i]), \
              "\t\t", decod_veredicto(y_test[t])
def reporte2(token_dict, y_pred, tope = 10):
    print "\MENSAJE\t\t\t\t\t\tPREDICCION"
    for i, t in enumerate(y_test.index):
        if(i == tope):
            break
        print "\t", token_dict[i][:30]+"...", "\t\t\t", decod_veredicto(y_pred[i])

def reporteCsvAcc(nombArch, X_raw, y_pred, y_test):
    import csv
    imprAcc(y_pred, y_test)
    with open(nombArch, 'wb') as csvfile:
        fwriter = csv.writer(csvfile, delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
        fwriter.writerow(['Message','Prediction','Topic'])
        for i, t in enumerate(y_test.index):
            fwriter.writerow([X_raw[t],y_pred[i],y_test[t]])    

def reporteCsvNews(nombArch, X_raw, y_pred):
    with open(nombArch, 'wb') as csvfile:
        fwriter = csv.writer(csvfile, delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
        fwriter.writerow(['Message','Prediction'])
        for i, t in enumerate(y_test.index):
            fwriter.writerow([X_raw[t],y_pred[i]])    

#Inicio del programa:


#leemos el archivo de entrenamiento (y prueba)
msgs = pd.read_csv('dataset.csv', sep=',', index_col=False).fillna("")


#seleccionamos los datos objetivo:

yy = msgs["Topic"]
XX = msgs["Message"]


#separamos datos de entrenamiento




test_prop = 0.2
X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size = test_prop,    \
                                                    random_state = 23)
###Entrenamiento:
#preprocesamiento:
token_dict = preprocesamiento(X_train)

#transformacion: (usando el tfidf de sklearn con el clasif. Multinomial Naive Bayes)
tfidf = TfidfVectorizer(tokenizer = tokenize)
tfs = tfidf.fit_transform(token_dict.values())

classifier = MultinomialNB(alpha=1.0)  #clasificador bayesiano
classifier.fit(tfs,y_train)

###Evaluacion del clasificador con la data de prueba (usando accuracy):
token_test = preprocesamiento(X_test)
tfs_pred = tfidf.transform(X_test)
y_pred = classifier.predict(tfs_pred)
#reporteAcc(token_test, y_pred, y_test, tope=10)

reporteCsvAcc("rep.csv",X_test, y_pred, y_test)

