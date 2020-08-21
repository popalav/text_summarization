# -*- coding: utf-8 -*-
"""
Created on Mon March  4 01:35:09 2020

@author: Lavi Popa
"""

import numpy as np
from matplotlib import pyplot
import pandas as pd 
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import matplotlib.pyplot as plt
from pyrouge import Rouge155
from pythonrouge.pythonrouge import Pythonrouge
from rouge.rouge import rouge_n_sentence_level
from pyrouge import Rouge155 
#To avoid warnings
import time
import warnings
warnings.filterwarnings('ignore')
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from collections import Counter
from keras import backend as K 
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.vis_utils import plot_model
from keras.optimizers import RMSprop, rmsprop, Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model 
from keras.models import load_model
from keras.models import model_from_json
import json
from keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Flatten, concatenate, Activation, Dense, RepeatVector, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score
from  plotly.offline import plot
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

amazon_dataset = pd.read_csv("Reviews1.csv",nrows=10000)
amazon_dataset.drop_duplicates(subset=['Text'],inplace=True)#dropping duplicates
amazon_dataset.dropna(axis=0,inplace=True)#dropping na
amazon_dataset.drop(["Score","UserId", "ProductId","UserId","ProfileName","HelpfulnessNumerator","HelpfulnessDenominator","Time"],axis=1, inplace=True)
amazon_dataset.info()
contractions_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}


def __lower_text(text_data):
      """Convert text to lower case"""
      text_data = text_data.lower()
      return text_data

def __remove_HTML(text_data):
      """ Removes HTML tags """
      text_data = BeautifulSoup(text_data,"lxml").text
      return text_data

def __remove_text_paranthesis(text_data):
      """Removes the text inside paranthesis () and []"""
      text_data = re.sub(r'\([^)]*\)','', text_data)
      return text_data

def __removes_s(text_data):
      """Removes 's"""
      text_data = re.sub(r"'s\b","", text_data)
      return text_data

def __remove_punctuation(text_data):
      """Remove punctuation and special characters """
      text_data = re.sub("[^a-zA-Z]"," ", text_data)
      return text_data
def __expand_contactions(text_data):
      """Expand contractions like he's to he is """
      text_data =' '.join([contractions_dict[i] if i in contractions_dict else i for i in text_data.split(" ")])
      return text_data

def __remove_short_words(text_data):
      """Removes short words with lengh <=2 """
      text_data = ' '.join(word for word in text_data.split() if len(word)>3)
      return text_data

def __remove_stopword(text_data):
      """Remove stopwords """
      stop_words = set(stopwords.words('english'))
      text_data = [word for word in text_data.split() if word not in stop_words]
      return text_data

stopWords = set(stopwords.words('english'))
def __one(text_data):
    
    text_data = __lower_text(text_data)
    text_data = __remove_HTML(text_data)
    text_data = __remove_text_paranthesis(text_data)
    text_data = re.sub('"'," ", text_data)
    text_data = __expand_contactions(text_data)
    text_data = __removes_s(text_data)
    text_data = __remove_punctuation(text_data)
    text_data = ' '.join(word for word in text_data.split() if len(word)>1)
    text_tokens = __remove_stopword(text_data)
    text_final = []
    for i in text_tokens:
        if len(i)>1:
          text_final.append(i)
    return (" ".join(text_final)).strip()

def __clean_field(amazon_dataset, option): 
    """Cleanes the Text field from the datset """
    clean = []
    if option == 1:
        for i in amazon_dataset['Text']:
            clean.append(__one(i))
    elif option == 2:
        for i in amazon_dataset['Summary']:
            clean.append(__one(i))
    else:
        pass
    return clean

cleanText = __clean_field(amazon_dataset,1)
cleanSummary = __clean_field(amazon_dataset,2)

# Creates two new columns in dataset with the cleaned text and summary 
amazon_dataset['cleanText'] = cleanText
amazon_dataset['cleanSummary'] = cleanSummary 
amazon_dataset.info()

# Replace empty values with Na and drop na, for rows 
amazon_dataset.replace('', np.nan, inplace=True)
amazon_dataset.dropna(axis=0,inplace=True)

def __clean_text_tokens(text_data):
    """Tokenieze the clean text eg cleanSummary """
    clean_tokens = []
    for j in text_data:
        for k in word_tokenize(j):
            clean_tokens.append(k)
    return clean_tokens

def __create_vocabulary(text_data):
    """Creates the vocab of words occurences; takes as input a list of tokeneized words"""
    vocabulary = Counter()
    vocabulary.update(text_data)
    return vocabulary

def __full_vocab(a,b):
    """Creates full vocab containg both 'Text' and "Summary" fields """
    a = __clean_text_tokens(a)
    b = __clean_text_tokens(b)
    c = a + b
    d = __create_vocabulary(c)
    # __full_vocab(amazon_dataset['cleanSummary'],amazon_dataset['cleanText']) -apel
    return d

def __tokens_min_occurence(option,min):
    """Keeps the tokens with a minimum occurence """
    voc = []
    tik_tok = []
    if option == 1:
        voc = __create_vocabulary(__clean_text_tokens(cleanText))
    elif option == 2:
        voc = __create_vocabulary(__clean_text_tokens(cleanSummary))
    else:
        pass
    tik_tok = [key for key,value in voc.items() if value >= min]
    return len(tik_tok)

def __find_max_lenght(option):
    """Finds the maximum length for the already cleaned text  """
    maximum = 0
    if option == 1:
        for i in amazon_dataset['cleanText']:
            if len(i.split()) > maximum:
                maximum = len(i.split())
    elif option == 2:
        for i in amazon_dataset['cleanSummary']:
            if len(i.split()) > maximum:
                maximum = len(i.split())
    else:
        pass
    return maximum

def __plot_the_tex_lenght():
   """Plot the text lenght before removing the stop words """
   text_lenght = []
   for i in amazon_dataset['Text']:
       text_lenght.append(len(i.split()))
   df_texts_lenght = pd.DataFrame({'text_len':text_lenght})
   df_texts_lenght['text_len'].plot(
    kind='hist',
    bins=100,
    x="text length",
    colormap='Purples_r',
    y="count",
    title='Text Length Distribution')
   plt.xlabel('text length')
   plt.ylabel('count')
   plt.savefig('text_lenght1.pdf')
   plt.savefig('text_lenght1.png')
   plt.show()
   
def __plot_the_summ_lenght():
   """Plot the summary lenght before removing the stop words """
   summ_lenght = []
   for i in amazon_dataset['Summary']:
       summ_lenght.append(len(i.split()))
   df_summ_lenght = pd.DataFrame({'summ_len':summ_lenght})
   df_summ_lenght['summ_len'].plot(
    kind='hist',
    bins=100,
    x="summary length",
    colormap='Purples_r',
    y="count",
    title='Summary Length Distribution')
   plt.xlabel('summary length')
   plt.ylabel('count')
   plt.savefig('summary_lenght.pdf')
   plt.savefig('summary_lenght1.png')
   plt.show()

def __top_words(corpus, n=None):
    """For the next function"""
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def __common_words_plot():
    """Plot the most common 20 words from the summary vocabulary before removing stop words """
    #summ_vocab = __create_vocabulary(__clean_text_tokens(amazon_dataset['cleanSummary']))
    common_words = __top_words(amazon_dataset['cleanSummary'], 20)
    for w, f in common_words:
        print(w, f)
    df1 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
    df1.groupby('ReviewText').sum()['count'].sort_values(ascending=False).plot(
                                                                              kind='bar',
                                                                               y='Count',
                                                                    colormap='Purples_r',
                               title='Top 20 words in review summary after removing stop words')
    plt.savefig('summary_commmon_words_clean.png')                           
    plt.show()
#__plot_the_tex_lenght()
#__plot_the_summ_lenght()
#__common_words_plot()

X = amazon_dataset['cleanText']
Y = amazon_dataset['cleanSummary']
XTRAIN, XTEST, YTRAIN, YTEST = train_test_split(X,Y,test_size=0.2,random_state=None, shuffle=False)
#### PT AFISARE CURATA NUMA ####
X1 = amazon_dataset['Text']
Y1 = amazon_dataset['Summary']
XTRAIN1, XTEST1, YTRAIN1, YTEST1 = train_test_split(X1,Y1,test_size=0.2,random_state=None,shuffle=False)

LONGEST_TEXT = 70#__find_max_lenght(1) 
LONGEST_SUMMARY =7# __find_max_lenght(2)
TEXT_RARE_WORDS = __tokens_min_occurence(1,6)
SUMMARY_RARE_WORDS = __tokens_min_occurence(2,2)
TEXT_VOCABULARY = len(__create_vocabulary(__clean_text_tokens(cleanText)))
SUMMARY_VOCABULARY = len(__create_vocabulary(__clean_text_tokens(cleanSummary)))
print(TEXT_RARE_WORDS)
print(TEXT_VOCABULARY)

#=====================================================================
"""Creates a tokenizer; takea as input XTRAIN or smth similar """
nr = TEXT_VOCABULARY - TEXT_RARE_WORDS
nr1 = SUMMARY_VOCABULARY - SUMMARY_RARE_WORDS
    
took = Tokenizer(nr)
took.fit_on_texts(list(XTRAIN))
encoder_train = took.texts_to_sequences(XTRAIN)
encoder_test = took.texts_to_sequences(XTEST)
XTRAIN = pad_sequences(encoder_train,LONGEST_TEXT,padding='post')
XTEST = pad_sequences(encoder_test,LONGEST_TEXT,padding='post')
x_vocab = took.num_words +1
print("size voc", took.num_words)

took1 = Tokenizer(nr1)
took1.fit_on_texts(list(YTRAIN))
encoder_train = took1.texts_to_sequences(YTRAIN)
encoder_test = took1.texts_to_sequences(YTEST)
YTRAIN = pad_sequences(encoder_train,LONGEST_SUMMARY,padding='post')
YTEST = pad_sequences(encoder_test,LONGEST_SUMMARY,padding='post')
y_vocab = took1.num_words +1

#====================================================================
from keras.layers import Layer
class Round(Layer):
    def __init__(self, **kwargs):
        super(Round, self).__init__(**kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        return K.round(X)

    def get_config(self):
        config = {"name": self.__class__.__name__}
        base_config = super(Round, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#=====================================================================


def za_model():
    K.clear_session()
    #HYPERPARAMETERS
    vocab_size = y_vocab 
    vocab_size_text = x_vocab
    cells = 128
    learning_rate = 0.002
    clip_norm = 0.5
   
    ######### ZA MODEL 1 ##################################
    inputs1 = Input(shape=(LONGEST_TEXT,))
    text1 = Embedding(vocab_size_text, cells)(inputs1)
    text2 = LSTM(cells , dropout=0.2, recurrent_dropout=0.2)(text1)
    text3 = RepeatVector(cells)(text2)
    # summary input model
    inputs2 = Input(shape=(LONGEST_SUMMARY,))
    summary1 = Embedding(vocab_size, cells)(inputs2)
    summary2 = LSTM(cells, dropout=0.2, recurrent_dropout=0.2)(summary1) 
    summary3 = RepeatVector(cells)(summary2)
    # decoder model
    decoder1 = concatenate([text3, summary3])
    decoder2 = LSTM(cells, return_sequences=True,return_state=True)(decoder1)
    decoder3 = LSTM(cells)(decoder2)
    outputs = Dense(LONGEST_SUMMARY, activation='softmax')(decoder3)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    Round()
    rmsprop = RMSprop(lr=learning_rate,clipnorm=clip_norm)
    opt = Adam(learning_rate=0.008)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model



def za_model2():
    K.clear_session()
    #HYPERPARAMETERS
    cells = 128
    vocab_size = y_vocab
    hidden_units = 128
    learning_rate = 0.002
    clip_norm = 0.5
    # text input and encoder
    inputs1 = Input(shape=(LONGEST_TEXT,))
    text1 = Embedding(x_vocab, cells)(inputs1)
    text2 = Bidirectional(LSTM(cells,dropout_U = 0.2, dropout_W = 0.2,return_state=True), merge_mode = 'concat')
    _, forward_h, forward_c, backward_h, backward_c = text2(text1)
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states_concat = [state_h, state_c]
    # summary input and decoder
    inputs2 = Input(shape=(LONGEST_SUMMARY,))
    summary1 = Embedding(vocab_size, cells)
    emb_summ = summary1(inputs2)
    summary2 = LSTM(2*cells, return_state=True, return_sequences=False)
    decoder_output, decoder_h, decoder_c = summary2(emb_summ,initial_state = encoder_states_concat)
    outputs = Dense(LONGEST_SUMMARY, activation='softmax')
    outputs2 = outputs(decoder_output) 
    model = Model(inputs=[inputs1, inputs2], outputs=outputs2)
     
    opt = Adam(learning_rate=0.0008)
    rmsprop = RMSprop(lr=0.0001, rho=0.9,  decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop,metrics=['accuracy'])
    return model
#=====================================================================

start = time.time()



model = za_model()
"""
#model = load_model('model_final_model1.h5')
# load json and create model
json_file = open('model_final_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_final_model1.h5")
print("Loaded model from disk")
opt = Adam(learning_rate=0.0008)
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
"""
#plot_model(model, to_file='model_repeat.png', show_shapes=True)

print(model.summary())
early_stop = EarlyStopping(monitor='loss', mode='min', verbose=1,patience=2)

y_tr = YTRAIN#.reshape(YTRAIN.shape[0], 1, YTRAIN.shape[1])[:,1:]
y_val = YTEST#.reshape(YTEST.shape[0], 1, YTEST.shape[1])
x_val = XTEST#.reshape(XTEST.shape[0], 1, XTEST.shape[1])
x_tr = XTRAIN#.reshape(XTRAIN.shape[0], 1, XTRAIN.shape[1])[:,1:]

history= model.fit(x=[x_tr,y_tr],
              y=y_tr,
              batch_size=25,
              epochs=20,
              verbose=1,
              shuffle=False,
              #callbacks=[early_stop],
              validation_data=([x_val,y_val], y_val))

stop = time.time()

def plot_training(history):
    print(history.history.keys())
    #  "Accuracy"
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('bidirect_acc.png')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('bidirect_loss.png')

    plt.show()
plot_training(history)

def evaluates_the_model():
    """Evaluates the model, regarding both test data and training data, the right way ;) """
    scores = model.evaluate([XTEST, YTEST], YTEST, verbose=0)
    print('LSTM test scores:', scores)
    print('\007')

def save_the_model():
    """Saves the model to a json and he weights to a h5 file """
    # serialize model to JSON
    model_json = model.to_json()
    with open("model_final_model1.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_final_model1.h5")
    print("Saved model to disk")

# Tokens and dicts for reversing purpose

reverse_word_map = dict(map(reversed, took.word_index.items()))
index_to_word_tok=took1.index_word
index_to_word_tok2=took.index_word
word_to_index_tok=took1.word_index

def translate_text(input_array):
    """Receives an array coresponding to text and transforms it to words """
    decoded_sequence = ''
    for i in input_array[0]:
        if i != 0:
            decoded_sequence = decoded_sequence + index_to_word_tok2[i] + ' '
    return decoded_sequence

def translate_summary(input_array):
    """Receives an array coresponding to text and transforms it to words """
    decoded_summary = ''
    for i in input_array[0]:
        if i != 0:
            decoded_summary = decoded_summary + index_to_word_tok[i] + ' '
    return decoded_summary

print(f"Training time: {stop - start}s")
def predict_the_future():
    """Makes prediction for the full dataset, plese modify it to make predictions only for 100 inputs """
    
 
    list_rouge_r = []
    list_rouge_p = []
    list_rouge_l = []
    for i,j,k,l in zip(XTRAIN,YTRAIN,XTRAIN1,YTRAIN1):
        iris1 = ''
        iris = ''
        #print("Original Text Preproccesed:",translate_text(i.reshape(1,LONGEST_TEXT)))
        #org_sum = translate_summary(j.reshape(1,LONGEST_SUMMARY))
        print("Original Text:",__remove_HTML(k))
        print("Human Summary:",l)
        p =  model.predict([i.reshape(1,LONGEST_TEXT),j.reshape(1,LONGEST_SUMMARY)],verbose=0,batch_size=25)
        j_auz = j.reshape(1,LONGEST_SUMMARY)
        maxx = max(j_auz[0])
        pred = np.int32(p*maxx)
        for mata in pred[0]:
            if mata != 0:
                iris1 =  reverse_word_map[mata]
                iris =  ' ' + iris + ' ' + iris1
            elif mata == 0:
                pass
            else:
                pass
        
        print("Generated Summary", iris)
        if iris != '' and l != '':

            recall, precision, rouge = rouge_n_sentence_level(iris, l, 2)
            list_rouge_r.append(recall)
            list_rouge_p.append(precision)
            list_rouge_l.append(rouge)
            print('ROUGE-2-R', recall)
            print('ROUGE-2-P', precision)
            print('ROUGE-2-F', rouge)
pyplot.plot(history.history['loss'], label='train') 
pyplot.plot(history.history['val_loss'], label='test') 
pyplot.legend()
plt.savefig('ceva.png')
pyplot.show()

def main():
    #evaluates_the_model()
    predict_the_future()
    save_the_model()

if __name__ == "__main__":
    main()