from os import listdir,path
from collections import Counter
from collections import defaultdict
from svmutil import *
from nltk import word_tokenize,sent_tokenize,pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from math import log
from multiprocessing import Process,Queue

__author__ = 'Sajal/Harshal'

# TODO : refactor name of file


''' Sajal : Including function load_data

Description
    Loads the input data for the project as a list of input training samples, their labels and input test samples.

Input
    None

Output
    3 list. One for each : Train X, Train Y, Test Y.
'''
def load_data():
    input_dir = 'data/'
    train_file = 'project_articles_train'
    test_file = 'project_articles_test'

    train_X=[]
    train_Y=[]
    test_X=[]

    with open(input_dir+train_file,'r') as fp_train:
        for line in fp_train:
            data = line.strip().rsplit('\t',1)
            train_X.append(data[0])
            train_Y.append(int(data[1]))

    with open(input_dir+test_file,'r') as fp_test:
        for line in fp_test:
            data = line.strip()
            test_X.append(data)

    return train_X,train_Y,test_X


def get_words_from_corpus(train_X,test_X):
    word_list=[]
    for article in train_X:
        sentences =  sent_tokenize(article.decode('utf-8'))
        for sent in sentences:
            sent=sent.encode('utf-8')
            sent_tokens = word_tokenize(sent.decode('utf-8'))
            for token in sent_tokens:
                token=token.encode('utf-8')
                word_list.append(token)

    for article in test_X:
        sentences =  sent_tokenize(article.decode('utf-8'))
        for sent in sentences:
            sent=sent.encode('utf-8')
            sent_tokens = word_tokenize(sent.decode('utf-8'))
            for token in sent_tokens:
                token=token.encode('utf-8')
                word_list.append(token)

    word_dict=Counter(word_list)
    return word_dict


def convert_to_word_freq(word_dict,X_train,X_test):
    word_ind={}

    word_list = list(word_dict.keys())

    ind=0
    for word in word_list:
        word_ind[word]=ind
        ind+=1

    word_freq_train=[]
    for article in X_train:
        new_list=[]
        sentences =  sent_tokenize(article.decode('utf-8'))
        for sent in sentences:
            sent=sent.encode('utf-8')
            sent_tokens = word_tokenize(sent.decode('utf-8'))
            for token in sent_tokens:
                token=token.encode('utf-8')
                val=word_ind[token]
                new_list.append(val)
        word_freq_train.append(new_list)

    word_freq_test=[]
    for article in X_test:
        new_list=[]
        sentences =  sent_tokenize(article.decode('utf-8'))
        for sent in sentences:
            sent=sent.encode('utf-8')
            sent_tokens = word_tokenize(sent.decode('utf-8'))
            for token in sent_tokens:
                token=token.encode('utf-8')
                val=word_ind[token]
                new_list.append(val)
        word_freq_test.append(new_list)

    return word_freq_train,word_freq_test,word_ind


def normalize_data(word_freq_train,word_freq_test):
    ind=0
    for data in word_freq_train:
        new_data={}
        data=Counter(data)
        total=sum(data.values())
        for key,el in data.iteritems():
            new_data[key]=el/float(total)
        word_freq_train[ind]=new_data
        ind+=1

    ind=0
    for data in word_freq_test:
        new_data={}
        data=Counter(data)
        total=sum(data.values())
        for key,el in data.iteritems():
            new_data[key]=el/float(total)
        word_freq_test[ind]=new_data
        ind+=1

    return word_freq_train,word_freq_test



def join_features(A,B):
    return [i+j for i,j in zip(A,B)]

''' Harshal : Including Function get_all_files()

Description: 
    Takes input as a string of the path of a directory and recursively lists all files under the system. Never actually tested on folders containing folders. But works well if directories only consist of files. Works for both if / is present or absent at the end of the string

Input:
    String containing path of directory

Output:
    List of files in the directory containing absolute paths of files inside the directory
'''
def get_all_files(directory):
    if directory[-1] == "/":
        directory = directory[:-1]
    files = []
    for i in listdir(directory):
        if path.isdir(directory+"/"+i):
            files_under_dir = get_all_files(directory+"/"+i)
            for j in files_under_dir:
                j = i+"/"+j
                files+=[directory+"/"+j]
        else:        
            files+=[directory+"/"+i]
    return files



''' Harshal : Including function flatten

Description:
    Takes input as a list of lists and returs a list containing elements inside the inner lists

Input:
    List of Lists

Output:
    List
'''
def flatten(l):
    temp = []
    for i in l:
        temp+=[j for j in i]
    return temp


'''Harshal : Adding function create_vocab

Description:
    This function will read the data from the specified corpus and generate a vocabulary where each word occures atleast limit times and returns the topk words. The function should only be used for the TRAINING DATA. We assume that the last value is the label and hence drop it.

Input: 
    String containing directory of the files to be used for training, a value specifing the minimum number of occurences(including the value) for the word to be added in the vocabulary and the number of values to be returned.

Output:
    list of words ranked according to their frequency of occurance.
'''

def create_vocab(feature_list, limit, topk):
    
    counts_words = Counter(feature_list)
    vocab_words = [(word,counts_words[word]) for word in counts_words if counts_words[word] >= limit]
    vocab_words.sort(key = lambda x:(x[1],x[0]),reverse=True)
    vocabulary = [words for words,values in vocab_words]

    if topk < 0 or topk > len(vocabulary):
        return vocabulary
    return vocabulary[0:topk]


'''Harshal : Adding function generate_data_labels

Description:
    This function is only for data formatted as an paragraph and label seperated by a tab. We take the last character as the label. This is specifically for the problem at hand and the way it is presented

Input:
    String containing the path of the labeled data.

Output:
    List of tuples containing sentences and their corresponding labels
'''

def generate_data_labels(filepath):
    file_handle = open(filepath,"r")
    sentence_labels = [(line.strip()[:-1].rstrip(),line.strip()[-1]) for line in file_handle]
    file_handle.close()
    return sentence_labels

'''Harshal: Adding function KLEntropy

Decription: 

Computes the KL Entropy for two given probabilites

Input:

Two float values representing probabilities of some word in P and Q distribution respectively

Output:

Float value consisting of the entropy

'''

def KLEntropy(p , q):
    if p == 0:
        return 0.0
    else:
        return p*log(p/q)


'''Harshal : Adding function compute_distribution

Description:

Computes the probability distribution of words present in word_list. Counts the frequency of words in the list and divides by the total number of words in the list.

Input:

A list of words as strings

Output:

A default dictionary containing the mapping between words and their probabilities in the word_list

'''

def compute_distribution(word_list):
    count_dict = Counter(word_list)
    count_sum = len(word_list)
    if count_sum <= 0 :
        return defaultdict(int)
    prob_dict = defaultdict(int, { word : count_dict[word]*1.0/count_sum for word in count_dict})
    return prob_dict

'''Harshal : Adding function KLDivergence

Description:

Computes the KLDivergence between two distributions P and Q. Computes the KL Entropy between for each word in P distribution and sums them all up.

Input:

Two default dictionaries containing words and their probabilities for two distributions P and Q.


'''

def KLDivergence(P, Q):
    kvalue = 0.0
    klvalue = sum([KLEntropy(P[word],Q[word]) for word in P if word in Q])
    return klvalue

def L2_normalization(data):
    if type(data) == dict or type(data) == defaultdict or type(data) == Counter:
        norm_data = {}
        norm_val = 0.0
        for key in data:
            norm_val += data[key]**2
        norm_val = norm_val**0.5
        for key in data:
            norm_data[key] = data[key]/norm_val
        return norm_data
    
    elif type(data) == list:
        norm_val = sum([val**2 for val in data])**0.5
        norm_data = [val/norm_val for val in data]
        return norm_data

        
def preprocess_data(sentence_list,lowercase = True,lemmatize=True,stem=True):
    preprocessed_data = [sentence for sentence in sentence_list]
    if lowercase:
        preprocessed_data = [sentence.lower() for sentence in preprocessed_data]
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        lemmatizer_func = lemmatizer.lemmatize
        preprocessed_data = [flatten([[lemmatizer_func(w).encode('utf-8') for w in word_tokenize(s)]  for s in sent_tokenize(sent.decode('utf-8'))]) for sent in preprocessed_data]
        preprocessed_data = [' '.join(word_list) for word_list in preprocessed_data]
   
    return preprocessed_data

def generate_features(sentence_list,vocab_limit=1,num_features=1000,feature_type='words'):
    feature_list = []
    
    if feature_type == 'words':
        feature_list = [flatten([[word.encode('utf-8') for word in word_tokenize(s)] for s in sent_tokenize(sent.decode('utf-8'))]) for sent in sentence_list]

    elif feature_type == 'pos':
        print 'Starting Parsing...this may take a while'
        
        #feature_list = [flatten([[pos for word,pos in pos_tag(word_tokenize(s))] for s in sent_tokenize(sent.decode('utf-8'))]) for sent in sentence_list]
        google_lookup = get_google_pos('data/google_pos')
        for e in sentence_list:
            sent_list = sent_tokenize(e.decode('utf-8'))
            pos_list = []
            for s in sent_list:
                pos_list += [google_lookup[pos] for word,pos in pos_tag(word_tokenize(s))]
            print '.',
            feature_list += [pos_list]
        print 'Parsing completed'

    elif feature_type == 'func_words':
        file_list = get_all_files('data/Function_Words')
        func_word_list = []
        for f in file_list:
            f_handle = open(f,'r')
            func_word_list += [word.strip().lower() for word in f_handle if word[0]!='/']
            f_handle.close()
        feature_list = [flatten([[word.encode('utf-8') for word in word_tokenize(s) if word.encode('utf-8') in func_word_list] for s in sent_tokenize(sent.decode('utf-8'))]) for sent in sentence_list]
    vocab = create_vocab(flatten(feature_list),vocab_limit,num_features)
    vocab.sort()
    return vocab,feature_list

def pos_tagging_parallel(sentence_list,Output_Queue):
    
    Output_Queue.put(feature_list)


def mapper(inputlist, num_splits):
    temp_list = [i for i in inputlist]
    num_ele_per_split = len(temp_list)/num_splits
    segments = [[] for i in range(num_splits)]
    for index,element in enumerate(temp_list):
        segments[index%num_splits].append(element)
    segments = filter(None,segments)
    return segments


def get_google_pos(f):
    f_handle = open(f,"r")
    google_dict = defaultdict(lambda : "NOUN")
    for line in f_handle:
        key,value = line.strip().split()
        google_dict[key] = value
    return google_dict


def write_features_to_file(features,file_name):
    file_handle = open(file_name,'w',seperator)
    for i in features:
        a = seperator.join([str(k) for k in i])
        file_handle.write(a+"\n")
    file_handle.close()
