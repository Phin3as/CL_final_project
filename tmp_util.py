from os import listdir,path
from collections import Counter
from collections import defaultdict
#from svmutil import *
from nltk import word_tokenize,sent_tokenize
from math import log

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

def create_vocab(sentence_list, limit, topk):
    #file_list = get_all_files(inputdir)
    word_list = []
    #for f in file_list:
    #f_handle = open(inputfile,"r")
    word_list+=flatten([[w.encode('utf-8') for w in word_tokenize(s.decode('utf-8'))] for s in sentence_list])
    
    counts_words = Counter(word_list)
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
