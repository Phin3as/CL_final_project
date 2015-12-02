from os import listdir,path
from collections import Counter
from collections import defaultdict
from svmutil import *
from nltk import word_tokenize,sent_tokenize

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

''' Harshal : Adding function generate_svm_files()

Description:
    This function will read the data as a list of (sentence,label) pairs and generate a file formated in the form required by libsvm ie (label features in the sentence). The output will be written in a file

Input:
    String containing the path to the directory containing the data, 
'''

def generate_svm_files(sentence_label_list, vocab, output_file):
    label_list = [value for sent,value in sentence_label_list]
    sent_list = [sent for sent,value in sentence_label_list]

    vocab_feature_space = {word : str(i+1) for i,word in enumerate(vocab)}
    
    unique_labels = list(set(label_list))
    unique_labels.sort()

    label_feature_space = {label : str(i+1) for i,label in enumerate(unique_labels)}

    output_file_handle = open(output_file, "w")
    
    for index,sent in enumerate(sent_list):
        words_list = flatten([[word.encode('utf-8') for word in word_tokenize(s)] for s in sent_tokenize(sent.decode('utf-8'))])
        counts = Counter(words_list)
        
        feature_list = [label_feature_space[label_list[index]]]
        for w in counts:
            if w in vocab_feature_space.keys():
                feature_list+=[vocab_feature_space[w]+":"+str(counts[w])]
        output_file_handle.write(' '.join(feature_list)+"\n")

    output_file_handle.close()



'''Harshal : Adding function train_test_model

Description:
   This function call the svm function in libsvm and outputs the classification accuracy

Input:
   Strings containing the path to the train datafile and test data file

Output:
  predicted labels, predicted accuracy and predicted values
 
'''

def train_test_model(train_datafile, test_datafile):
    y_train, x_train = svm_read_problem(train_datafile)
    problem = svm_problem(y_train, x_train)
    param = svm_parameter('-t 0 -e .01 -m 1000 -h 0')
    m = svm_train(problem,param)
    y_test, x_test = svm_read_problem(test_datafile)
    p_labels, p_acc, p_vals = svm_predict(y_test, x_test, m)
    return p_labels, p_acc, p_vals
