__author__ = 'Sajal'

from tmp_util import *
from math import sqrt

def all_zero(X_train,Y_train,X_test):
    return [0]*len(X_test)

def all_one(X_train,Y_train,X_test):
    return [1]*len(X_test)

'''Harshal : Adding function svm_driver to perform svm and be used by cross validation

Description:

 This function will call the generate_svm_files method to generate the svm formatted training and test files and subsequently the train_test method to perform svm on the data

Input : 

a list of training sentences X_train, their corresponding labels, Y_train and sentences on which the prediction is mad
e X_test.

Output:

a list of predictions obtained by running svm on test data X_test

'''

def svm_driver(X_train, Y_train, X_test):
    
    vocab = create_vocab(X_train,1,1000)
    sentence_label_list = zip(X_train,Y_train)
    outputfile_path = 'DataGen/models_train_svm'
    generate_svm_files(sentence_label_list,vocab,outputfile_path)
    
    predicted_labels = train_test_model(outputfile_path,X_test)
    
    return predicted_labels


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
   Strings containing the path to the train datafile and list of test data whose labels are to be predicted. NOTE THAT DUE TO THE STRANGE DEFINITION OF svm_predict in libsvm, we have to pass a y_test to it.

Output:
  predicted labels
 
'''

def train_test_model(train_datafile, X_test):
    y_train, x_train = svm_read_problem(train_datafile)
    problem = svm_problem(y_train, x_train)
    param = svm_parameter('-t 0 -e .01 -m 1000 -h 0')
    m = svm_train(problem,param)
    #y_test, x_test = svm_read_problem(test_datafile)
    Y_test = [1]*len(X_test);
    p_labels, p_acc, p_vals = svm_predict(y_test,X_test, m)
    return p_labels

'''Harshal : Adding KL_Classifier

Description:

This function is a simple method which simply checks if the unlabled paragraph is similar to the authors excerpt or not. The similarity is measured through KL Divergence between sentences.

Input:

A list of sentences, a list containing labels for the sentence in the previous list and Test senetences

Output:

A list of predicted labels
'''

def KL_Classifier(train_X, train_Y, test_X):
    
    stop_words_file = open('data/stopwords.txt','r')
    stop_words_list = [line.strip() for line in stop_words_file ]
    stop_words_file.close()

    author_excerpts = [sentence for sentence,label in zip(train_X,train_Y) if label == 1]
    other_excerpts = [sentence for sentence,label in zip(train_X,train_Y) if label == 0]

    author_excerpts_words = flatten([[[word.encode('utf-8') for word in word_tokenize(sentence) if word not in stop_words_list] for sentence in sent_tokenize(excerpt.decode('utf-8'))] for excerpt in author_excerpts])
                                        
    other_excerpts_words = flatten([[[word.encode('utf-8') for word in word_tokenize(sentence) if word not in stop_words_list] for sentence in sent_tokenize(excerpt.decode('utf-8'))] for excerpt in other_excerpts])
    

    author_excerpts_word_dist = compute_distribution(flatten(author_excerpts_words))
    others_excerpts_word_dist = compute_distribution(flatten(other_excerpts_words))
                                   
    test_Y = []

    for sentence in test_X:
        words = flatten([[word.encode('utf-8') for word in word_tokenize(s) if word not in stop_words_list] for s in sent_tokenize(sentence.decode('utf-8'))])
        word_dist = compute_distribution(words)
        
        kl_value_author = KLDivergence(word_dist,author_excerpts_word_dist)
        kl_value_other = KLDivergence(word_dist,others_excerpts_word_dist)

        if kl_value_author < kl_value_other:
            test_Y.append(0)
        else:
            test_Y.append(1)
    
    return test_Y


def model_cos_sim(train_X, train_Y, test_X):

    Y_test = []
    for data in test_X:
        max_sim=0
        max_idx=-1
        ind=0
        for data2 in train_X:
            sim=cosine_sim(data,data2)
            if sim > max_sim:
                max_sim=sim
                max_idx=ind
            ind+=1
        Y_test.append(train_Y[max_idx])

    return Y_test


def cosine_sim(l1, l2):
    # print 's'
    n=l1.__len__()
    m=l2.__len__()
    if (n!=m):
        return 0
    sum=0
    l1_den=0
    l2_den=0
    for key,val in l1.iteritems():
        if not l2.has_key(key):
            continue
        val2=l2[key]
        sum+=val*val2
        l1_den+=val*val
        l2_den+=val2*val2
        # print l1_den,l2_den

    if sqrt(l1_den)*sqrt(l2_den)==0:
        sim=0
    else:
        sim=sum/float(sqrt(l1_den)*sqrt(l2_den))
    return sim
