__author__ = 'Sajal'

from tmp_util import *
from math import sqrt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression,Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.lda import LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GMM
from sklearn.decomposition import SparsePCA

def all_zero(X_train,Y_train,X_test):
    return [0]*len(X_test)

def all_one(X_train,Y_train,X_test):
    return [1]*len(X_test)

def GMM_driver(X_train,Y_train,X_test):
    X_train_prep = X_train
    X_test_prep = X_test
    vocab_train,transformed_train_X = generate_features(X_train_prep,1,1000,'words')
    vocab_test, transformed_test_X = generate_features(X_test_prep,1,1000,'words')

    sentence_label_list_train = zip(transformed_train_X,Y_train)
    sentence_label_list_test = zip(transformed_test_X,[1]*len(X_test))

    features_train = format_features_sklearn(sentence_label_list_train,vocab_train,'tf-idf')
    features_test = format_features_sklearn(sentence_label_list_test,vocab_train,'tf-idf')
    
    model = GMM(n_components = 10)
    model.fit(features_train)
    labels_train = model.predict(features_train)
    decisions = {}
    index_author = [i for i in range(len(Y_train)) if Y_train[i]==1]
    index_other = [i for i in range(len(Y_train)) if Y_train[i] == 0]
    likely_labels = {i : [] for i in list(set(labels_train))}
    for i,e in enumerate(labels_train):
        likely_labels[e] += [Y_train[i]]
    for i in likely_labels:
        likely_labels[i] = 1 if sum(likely_labels[i]) > len(likely_labels[i])/2.0 else 0
    
    labels = model.predict(features_test)
    predicted_labels = []
    for i in labels:
        predicted_labels +=[likely_labels[i]]
    return predicted_labels
    #predicted_labels = model.predict(features_test)
    
    #return predicted_labels

def knn_driver(X_train,Y_train,X_test):
    X_train_prep = X_train
    X_test_prep = X_test
    vocab_train,transformed_train_X = generate_features(X_train_prep,1,1000,'words')
    vocab_test, transformed_test_X = generate_features(X_test_prep,1,1000,'words')

    sentence_label_list_train = zip(transformed_train_X,Y_train)
    sentence_label_list_test = zip(transformed_test_X,[1]*len(X_test))

    features_train = format_features_sklearn(sentence_label_list_train,vocab_train,'tf-idf')
    features_test = format_features_sklearn(sentence_label_list_test,vocab_train,'tf-idf')
    
    model = KNeighborsClassifier()
    model.fit(features_train,Y_train)
    
    predicted_labels = model.predict(features_test)
    
    return predicted_labels

def perceptron_driver(X_train,Y_train,X_test):
    X_train_prep = X_train
    X_test_prep = X_test
    vocab_train,transformed_train_X = generate_features(X_train_prep,1,1000,'words')
    vocab_test, transformed_test_X = generate_features(X_test_prep,1,1000,'words')

    sentence_label_list_train = zip(transformed_train_X,Y_train)
    sentence_label_list_test = zip(transformed_test_X,[1]*len(X_test))

    features_train = format_features_sklearn(sentence_label_list_train,vocab_train,'tf-idf')
    features_test = format_features_sklearn(sentence_label_list_test,vocab_train,'tf-idf')
    
    model = Perceptron()
    model.fit(features_train,Y_train)
    
    predicted_labels = model.predict(features_test)
    
    return predicted_labels

def LDA_driver(X_train,Y_train,X_test):
    
    X_train_prep = X_train
    X_test_prep = X_test
    vocab_train,transformed_train_X = generate_features(X_train_prep,1,1000,'words')
    vocab_test, transformed_test_X = generate_features(X_test_prep,1,1000,'words')

    sentence_label_list_train = zip(transformed_train_X,Y_train)
    sentence_label_list_test = zip(transformed_test_X,[1]*len(X_test))

    features_train = format_features_sklearn(sentence_label_list_train,vocab_train,'tf-idf')
    features_test = format_features_sklearn(sentence_label_list_test,vocab_train,'tf-idf')
    
    model = LDA()
    model.fit(features_train,Y_train)
    
    predicted_labels = model.predict(features_test)
    
    return predicted_labels

def NB_driver(X_train,Y_train,X_test):
    #X_train_prep = preprocess_data(X_train)
    #X_test_prep = preprocess_data(X_test)
    
    X_train_prep = X_train
    X_test_prep = X_test
    vocab_train,transformed_train_X = generate_features(X_train_prep,1,1000,'words')
    vocab_test, transformed_test_X = generate_features(X_test_prep,1,1000,'words')

    sentence_label_list_train = zip(transformed_train_X,Y_train)
    sentence_label_list_test = zip(transformed_test_X,[1]*len(X_test))

    features_train = format_features_sklearn(sentence_label_list_train,vocab_train)
    features_test = format_features_sklearn(sentence_label_list_test,vocab_train)
    
    model = GaussianNB()
    model.fit(features_train,Y_train)
    
    predicted_labels = model.predict(features_test)
    
    return predicted_labels

def AdaBoost_driver(X_train,Y_train,X_test):
    #X_train_prep = preprocess_data(X_train)
    #X_test_prep = preprocess_data(X_test)    
    X_train_prep = X_train
    X_test_prep = X_test
    vocab_train,transformed_train_X = generate_features(X_train_prep,1,1000,'words')
    vocab_test, transformed_test_X = generate_features(X_test_prep,1,1000,'words')
    
    sentence_label_list_train = zip(transformed_train_X,Y_train)
    sentence_label_list_test = zip(transformed_test_X,[1]*len(X_test))

    features_train = format_features_sklearn(sentence_label_list_train,vocab_train)
    features_test = format_features_sklearn(sentence_label_list_test,vocab_train)
    
    model = AdaBoostClassifier(n_estimators = 10000)
    model.fit(features_train,Y_train)
    
    predicted_labels = model.predict(features_test)
    
    return predicted_labels


class NGram:
    
    probabilities = None
    probabilities_denom = None
    vocab = None
    V = 0
    k = 0.25
    feature_type=None
    N = 0

    def probability(self,ngram):
        if self.probabilities[ngram[0:-1]] == 0:
            float('Inf')
        prob = (self.probabilities[ngram]+self.k)/(self.probabilities_denom[ngram[0:-1]]+self.k*self.V)
        return prob

    def log_prob(self,sentence):
        preprocessed_list = preprocess_data([sentence])
        vocab,transformed_data = generate_features(preprocessed_list,1,1000,self.feature_type)
        transformed_UNK = [[word if word in self.vocab else "<UNK>" for word in w] for w in transformed_data]
        transformed_data = [['<s>']+w for w in transformed_UNK]
        pattern_list = []
        for i in transformed_data:
            for j in range(len(i)-self.N+1):
                pattern_list.append(tuple(i[j:j+self.N]))
        
        prob = [log(self.probability(i)) for i in pattern_list if self.probability(i)!=float('Inf')]
        if len(prob) == 0:
            return None
        return sum(prob)

    def __init__(self,sentence_list,N,feature_type):
        
        self.N = N
        preprocessed_list = preprocess_data(sentence_list)
        vocab,transformed_X = generate_features(preprocessed_list,1,1000,feature_type)
        self.feature_type = feature_type
        self.vocab = vocab
        transformed_X_UNK = [[word if word in vocab else "<UNK>" for word in w ] for w in transformed_X]
        self.V = len(list(set(flatten(transformed_X_UNK))))
        transformed_X = [['<s>']+w for w in transformed_X_UNK]
        pattern_list = []
        pattern_list_denom = []
        for i in transformed_X:
            for j in range(len(i)-N+2):
                pattern_list_denom.append(tuple(i[j:j+N-1]))
            for j in range(len(i)-N+1):
                pattern_list.append(tuple(i[j:j+N]))
        self.probabilities = Counter(pattern_list)
        self.probabilities_denom = Counter(pattern_list_denom)
        sum_total = len(pattern_list)
        sum_total_denom = len(pattern_list_denom)
        self.probabilities = defaultdict(int,{word: self.probabilities[word]/float(sum_total) for word in self.probabilities})
        self.probabilities_denom = defaultdict(int,{word :self.probabilities_denom[word]/float(sum_total_denom) for word in self.probabilities_denom})

def NGram_driver(X_train,Y_train,X_test):
    author_train = [X_train[i] for i in range(len(Y_train)) if Y_train[i] == 1]
    other_train = [X_train[i] for i in range(len(Y_train)) if Y_train[i]==0]

    model_author = NGram(author_train,3,'words')
    model_other = NGram(other_train,3,'words')
    
    predicted_labels = []
    for sentence in X_test:
        author_prob = model_author.log_prob(sentence)
        other_prob = model_other.log_prob(sentence)
        
        if author_prob == None:
            predicted_labels+=[0]
        elif other_prob == None:
            predicted_labels+=[1]
        elif  author_prob >= other_prob:
            predicted_labels+=[1]
        else:
            predicted_labels+=[0]
    return predicted_labels


def logistic_driver(X_train,Y_train,X_test):
    #X_train_prep = preprocess_data(X_train)
    #X_test_prep = preprocess_data(X_test)

    
    X_train_prep = X_train
    X_test_prep = X_test
    vocab_train,transformed_train_X = generate_features(X_train_prep,1,1000,'words')
    vocab_test, transformed_test_X = generate_features(X_test_prep,1,1000,'words')

    sentence_label_list_train = zip(transformed_train_X,Y_train)
    sentence_label_list_test = zip(transformed_test_X,[1]*len(X_test))

    features_train_count = format_features_sklearn(sentence_label_list_train,vocab_train,data_type='tf-idf')
    features_test_count = format_features_sklearn(sentence_label_list_test,vocab_train,data_type='tf-idf')
    
    features_train_type_token = format_features_sklearn(sentence_label_list_train,vocab_train,data_type='type-token')
    features_test_type_token = format_features_sklearn(sentence_label_list_test,vocab_train,data_type='type-token')
    
    features_train = join_features(features_train_count,features_train_type_token)
    features_test = join_features(features_test_count,features_test_type_token)
    
    
    model = LogisticRegression()
    model.fit(features_train,Y_train)
    
    predicted_labels = model.predict(features_test)
    
    return predicted_labels

def kernel_intersection(A,B):
    kernel_mat = [[0 for j in B] for i in A]
    for i in range(len(A)):
        for j in range(i,len(B)):
            kernel_mat[i][j] = sum([min(k,l) for k,l in zip(A[i],B[j])])
            kernel_mat[j][i] = kernel_mat[i][j]
    return kernel_mat

def svm_driver_sklearn(X_train,Y_train,X_test):
    #X_train_prep = preprocess_data(X_train)
    #X_test_prep = preprocess_data(X_test)
    
    X_train_prep = X_train
    X_test_prep = X_test
    vocab_train,transformed_train_X = generate_features(X_train_prep,1,1000,'words')
    vocab_test, transformed_test_X = generate_features(X_test_prep,1,1000,'words')
    
    sentence_label_list_train = zip(transformed_train_X,Y_train)
    sentence_label_list_test = zip(transformed_test_X,[0]*len(X_test))

    features_train_counts = format_features_sklearn(sentence_label_list_train,vocab_train,data_type='tf-idf')
    features_test_counts = format_features_sklearn(sentence_label_list_test,vocab_train,data_type='tf-idf')
    
    #features_train_length = format_features_sklearn(sentence_label_list_train,vocab_train,data_type='length')
    #features_test_length = format_features_sklearn(sentence_label_list_test,vocab_train,data_type='length')
    
    #features_train_type_token = format_features_sklearn(sentence_label_list_train,vocab_train,data_type='type-token')
    #features_test_type_token = format_features_sklearn(sentence_label_list_test,vocab_train,data_type='type-token')
    
    features_train = features_train_counts#,features_train_type_token)
    features_test = features_test_counts#,features_test_type_token)

    svm_model = SVC()
    svm_model.fit(features_train,Y_train)
    
    #return features_train,features_test
    return svm_model.predict(features_test)

def format_features_sklearn(feature_label_list,vocab,normalize=False,data_type='counts'):
    label_list = [label for feat,label in feature_label_list]
    feat_list = [feat for feat,label in feature_label_list]
    
    vocab_feature_space = {word : str(i+1) for i,word in enumerate(vocab)}
    
    unique_labels = list(set(label_list))
    unique_labels.sort()

    label_feature_space = {label : str(i) for i,label in enumerate(unique_labels)}
    
    feature_space = []
    if data_type == 'counts':
        for index,words_list in enumerate(feat_list):
            counts = Counter(words_list)
            if normalize:
                counts = L2_normalization(counts)
            feature_list = []
            for w in vocab_feature_space:
                if w in counts:
                    feature_list+=[counts[w]]
                else:
                    feature_list+=[0]
            feature_space+=[feature_list]


    elif data_type == 'tf-idf':
        unique_word_doc = [list(set(i)) for i in feat_list]
        doc_counts = Counter(flatten(unique_word_doc))
        N = len(feat_list)
        for index,word_list in enumerate(feat_list):
            feature_list = []
            tf = Counter(word_list)
            unique_words = list(set(word_list))
            for w in vocab_feature_space:
                if w in unique_words:
                    feature_list+=[tf[w]*log(N*1.0/doc_counts[w])]
                else:
                    feature_list+=[0]
            feature_space+=[feature_list]
    
    elif data_type == 'type-token':
        for index,word_list in enumerate(feat_list):
            types = len(list(set(word_list)))
            tokens = len(word_list)
            feature_space+=[[types/float(tokens)]]
    
    elif data_type == 'length':
        for index,word_list in enumerate(feat_list):
            feature_space+=[[len(word_list)]]
    return feature_space

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
    
    X_train_prep = preprocess_data(X_train)
    X_test_prep = preprocess_data(X_test)
    
    #X_test_prep = X_test
    #X_train_prep = X_train
    vocab_train,transformed_train_X = generate_features(X_train_prep,1,1000,'func_words')
    vocab_test, transformed_test_X = generate_features(X_test_prep,1,1000,'func_words')

    sentence_label_list_train = zip(transformed_train_X,Y_train)
    sentence_label_list_test = zip(transformed_test_X,[0]*len(X_test))

    outputfile_path_train = 'DataGen/models_train_svm'
    outputfile_path_test = 'DataGen/models_test_svm'

    generate_svm_files(sentence_label_list_train,vocab_train,outputfile_path_train,normalize=False)
    generate_svm_files(sentence_label_list_test,vocab_train,outputfile_path_test,normalize=False)

    predicted_labels = train_test_model(outputfile_path_train,outputfile_path_test)
    
    return predicted_labels


''' Harshal : Adding function generate_svm_files()

Description:
    This function will read the data as a list of (sentence,label) pairs and generate a file formated in the form required by libsvm ie (label features in the sentence). The output will be written in a file

Input:
    String containing the path to the directory containing the data, 
'''

def generate_svm_files(feature_label_list, vocab, output_file,normalize=False):
    label_list = [label for feat,label in feature_label_list]
    feat_list = [feat for feat,label in feature_label_list]
    
    vocab_feature_space = {word : str(i+1) for i,word in enumerate(vocab)}
    
    unique_labels = list(set(label_list))
    unique_labels.sort()

    label_feature_space = {label : str(i+1) for i,label in enumerate(unique_labels)}
    
    output_file_handle = open(output_file, "w")
    
    for index,words_list in enumerate(feat_list):
        counts = Counter(words_list)
        if normalize:
            counts = L2_normalization(counts)
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
   Strings containing the path to the train datafile and string path of test data whose labels are to be predicted. NOTE THAT DUE TO THE STRANGE DEFINITION OF svm_predict in libsvm, we have to pass a y_test to it that are just all ones.

Output:
  predicted labels
 
'''

def train_test_model(train_datafile, test_datafile):
    y_train, x_train = svm_read_problem(train_datafile)
    problem = svm_problem(y_train, x_train)
    param = svm_parameter('-t 0 -e .01 -m 1000 -h 0')
    m = svm_train(problem,param)
    Y_test, X_test = svm_read_problem(test_datafile)
    p_labels, p_acc, p_vals = svm_predict(Y_test,X_test,m)
    p_labels = [1 if int(i) == 2 else 0 for i in p_labels]
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

        if kl_value_author > kl_value_other:
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
