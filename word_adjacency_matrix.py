

from tmp_util import *

class WAN:
    
    func_words = None
    word_mat = None
    window_size = None
    alpha = None
    
    def similarity(self,P2):
        prob = []
        for i in P2.word_mat:
            for j in P2.word_mat[i]:
                if self.word_mat[i][j] == 0:
                    prob+=[0]
                else:
                    prob+=[self.word_mat[i][j]*log(self.word_mat[i][j]/P2.word_mat[i][j])]
        return sum(prob)

    def __init__(self,sent_list):
        
        func_list = open('data/func_wan.txt','r')
        self.func_words = [i.strip() for i in func_list]
        func_list.close()
        
        prep_data = preprocess_data(sent_list,stem=False)
        vocab,transformed_features = generate_features(prep_data,feature_type='words')
        
        self.word_mat = defaultdict(lambda : defaultdict(int))
        
        self.window_size = 4
        self.alpha = 0.8

        for sent in transformed_features:
            for i in range(0,len(sent)-self.window_size+1,self.window_size):
                window_list = sent[i:i+self.window_size]
                func_list = [i for i in enumerate(window_list) if i[1] in self.func_words]
                for i in range(len(func_list)):
                    for j in range(i+1,len(func_list)):
                        word1 = func_list[i]
                        word2 = func_list[j]
                        self.word_mat[word1[1]][word2[1]] += self.alpha**(word2[0]-word1[0]-1)
                        
        
        for i in self.word_mat:
            total = sum(self.word_mat[i].values())
            for j in self.word_mat[i]:
                self.word_mat[i][j] = self.word_mat[i][j]/total
        


def WAN_driver(X_train,Y_train,X_test):
    X_train_author = [X_train[i] for i in range(len(Y_train)) if Y_train[i]==1]
    X_train_other = [X_train[i] for i in range(len(Y_train)) if Y_train[i]==0]
    
    author_profile = WAN(X_train_author)
    other_profile = WAN(X_train_other)

    predicted_labels = []
    for sample in X_test:
        Profile = WAN([sample])
        if author_profile.similarity(Profile) > other_profile.similarity(Profile):
            predicted_labels+=[0]
        else:
            predicted_labels += [1]
    return predicted_labels
