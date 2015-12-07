import time
import tmp_util
import models
import cross_validation as cv

__author__ = 'Sajal/Harshal'

# TODO: refactor name of file

if __name__ == '__main__':
    start_time = time.time()

    X_train,Y_train,X_test=tmp_util.load_data()
    print 'Data Loaded'

    # cv.cross_validation(X_train,Y_train,10,models.all_zero)
    # cv.cross_validation(X_train,Y_train,10,models.all_one)

    word_dict=tmp_util.get_words_from_corpus(X_train,X_test)
    print 'WORD DICT EXTRACTED'

    word_freq_train,word_freq_test,word_ind=tmp_util.convert_to_word_freq(word_dict,X_train,X_test)
    print 'WORD FREQ TABLE EXTRACTED'

    word_freq_train,word_freq_test=tmp_util.normalize_data(word_freq_train,word_freq_test)
    print 'Word Freq Normalised'

    cv.cross_validation(word_freq_train,Y_train,10,models.model_cos_sim)

    # op=models.model_cos_sim(word_freq_train,Y_train,word_freq_test)
    # print op
    # print op.__len__()

    end_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))