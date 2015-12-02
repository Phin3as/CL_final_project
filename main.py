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

    cv.cross_validation(X_train,Y_train,10,models.all_zero)
    cv.cross_validation(X_train,Y_train,10,models.all_one)

    end_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))