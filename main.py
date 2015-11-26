import time
import tmp_util

__author__ = 'Sajal/Harshal'

# TODO: refactor name of file

if __name__ == '__main__':
    start_time = time.time()

    train_X,train_Y,test_X=tmp_util.load_data()

    end_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))