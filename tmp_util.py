__author__ = 'Sajal/Harshal'

# TODO : refactor name of file

# name: load_data
# params:
# return: train_X,train_Y,test_X
# The directory location is hard coded. It should be in the same place as this file in a folder called data.
def load_data():
    # SAJAL
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
            train_Y.append(data[1])

    with open(input_dir+test_file,'r') as fp_test:
        for ine in fp_test:
            data = line.strip()
            test_X.append(data)

    return train_X,train_Y,test_X