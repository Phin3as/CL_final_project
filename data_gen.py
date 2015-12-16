from script import *


X_train,Y_train,X_test = load_data()


author_data = [X_train[i] for i in range(len(X_train)) if Y_train[i]==1]
other_data = [X_train[i] for i in range(len(X_train)) if Y_train[i]==0]

print 'Number of Examples Before preprocessing',len(author_data),len(other_data)
print 'Average Length of Examples Before preprocessing',sum([len(i) for i in author_data])/float(len(author_data)),sum([len(i) for i in other_data])/float(len(other_data))

print 'Sentences per excerpt Before preprocessing',sum([len(sent_tokenize(s.decode('utf-8'))) for s in author_data])/float(len(author_data)),sum([len(sent_tokenize(s.decode('utf-8'))) for s in other_data])/float(len(other_data))

print 'Words per excerpt Before preprocessing',sum([len(sent_tokenize(s.decode('utf-8'))) for s in author_data])/float(len(author_data)),sum([len(sent_tokenize(s.decode('utf-8'))) for s in other_data])/float(len(other_data))



X_train_prep = preprocess_data(X_train,lemmatize=False,stem=False)
X_test_prep = preprocess_data(X_test,lemmatize=False,stem=False)


author_data = [X_train_prep[i] for i in range(len(X_train)) if Y_train[i]==1]
other_data = [X_train_prep[i] for i in range(len(X_train)) if Y_train[i]==0]

print 'Number of Examples',len(author_data),len(other_data)
print 'Average Length of Examples',sum([len(i) for i in author_data])/float(len(author_data)),sum([len(i) for i in other_data])/float(len(other_data))

print 'Sentences per excerpt',sum([len(sent_tokenize(s.decode('utf-8'))) for s in author_data])/float(len(author_data)),sum([len(sent_tokenize(s.decode('utf-8'))) for s in other_data])/float(len(other_data))

print 'Words per excerpt',sum([len(sent_tokenize(s.decode('utf-8'))) for s in author_data])/float(len(author_data)),sum([len(sent_tokenize(s.decode('utf-8'))) for s in other_data])/float(len(other_data))



vocab_author,transformed_author = generate_features(author_data,1,1500,'words')
vocab_other,transformed_other = generate_features(other_data,1,1500,'words')

'''vocab_train,transformed_train_X = generate_features(X_train_prep,1,1500,'words')

vocab_train_func,transformed_train_X_func = generate_features(X_train_prep,1,1000,'func_words')

#vocab_train_pos,transformed_train_X_pos = generate_features(X_train_prep,1,1000,'pos')

vocab_train_char_bigram,transformed_train_X_char_bigram = generate_features(X_train_prep,1,1000,'char',ngram=True)

vocab_train_char_trigram,transformed_train_X_char_trigram = generate_features(X_train_prep,1,1000,'char',ngram=True,ngram_count = 3)

vocab_train_func_bigram,transformed_train_X_func_bigram = generate_features(X_train_prep,1,1000,'func_words',ngram=True)

#vocab_train_pos_bigram,transformed_train_X_char = generate_features(X_train_prep,1,1000,'pos',ngram=True)


f = open('data_for_graph/vocab_counts','w')
f.write(' '.join(vocab_train))
f.close()


f = open('data_for_graph/vocab_func','w')
f.write(' '.join(vocab_train_func))
f.close()


#f = open('data_for_graph/vocab_pos','w')
#f.write(' '.join(vocab_train_pos))
#f.close()


f = open('data_for_graph/vocab_char_bigram','w')
f.write(' '.join([str(i) for i in vocab_train_char_bigram]))
f.close()


f = open('data_for_graph/vocab_char_trigram','w')
f.write(' '.join([str(i) for i in vocab_train_char_trigram]))
f.close()


#f = open('data_for_graph/vocab_pos_bigram','w')
#f.write(' '.join([str(i) for i in vocab_train_pos_bigram]))
#f.close()


sentence_label_list_train = zip(transformed_train_X,Y_train)
sentence_label_list_train_func = zip(transformed_train_X_func,Y_train)
#sentence_label_list_train_pos = zip(transformed_train_X_pos,Y_train)
sentence_label_list_train_char_bigram = zip(transformed_train_X_char_bigram,Y_train)
sentence_label_list_train_char_trigram = zip(transformed_train_X_char_trigram,Y_train)
sentence_label_list_train_func_bigram = zip(transformed_train_X_func_bigram,Y_train)
#sentence_label_list_train_pos_bigram = zip(transformed_train_X_func_bigram,Y_train)


features_train_words = format_features_sklearn(sentence_label_list_train,vocab_train,data_type='counts')
features_train_words_tfidf = format_features_sklearn(sentence_label_list_train,vocab_train,data_type='tf-idf')
features_train_func = format_features_sklearn(sentence_label_list_train_func,vocab_train_func,data_type='tf-idf')
#features_train_pos = format_features_sklearn(sentence_label_list_train_pos,vocab_train_pos,data_type='tf-idf')
features_train_char_bigram = format_features_sklearn(sentence_label_list_train_char_bigram,vocab_train_char_bigram,data_type='tf-idf')
features_train_char_trigram = format_features_sklearn(sentence_label_list_train_char_trigram,vocab_train_char_trigram,data_type='tf-idf')
#features_train_pos_bigram = format_features_sklearn(sentence_label_list_train_pos_bigram,vocab_train_pos_bigram,data_type='tf-idf')
features_train_func_bigram = format_features_sklearn(sentence_label_list_train_func_bigram,vocab_train_func_bigram,data_type='tf-idf')


author_words = [features_train_words[i] for i in range(len(Y_train)) if Y_train[i]==1]
author_words_tfidf = [features_train_words_tfidf[i] for i in range(len(Y_train)) if Y_train[i]==1]
author_func = [features_train_func[i] for i in range(len(Y_train)) if Y_train[i]==1]
#author_pos = [features_train_pos[i] for i in range(Y_train) if Y_train[i]==1]
author_char_bigram = [features_train_char_bigram[i] for i in range(len(Y_train)) if Y_train[i]==1]
author_char_trigram = [features_train_char_trigram[i] for i in range(len(Y_train)) if Y_train[i]==1]
#author_pos_bigram = [features_train_pos_bigram[i] for i in range(Y_train) if Y_train[i]==1]
author_func_bigram = [features_train_func_bigram[i] for i in range(len(Y_train)) if Y_train[i]==1]




other_words = [features_train_words[i] for i in range(len(Y_train)) if Y_train[i]==0]
other_words_tfidf = [features_train_words_tfidf[i] for i in range(len(Y_train)) if Y_train[i]==0]
other_func = [features_train_func[i] for i in range(len(Y_train)) if Y_train[i]==0]
#other_pos = [features_train_pos[i] for i in range(Y_train) if Y_train[i]==0]
other_char_bigram = [features_train_char_bigram[i] for i in range(len(Y_train)) if Y_train[i]==0]
other_char_trigram = [features_train_char_trigram[i] for i in range(len(Y_train)) if Y_train[i]==0]
#other_pos_bigram = [features_train_pos_bigram[i] for i in range(Y_train) if Y_train[i]==0]
other_func_bigram = [features_train_func_bigram[i] for i in range(len(Y_train)) if Y_train[i]==0]

write_features_to_file(author_words,'data_for_graph/author_words',' ')
write_features_to_file(author_words_tfidf,'data_for_graph/author_words_tfidf',' ')
write_features_to_file(author_func,'data_for_graph/author_func',' ')
#write_features_to_file(author_pos,'data_for_graph/author_pos',' ')
write_features_to_file(author_char_bigram,'data_for_graph/author_char_bigram',' ')
write_features_to_file(author_char_trigram,'data_for_graph/author_char_trigram',' ')
#write_features_to_file(author_pos_bigram,'data_for_graph/author_pos_bigram',' ')
write_features_to_file(author_func_bigram,'data_for_graph/author_func_bigram',' ')



write_features_to_file(other_words,'data_for_graph/other_words',' ')
write_features_to_file(other_words_tfidf,'data_for_graph/other_words_tfidf',' ')
write_features_to_file(other_func,'data_for_graph/other_func',' ')
#write_features_to_file(other_pos,'data_for_graph/other_pos',' ')
write_features_to_file(other_char_bigram,'data_for_graph/other_char_bigram',' ')
write_features_to_file(other_char_trigram,'data_for_graph/other_char_trigram',' ')
#write_features_to_file(other_pos_bigram,'data_for_graph/other_pos_bigram',' ')
write_features_to_file(other_func_bigram,'data_for_graph/other_func_bigram',' ')

'''
unique_author = list(set(flatten(transformed_author)))
unique_other = list(set(flatten(transformed_other)))

total_author = sum([len(i) for i in transformed_author])
total_other = sum([len(i) for i in transformed_other])


print 'Unique Author and Other',len(unique_author),len(unique_other)
print 'Total Author and Other',total_author,total_other

vocab_train,transformed_train_X = generate_features(X_train_prep,1,1500,'words')
vocab_test,transformed_test_X = generate_features(X_test_prep,1,1500,'words')


f = open('data_for_graph/top_words','w')
f.write('Top Author words :'+'\n')
f.write(' '.join(vocab_author[1:20])+'\n')
f.write('Other Author words :'+'\n')
f.write(' '.join(vocab_other[1:20])+'\n')
f.write('Combined Vocab Train words :'+'\n')
f.write(' '.join(vocab_train[1:20])+'\n')
f.write('Combined Vocab Test words'+'\n')
f.write(' '.join(vocab_test[1:20])+'\n')

print 'Words used by author but not by others and in vocabulary',len(set(vocab_train)&(set(vocab_author)-set(vocab_other)))
print 'Words used by others but not by author and in vocabulary',len(set(vocab_train)&(set(vocab_other)-set(vocab_author)))

f.close()
