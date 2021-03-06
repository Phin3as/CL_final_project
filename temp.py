
'''Harshal : Added Function get_google_pos
Description:
   The function creates a google pos dictionary which maps penn tree bank tags to google universal tags

Input:
   A string containing file name

Output:
   A dictionary containing a mapping of penn tree bank tags to google tags

'''

def get_google_pos(f):
    f_handle = open(f,"r")
    google_dict = {}
    for line in f_handle:
        key,value = line.strip().split()
        google_dict[key] = value
    return google_dict

'''Harshal : Adding function parse_tagged_file

Description:
   The function parses Penn tree bank files and outputs a dictionary c

'''

def parse_taggedfile(wsjfile, tagmap):
    wsjfile_handle = open(wsjfile,"r")
    pos_list = []
    pos_line_list = []
    for line in wsjfile_handle:
        if len(line.strip()) == 0 or "=" in line:
            if len(pos_line_list) != 0:
                pos_list+=[pos_line_list]
                pos_line_list = []
            continue
        temp_list = line.strip().split()
        temp_list = filter(lambda x: x != '[' and x!=']' , temp_list)
        for s in temp_list:
            s_list = s.split('/')
            v1 = ''.join(s_list[0:-1])
            v2 = s_list[-1]
            pos_line_list +=[(v1.lower(),v2)]
    wsjfile_handle.close()
    if len(pos_line_list) != 0:
        pos_list+=[pos_line_list]
    pos_list = transform_tags(pos_list,tagmap)
    return pos_list

'''Harshal : Adding function create_vocabulary_pos

Description:
   The function creates a vocabulary from the words in a list of files. It accepts the minimium frequency of a words to included in the vocabulary and the number of words to be returned

Input:
   a list of files, an integer limit specifying the minimum count, the google_pos tags (Because it was there in the assignment) and the amount of words to be retrieved.

Output:
   a list of words sorted according to counts

'''
def create_vocabulary_pos(filelist,limit, google_pos,topk):
    vocab = []
    for f in filelist:
        pos_list = parse_taggedfile(f, google_pos)
        vocab += [word1 for word1, word2 in flatten(pos_list)]
    word_count = Counter(vocab)
    tup_list = [(word, word_count[word]) for word in word_count]
    tup_list.sort(key = lambda x : (x[1],x[0]),reverse=True)
    word_count_sorted = [word for word, value in tup_list if value >=limit]
    if topk == -1:
        topk=len(word_count_sorted)
    vocab = ["<s>","<UNK>"]+word_count_sorted[:topk]
    return vocab


''' Harshal : Adding function tranform_tags
Description:
    The function is used so as to allow parse_taggedfile function to output both google_pos tags and Penn tree bank tags.

Input:
   a list containing tuples with a word and Penn treebank tags, and a dictionary containing a mapping from Penn TreeBank tags to google_pos_tags

Output:
   a list containing word, google_pos tuples if new_tag is not empty else the same list

'''
def transform_tags(old_tag_list, new_tag):
    result_list = []
    if len(new_tag)!=0:
        for i in old_tag_list:
            new_tup_list = []
            for j in i:
                jtemp = list(j)
                jtemp[1] = new_tag[jtemp[1]]
                new_tup_list += [tuple(jtemp)]
            result_list+=[new_tup_list]
        return result_list
    return old_tag_list


''' Harshal : Including prep_data function to prepare data to be feed to libsvm

Description : 
    prep_data creates a file containing the features of words in a corpus in the vocabulary. The number of words are specified by the window size. If wouds are not present in the vocabulary they are marked with "<UNK>" tag. Additionally a <s> tag is added at the beginning of a sentence. The pos_list is given by the parse_taggedfile function

Input: 
   String of directory name, String containing output file where the features are written, a dictionary tagmap and the vocab list

Output:
   None
'''
    
def prep_data(dirname, outfile, windowsize, tagmap, vocab):
    file_list = get_all_files(dirname)
    outfile_handle = open(outfile,"w")
    for f in file_list:
        pos_list = parse_taggedfile(f, tagmap)
        pos_sent = [[word2 for word1,word2 in sentence] for sentence in pos_list]
        word_list = [["<s>"]*(windowsize/2)+[word1 if word1 in vocab else "<UNK>" for word1,word2 in sentence]+["<s>"]*(windowsize/2) for sentence in pos_list]
        for pos_sent_i, sentence in zip(pos_sent,word_list):
            window_iterator = 0
            for s in pos_sent_i:
                context_window = ' '.join(sentence[window_iterator : window_iterator + windowsize])
                window_iterator+=1
                outfile_handle.write(s+'\t'+context_window+'\n')
    outfile_handle.close()
        
'''Harshal : Adding function convert_to_svm

Description:
   The function takes in the pre-processed file and converts it to a format used by libsvm. The function creates a feature vector representation of the words, indexing the words in the vocabulary.

Input:
   string containing the path of the preppedfile, the path of the output file, the google pos set and the vocabulary representing the features

Output:
   None

'''
def convert_to_svm(preppedfile, outfile, posset, vocab):
    vocab.sort()
    outfile_handle = open(outfile,"w")
    feature_space = {vocab[i-1] : i for i in range(1,len(vocab)+1)}
    preppedfile_handle = open(preppedfile, "r")
    V = len(vocab)
    posset_list = list(posset)
    posset_list.sort()
    posset_feature_space = {posset_list[i-1] : i for i in range(1,len(posset_list)+1)}
    for line in preppedfile_handle:
        line_list = line.strip().split()
        libsvm_list = [str(posset_feature_space[line_list[0]])]
        for n,e in enumerate(line_list[1:]):
            libsvm_list+=[str(n*V+feature_space[e])+":"+'1']
        final_svm_str = ' '.join(libsvm_list)
        outfile_handle.write(final_svm_str+"\n")
    outfile_handle.close()
    preppedfile_handle.close()
