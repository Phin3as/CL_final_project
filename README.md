Team Name : Phinedroids & Ferbots


Summary of Three papers and Methods Utilized:

Survey Paper:

1. Most Frequent Features (words) are most helpful in author detection since they capture the distribution of functional words

2. Lexical Features include:  
    a. N-gram approach at word and character level( paper mentions 4-gram works best at character level)
    b. Writing errors and spell check ( Requires a good spell checker. http://norvig.com/spell-correct.html is supposedly           good)
    c. Perform Lemmatization and stemming and perform frequency counting
    d. Vocabulary richness (type/token ratio), word lengths, sentence lengths

3. Syntactic Features include:
    a. Rule frequencies using POS tagger ( it works bad alone according to the paper) + Lexical Features.
    b. Frequency Counts of POS tags and NGram model Based on NGram Frequencies. 
    c. Syntactic error measuring (Really difficult to implement according to me)

4. Semantic and Application Specific Approaches include:
    a. Systemic Functional Grammar : Categorize senteces as their actions on previous sentence e.g 'ELABORATION'
    b. Content Specific Keywords( Not sure if the excerpts are all on same topic. Will check)
    c. Instable Features : (Frequency of I assume)Words with similar meanings but are different e.g 'rise' and 'ascension'          rather than functional words
    d. Synonyms, Hypernynm information, Latent Semantic Analysis (Unsure about this)

5. Other Stuff :
    a. Profile based approach: Simplified Profile Intersection(similarity measure), Z-score based approach, Common NGram            distance (might perform badly due to imbalance in number of  positive and negative example in training file)
    b. Compression Based Approach : Compress a file of author excerpts and another file containing test example and check s         size. RAR works best.

Word Adjacency Networks:

1. Pretty cool approach. Take 45-60 of most common stop words. Build an transition matrix between them using the distance       metric specified in the paper[Basically the distance between their indexes in a given window with some decay factor that     penalizes function words far apart].
2. Use KL-Divergence( with somthing called limiting distribution....will read more about it ) to measure simlarity between      transition matrix of unlabeled text and transition matrix of author and choose author that gives minimum KL-Divergence.
3. Requires some parameter tuning through cross-validation


SVM based Approach:

1. Almost similar to our baseline approach. Author states that data used for training SVM performs well when normalized by L2    norm. 
2. Features used by author include function word - tag pairs. Lemmatize the words in the corpus. Tag it through Parser(ignore    SUB,VERB and ADJ tags). The tags correspond to the different tags the lemmatized function word is tagged with. Final         features also included frequency of words of different lengths and Bigram Frequency of Tag-word paris (will recheck this).



Other Methods we can try:

1.KL-Divergence based similarity - 72% CV accuracy

2.Using sklearn to run basic algorithms like:
  a. KNN.
  b. Decision Trees with Boosting.
  c. Naive Bayes
  d. Neural Nets and RBF-Networds(the number of samples seems less for this)
  e. NGram with replacement(using word2vec)

Following methods are implemented (in file models.py):

All 0
All 1
