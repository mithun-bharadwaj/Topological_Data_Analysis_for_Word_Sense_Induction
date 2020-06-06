
import numpy as np
import scipy
import matplotlib
import subprocess
import gensim 
import glove

########################################################

dimension = 100
glove_no_components = 5

########################################################

def get_barcodes(word_vectors,d):
    f = open("temp.txt","w+")
    for i in range(len(vocab)):
        for j in range(d):
            f.write("%e" % word_vectors[i][j])
            f.write(" ")
        f.write("\n")
    f.close()

    compile_cmd="g++ -o3 -I boost_1_71_0/ -std=c++11 compute_barcodes.cpp -o compute_barcodes"
    compile_proc = subprocess.run(compile_cmd.split(), stderr=subprocess.PIPE)
    if compile_proc.stderr!=b'':
        print(compile_proc.stderr)
        quit()

    run_cmd="./compute_barcodes"
    barcodes = subprocess.run(run_cmd.split(), stdout=subprocess.PIPE)
    barcodes = (str(barcodes.stdout,'utf-8')).split()
    for i in range(len(barcodes)):
        barcodes[i] = float(barcodes[i])

    print('Death Dates')
    print(barcodes)
    return barcodes

########################################################

corpus=[['word','words'],['foo','choice'],['bar','koo']] # list of articles which are lists of words

vocab_dict={}
for a in corpus:
	for w in a:
		vocab_dict[w]=1
vocab=[]
for w in vocab_dict:
	vocab.append(w)

########################################################
glove_corpus = glove.Corpus() 
glove_corpus.fit(corpus, window=10)
glove_model = glove.Glove(no_components=glove_no_components, learning_rate=0.05)
glove_model.fit(glove_corpus.matrix, epochs=30, no_threads=4, verbose=False)
glove_model.add_dictionary(glove_corpus.dictionary)

word_vectors_glove=np.zeros((len(vocab),glove_no_components))

count=0
for w in vocab:
	word_vectors_glove[count][:]=glove_model.word_vectors[glove_model.dictionary[w]]
	count=count+1
	
get_barcodes(word_vectors_glove,glove_no_components)
########################################################
model1 = gensim.models.Word2Vec(corpus, min_count = 1,  
                              size = dimension, window = 5) 

model2 = gensim.models.Word2Vec(corpus, min_count = 1, size = dimension, 
                                             window = 5, sg = 1) 

word_vectors_w2v=np.zeros((len(vocab),dimension))

count=0
for w in vocab:
	word_vectors_w2v[count][:]=model1.wv[w]
	count=count+1
	
get_barcodes(word_vectors_w2v,dimension)
##########################################################



