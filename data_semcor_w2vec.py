import numpy as np
import scipy
import matplotlib
import subprocess
import gensim
import os
import pandas as pd
import collections
import datetime
import random
import sys
import xml.etree.ElementTree as ET

#####################################################################################################

"""Train Word2Vec on the corpus and return a (no. of words in vocab)x(dimension of embedding matrix)"""
def get_word_vectors():
    sentences, vocab = generate_corpus()

    sentences_split = []
    for s in sentences:
        sentences_split.append(s.split())
    
    if os.path.exists('w2v.model.wv.'+str(dimension)) == False:
        model = gensim.models.Word2Vec(sentences_split, size=dimension, window=5, min_count=5, workers=6, sg=1)
        model.wv.save("w2v.model.wv."+str(dimension))
        word_vectors = model.wv
    else:
        word_vectors = gensim.models.KeyedVectors.load("w2v.model.wv."+str(dimension), mmap = 'r')

    number_vocab_wv=0
    for w in vocab:
        if w in word_vectors:
            number_vocab_wv=number_vocab_wv+1

    w2v_vectors = np.zeros((number_vocab_wv, dimension))
    w2v_dict = {}
    count = 0
    for w in vocab:
        if w in word_vectors:
            w2v_vectors[count][:] = word_vectors[w]
            w2v_dict[w] = list(word_vectors[w])
            count += 1
    return w2v_dict, w2v_vectors

#####################################################################################################

"""Calculate barcodes"""

def get_barcodes(word_vectors,d,instance):
    f = open("temp.txt."+str(instance),"w+")
    for i in range(len(word_vectors)):
        for j in range(d):
            f.write("%e" % word_vectors[i][j])
            f.write(" ")
        f.write("\n")
    f.close()

    run_cmd="./compute_barcodes "+str(instance)
    barcodes = subprocess.run(run_cmd.split(), stdout=subprocess.PIPE)
    barcodes = (str(barcodes.stdout,'utf-8')).split()
    for i in range(len(barcodes)):
        barcodes[i] = float(barcodes[i])

    return barcodes

###################################################################################################

"""Sort words by cosine similarity"""
def sort_by_cos_dist(w2v_dict, local_homology_threshold, target_word):
    cos_dist = []
    words = []
    word_vec_topn = []

    w2v_array = np.asarray(list(w2v_dict.values()))
        

    cos_dist = 1.0- \
      np.divide(np.dot(w2v_array,
                       np.asarray(w2v_dict[target_word])),
                np.linalg.norm(w2v_array)) \
     / np.linalg.norm(np.asarray(w2v_dict[target_word]))

    words = list(w2v_dict.keys())

    cos_dist, words = (list(t) for t in zip(*sorted(zip(cos_dist, words))))

    print('words',len(words))
    for i in range(1,local_homology_threshold):
        word_vec_topn.append(w2v_dict[words[i]])

    return cos_dist[1:local_homology_threshold], words[1:local_homology_threshold], word_vec_topn

##################################################################################################################

"""Count barcodes and remove noise"""
def get_number_of_senses(life_span_vector):
    if len(life_span_vector)>1:
        life_span_vector = life_span_vector[:-1]
        mean = np.mean(life_span_vector)
        std_dev = np.std(life_span_vector)
        senses = np.sum(life_span_vector > (mean + 2*std_dev))
    #    print(mean + 2*std_dev)
        return senses+1
    elif len(life_span_vector)==1:
        print('93 ',life_span_vector)
        return 1
    else:
        print('96 ',life_span_vector)
        return 0


#################################################################################################################

""" Generate corpus"""
def generate(file_name):
    tree = ET.ElementTree(file=file_name)
    root = tree.getroot()

    lemma_list = []
    sentences = []
    poss = []
    targets = []
    targets_index_start = []
    targets_index_end = []
    lemmas = []

    for doc in root:
        for sent in doc:
            sentence = []
            pos = []
            target = []
            target_index_start = []
            target_index_end = []
            lemma = []
            for token in sent:
                assert token.tag == 'wf' or token.tag == 'instance'
                if token.tag == 'wf':
                    for i in token.text.split(' '):
                        sentence.append(i)
                        pos.append(token.attrib['pos'])
                        target.append('X')
                        lemma.append(token.attrib['lemma'])
                if token.tag == 'instance':
                    target_start = len(sentence)
                    for i in token.text.split(' '):
                        sentence.append(i)
                        pos.append(token.attrib['pos'])
                        target.append(token.attrib['id'])
                        lemma.append(token.attrib['lemma'])
                    target_end = len(sentence)
                    assert ' '.join(sentence[target_start:target_end]) == token.text
                    target_index_start.append(target_start)
                    target_index_end.append(target_end)
            sentences.append(sentence)
            poss.append(pos)
            targets.append(target)
            targets_index_start.append(target_index_start)
            targets_index_end.append(target_index_end)
            lemmas.append(lemma)


    corpus = []
    vocab_dict = {}
    gold_keys = []
    with open(file_name.rstrip('.data.xml') + '.gold.key.txt', "r", encoding="utf-8") as m:
        key = m.readline().strip().split()
        while key:
            gold_keys.append(key[1])
            key = m.readline().strip().split()


    output_file = file_name.rstrip('.data.xml') + '.csv'
    with open(output_file, "w", encoding="utf-8") as g:
        g.write('sentence\ttarget_index_start\ttarget_index_end\ttarget_id\ttarget_lemma\ttarget_pos\tsense_key\n')
        num = 0
        for i in range(len(sentences)):
            for j in range(len(targets_index_start[i])):
                sentence = ' '.join(sentences[i])
                target_start = targets_index_start[i][j]
                target_end = targets_index_end[i][j]
                target_id = targets[i][target_start]
                target_lemma = lemmas[i][target_start]
                for w in sentence.split():
                    vocab_dict[w]=1
                target_pos = poss[i][target_start]
                sense_key = gold_keys[num]
                num += 1
                g.write('\t'.join((sentence, str(target_start), str(target_end), target_id, target_lemma, target_pos, sense_key)))
                g.write('\n')
                if sentence not in corpus:
                    corpus.append(sentence)
    return corpus, list(vocab_dict.keys())

################################################################################################################
def generate_corpus():
    eval_dataset = ['senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015', 'ALL']
    # train_dataset = ['SemCor', 'SemCor+OMSTI']
    train_dataset = ['SemCor+OMSTI']
    # train_dataset = ['SemCor']

    file_name = []
    for dataset in eval_dataset:
        file_name.append('./WSD_Evaluation_Framework/Evaluation_Datasets/' + dataset + '/' + dataset + '.data.xml')
    for dataset in train_dataset:
        file_name.append('./WSD_Training_Corpora/' + dataset + '/' + dataset.lower() + '.data.xml')

    for file in file_name:
#        print(file)
        corpus, vocab = generate(file)
        # print(corpus)
        return corpus, vocab
#################################################################################################################
def get_sense_counts():
    eval_dataset = ['senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015', 'ALL']
#    train_dataset = ['SemCor']
    train_dataset = ['SemCor+OMSTI']

    file_path = []
    for dataset in eval_dataset:
        file_path.append('./WSD_Evaluation_Framework/Evaluation_Datasets/' + dataset + '/' + dataset)
    for dataset in train_dataset:
        file_path.append('./WSD_Training_Corpora/' + dataset + '/' + dataset.lower())

    for file_name in file_path:
        gold_key_file_name = file_name + '.gold.key.txt'
        train_file_name = file_name + '.csv'
        if file_name == './WSD_Training_Corpora/SemCor+OMSTI/semcor+omsti':
#        if file_name == './WSD_Training_Corpora/SemCor/semcor':
            train_file_final_name = file_name + '_train_sent_cls_ws.csv'
        else:
            train_file_final_name = file_name + '_test_sent_cls_ws.csv'

        # print(gold_key_file_name)
        # print(train_file_name)
        # print(train_file_final_name)

        sense_counts_dict = generate_sense(gold_key_file_name, train_file_name, train_file_final_name)
#        print(corpus)
#        exit()
        #print(gold_key_file_name, train_file_name, train_file_final_name)        
        return sense_counts_dict


#################################################################################################################
def generate_sense(gold_key_file_name, train_file_name, train_file_final_name):

    sense_data = pd.read_csv("./wordnet/index.sense.gloss",sep="\t",header=None).values

    lemma2sense_dict = collections.OrderedDict()

    d = dict()
    for i in range(len(sense_data)):
        s = sense_data[i][0]
        pos = s.find("%")
        try:
            d[s[:pos + 2]].append((sense_data[i][0],sense_data[i][-1]))
        except:
            d[s[:pos + 2]]=[(sense_data[i][0], sense_data[i][-1])]

    train_data = pd.read_csv(train_file_name,sep="\t",na_filter=False).values

    gold_keys=[]
    with open(gold_key_file_name,"r",encoding="utf-8") as f:
        s=f.readline().strip()
        while s:
            tmp = s.split()[1:]
            gold_keys.append(tmp)
            s=f.readline().strip()

    with open(train_file_final_name,"w",encoding="utf-8") as f:
        f.write('target_id\tlabel\tsentence\tgloss\tsense_key\n')
        for i in range(len(train_data)):
            assert train_data[i][-2]=="NOUN" or train_data[i][-2]=="VERB" or train_data[i][-2]=="ADJ" or train_data[i][-2]=="ADV"
            orig_sentence = train_data[i][0].split(' ')
            start_id = int(train_data[i][1])
            end_id = int(train_data[i][2])
            sentence = []
            for w in range(len(orig_sentence)):
                if w == start_id or w == end_id:
                    sentence.append('"')
                sentence.append(orig_sentence[w])
            sentence = ' '.join(sentence)
            orig_word = ' '.join(orig_sentence[start_id:end_id])
            senses = []
            for category in ["%1", "%2", "%3", "%4", "%5"]:
                word = train_data[i][-3]
                query = word+category
                try:
                    sents = d[query]
                    gold_key_exist = 0
                    for j in range(len(sents)):
                        if sents[j][0] in gold_keys[i]:
                            f.write(train_data[i][3]+"\t"+"1")
                            gold_key_exist = 1
                        else:
                            f.write(train_data[i][3]+"\t"+"0")
                        f.write("\t"+sentence+"\t"+orig_word+" : "+sents[j][1]+"\t"+sents[j][0]+"\n")
                        senses.append(sents[j][1])
                        
                    assert gold_key_exist == 1
                except:
                    pass
            lemma2sense_dict[orig_word] = senses
            
    sense_counts_dict={}
    for key,val in lemma2sense_dict.items():
        sense_counts_dict[key]=len(val)
    return sense_counts_dict

##################################################################

dimension = int(sys.argv[1])
local_homology_threshold = int(sys.argv[2])
instance=random.randint(1,100000)
max_iter=200
max_word_sense=19
min_word_sense=10
verbose=False

"""Extract vocabulary from SemCor dataset"""
# vocab = semcor.words()
corpus, vocab = generate_corpus()
sense_counts_dict=get_sense_counts()
w2v_dict, w2v_vectors = get_word_vectors()

rel_error=[]
abs_error=[]
counter=1
for word in sense_counts_dict:
    if counter>max_iter: break
    
    counter=counter+1
    
    if (sense_counts_dict[word]<=max_word_sense) and (sense_counts_dict[word]>=min_word_sense) and word in w2v_dict:
        closest_n_cos_dist, closest_n_words , closest_n_word_vec  = sort_by_cos_dist(w2v_dict, local_homology_threshold, word)

        life_span = get_barcodes(closest_n_word_vec, dimension,instance)

        rel_error.append(abs(get_number_of_senses(life_span)-sense_counts_dict[word])/sense_counts_dict[word])

        abs_error.append(abs(get_number_of_senses(life_span)-sense_counts_dict[word]))

print('dimen',dimension,\
'local_homology_threshold',local_homology_threshold,\
'rel_error',np.mean(rel_error),\
'abs_error',np.mean(abs_error))


