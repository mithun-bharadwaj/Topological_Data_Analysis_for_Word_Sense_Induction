
#python3 data_semcor_w2vec.py word_vec_dimension local_threshold

locality=100
python3 data_semcor_w2vec.py 100 $locality &
python3 data_semcor_w2vec.py 500 $locality &
python3 data_semcor_w2vec.py 1000 $locality &

locality=200
python3 data_semcor_w2vec.py 100 $locality &
python3 data_semcor_w2vec.py 500 $locality &
python3 data_semcor_w2vec.py 1000 $locality &


