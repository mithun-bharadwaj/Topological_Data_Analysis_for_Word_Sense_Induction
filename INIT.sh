rm -rf boost_1_71_0 \
WSD_Evaluation_Framework \
WSD_Training_Corpora \
semeval-2013-task12-test-data \
trial \
w2v* \
temp.text* \
compute_barcodes

unzip semeval-2013-task12-test-data.zip
tar -xf boost_1_71_0.tar.xz   
tar -xf WSD_Evaluation_Framework.tar.xz
tar -xf semeval-12-trial.tar  
tar -xf WSD_Training_Corpora.tar.xz

g++ -o3 -I boost_1_71_0/ -std=c++11  compute_barcodes.cpp -o compute_barcodes

