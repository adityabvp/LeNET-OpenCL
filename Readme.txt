Instructions to Run code

cd VGG19
For compilations :- g++ main.cpp Utils/utils.c -o fout -lOpenCL
To run it :- ./fout


About Files in folder VGG19

1.) Main.cpp :- This file contains the main code of VGG which calls each kernel of Convolution ,maxpool and dense layer.Parameter Testcases can be used to specify number of test cases during inference (1-10,000). 
2.) Model.h :- This file contain the VGG19 architecture , code is designed in such a way that just by changing this model file any new architecture can be implemented.
3.) Cl kernel folder :- This folder contains all the kernel of three layers (convolution,Max_pool,Dense layer).
4.) Utils folder :- It contains the utilities reuired
5.) REsults_log.txt :- This is log file for inference of 1000 cases. The accuracy achieved was 92% which is the current state of art
6.) weight_vgg.dat:- It contains the weights of VGG19 in binary form
7.) test_label_new.txt:- It contains the test labels (10,000)
8.)test_processed_new.txt:- It contains the the test input images in binary form (10,000)
9.) main_time.cpp.txt :- Is the main.cpp with timing analysis code

