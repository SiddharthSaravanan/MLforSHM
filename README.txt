
MLforSHM,
by SIDDHARTH S, SHAH VISHWA VIPULKUMAR and THUKRAL SHAURYA MANISH

The objectives of this project align with the objectives of the paper [1]. We plan to use Machine Learning for Structural Health Monitoring. 
The ultimate aim of this project is to take noisy accelerometer data from accelerometers placed on two adjacent floors and be able to accurately predict
the status of the building. The status of the building can be one of 3 classes, Immediate Occupancy (IO), Life Safety (LS) or Collapse Prevention (CP) 
as explained in the paper summary of [1]. 

This project contains 4 folders - Code, Documents, Input, Output

Code folder contains:
finalc.py
Link to code on Colab.txt
datasetgeneration.m
testsetgeneration.m

Documents folder contains:
MLforSHMreport.pdf
REFERENCES.pdf
dataset creation.png
Denoising approach.png
Direct approach.png
Getting final classification.png

Input folder contains:
noisy.xlsx
test_noisy.xlsx

Output folder contains:
pure.xlsx
test_pure.xlsx

The code flowcharts given in the documents folder has been coded in python and the code explaination is given in the code itself through comments.

There is more info about the datasets, the source code and the procedure to execute code given below...

A.Datasets - MATLAB
These datasets have been created using MATLAB code datasets.m, we have made use of sin and awgn functions
for randomizing and creating the representative accelerometer data
1.Input Data Set for Training: pure.xlsx, noisy.xlsx
2.Input Data Set for Testing: test_pure.xlsx, test_noisy.xlsx


B. Source Code - Python
These datasets have been fed to the code finalc.ipynb, This code performs the following high level operations.
1. Randomization and Reshaping the datasets to fit the Deep Neural Networks
2. CNN and ANN models for denoising noisy data which have been finalized after various iterations and accuracy on the test set.
3. Testing the model on the testing datasets and checking the extent of denoising
4. Labeling the pure, noisy and denoised dataset into IO, LS and CP using IDR calculations as mentioned in [1]
5. Testing the accuracy of classification for noisy and denoised datasets against pure dataset as reference.


C. Procedure to Execute the Code
We have implemented and tested this code on Google Colab and the steps to execute the code is given here.
We strongly recommend running the code using the first method as the second method is very likely to give issues during the installation of
TensorFlow and has not been extensively tested

There are two ways to execute the code:
a)On google colab
b)On the terminal using finalc.py

a)Steps to run the code on google colab

1.Open the below link

https://colab.research.google.com/drive/1wVfQ8GnsTKe9uhZHovUatj72gWzqNBdJ?usp=sharing

2.connect to a runtime
3.upload the dataset files pure.xlsx, noisy.xlsx, test_pure.xlsx, test_noisy.xlsx to the colab session
4.Change the runtime type to GPU
5.run the first cell which installs TensorFlow
6.Run the next cell which contains all the code

b)Steps to run the code in the terminal

1.install the latest version of python
2.Open the terminal and execute the following pip install commands

pip install numpy
pip install -q tensorflow tensorflow-datasets
pip install matplotlib
pip install pandas
pip install scipy

3.include pure.xlsx, noisy.xlsx, test_pure.xlsx, test_noisy.xlsx as well as finalc.py in the same directory
4.run finalc.py on the terminal


References
[1] A. Ibrahim, A. Eltawil, Y. Na and S. El-Tawil, &quot;A Machine Learning Approach for
Structural Health Monitoring Using Noisy Data Sets,&quot; in IEEE Transactions on Automation Science and Engineering, vol. 17, no. 2, pp. 900-908, April 2020, doi:
10.1109/TASE.2019.2950958.