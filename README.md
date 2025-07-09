# See the paper CryptDNN: A fast privacy-preserving machine learning architecture on deep neural network

## This work combined BFV(FHE) + SSE(Searchable symmetric encryption) to achiveve privacy-preserving image classification on deep neural network

## To run this work, you should install some packages as follows:
### numpy
### torch
### torchvision
### Crypto
### Pyfhel=2.3.1

## Note to intsall Pyfhel=2.3.1, please follow these steps as follow
### 1. git clone --recursive https://github.com/ibarrond/Pyfhel.git --branch=v2.3.1
### 2. cd Pyfhel
### 3. cd Pyfhel/SEAL/SEAL/seal
### 4. vim locks.h
### 5. add two notations after \#include <shared_mutex>
       #include<shared_mutex>
       #include<mutex>
       #include<stdexcept>
### 6. pip install .
### 7. cd Pyfhel/Pyfhel
### 8. rm -rf __init__.py


# If you have any questions, please contact me

