from Pyfhel import Pyfhel

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import time
from BFVSSE_MNIST import util

import sys

transform = transforms.ToTensor()

test_set = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)



model_file = "/home/lilvmy/paper-demo/paper7/CryptDNNExperiment/BFVSSE_MNIST/LeNet1_single_ReLU.pt"

class ReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t):
        return torch.where(t > 0, t, 0)
model = torch.load(model_file)
model.eval()

# Code for matrix encoding/encryption
def encode_matrix(HE, matrix):
    try:
        return np.array(list(map(HE.encodeFrac, matrix)))
    except TypeError:
        return np.array([encode_matrix(HE, m) for m in matrix])


def decode_matrix(HE, matrix):
    try:
        return np.array(list(map(HE.decodeFrac, matrix)))
    except TypeError:
        return np.array([decode_matrix(HE, m) for m in matrix])


def encrypt_matrix(HE, matrix):
    try:
        return np.array(list(map(HE.encryptFrac, matrix)))
    except TypeError:
        return np.array([encrypt_matrix(HE, m) for m in matrix])


def decrypt_matrix(HE, matrix):
    try:
        return np.array(list(map(HE.decryptFrac, matrix)))
    except TypeError:
        return np.array([decrypt_matrix(HE, m) for m in matrix])


# Code for encoded CNN
class ConvolutionalLayer:
    def __init__(self, HE, weights, stride=(1, 1), padding=(0, 0), bias=None):
        self.HE = HE
        self.weights = encode_matrix(HE, weights)
        self.stride = stride
        self.padding = padding
        self.bias = bias
        if bias is not None:
            self.bias = encode_matrix(HE, bias)

    def __call__(self, t):
        t = apply_padding(t, self.padding)
        result = np.array([[np.sum([convolute2d(image_layer, filter_layer, self.stride)
                                    for image_layer, filter_layer in zip(image, _filter)], axis=0)
                            for _filter in self.weights]
                           for image in t])

        if self.bias is not None:
            return np.array([[layer + bias for layer, bias in zip(image, self.bias)] for image in result])
        else:
            return result


def convolute2d(image, filter_matrix, stride):
    x_d = len(image[0])
    y_d = len(image)
    x_f = len(filter_matrix[0])
    y_f = len(filter_matrix)

    y_stride = stride[0]
    x_stride = stride[1]

    x_o = ((x_d - x_f) // x_stride) + 1
    y_o = ((y_d - y_f) // y_stride) + 1

    def get_submatrix(matrix, x, y):
        index_row = y * y_stride
        index_column = x * x_stride
        return matrix[index_row: index_row + y_f, index_column: index_column + x_f]

    return np.array(
        [[np.sum(get_submatrix(image, x, y) * filter_matrix) for x in range(0, x_o)] for y in range(0, y_o)])


def apply_padding(t, padding):
    y_p = padding[0]
    x_p = padding[1]
    zero = t[0][0][y_p+1][x_p+1] - t[0][0][y_p+1][x_p+1]
    return [[np.pad(mat, ((y_p, y_p), (x_p, x_p)), 'constant', constant_values=zero) for mat in layer] for layer in t]


class LinearLayer:
    def __init__(self, HE, weights, bias=None):
        self.HE = HE
        self.weights = encode_matrix(HE, weights)
        self.bias = bias
        if bias is not None:
            self.bias = encode_matrix(HE, bias)

    def __call__(self, t):
        result = np.array([[np.sum(image * row) for row in self.weights] for image in t])
        if self.bias is not None:
            result = np.array([row + self.bias for row in result])
        return result


class ReLULayer:
    def __init__(self, HE):
        self.HE = HE

    def __call__(self, image):
        return relu(self.HE, image)


def relu(HE, image):
    pass
class FlattenLayer:
    def __call__(self, image):
        dimension = image.shape
        return image.reshape(dimension[0], dimension[1] * dimension[2] * dimension[3])


class AveragePoolLayer:
    def __init__(self, HE, kernel_size, stride=(1, 1), padding=(0, 0)):
        self.HE = HE
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, t):
        t = apply_padding(t, self.padding)
        return np.array([[_avg(self.HE, layer, self.kernel_size, self.stride) for layer in image] for image in t])


def _avg(HE, image, kernel_size, stride):
    x_s = stride[1]
    y_s = stride[0]

    x_k = kernel_size[1]
    y_k = kernel_size[0]

    x_d = len(image[0])
    y_d = len(image)

    x_o = ((x_d - x_k) // x_s) + 1
    y_o = ((y_d - y_k) // y_s) + 1

    denominator = HE.encodeFrac(1 / (x_k * y_k))

    def get_submatrix(matrix, x, y):
        index_row = y * y_s
        index_column = x * x_s
        return matrix[index_row: index_row + y_k, index_column: index_column + x_k]

    return [[np.sum(get_submatrix(image, x, y)) * denominator for x in range(0, x_o)] for y in range(0, y_o)]


# We can now define a function to "convert" a PyTorch model to a list of sequential HE-ready-to-be-used layers:
def build_from_pytorch(HE, net):
    # Define builders for every possible layer

    def conv_layer(layer):
        if layer.bias is None:
            bias = None
        else:
            bias = layer.bias.detach().numpy()

        return ConvolutionalLayer(HE, weights=layer.weight.detach().numpy(),
                                  stride=layer.stride,
                                  padding=layer.padding,
                                  bias=bias)

    def lin_layer(layer):
        if layer.bias is None:
            bias = None
        else:
            bias = layer.bias.detach().numpy()
        return LinearLayer(HE, layer.weight.detach().numpy(),
                           bias)

    def avg_pool_layer(layer):
        # This proxy is required because in PyTorch an AvgPool2d can have kernel_size, stride and padding either of
        # type (int, int) or int, unlike in Conv2d
        kernel_size = (layer.kernel_size, layer.kernel_size) if isinstance(layer.kernel_size,
                                                                           int) else layer.kernel_size
        stride = (layer.stride, layer.stride) if isinstance(layer.stride, int) else layer.stride
        padding = (layer.padding, layer.padding) if isinstance(layer.padding, int) else layer.padding

        return AveragePoolLayer(HE, kernel_size, stride, padding)

    def flatten_layer(layer):
        return FlattenLayer()

    def relu_layer(layer):
        return ReLULayer(HE)

    # Maps every PyTorch layer type to the correct builder
    options = {"Conv": conv_layer,
               "Line": lin_layer,
               "Flat": flatten_layer,
               "AvgP": avg_pool_layer,
               "ReLU": relu_layer
               }

    encoded_layers = [options[str(layer)[0:4]](layer) for layer in net]
    return encoded_layers

p = 953983721
m = 4096

HE = Pyfhel()
HE.contextGen(p=p, m=m)
HE.keyGen()
relinKeySize = 3
HE.relinKeyGen(bitCount=5, size=relinKeySize)

model.to("cpu")
model_encoded = build_from_pytorch(HE, model)

def enc_and_process(image):
    plain_image = image
    print(f"plain image is {plain_image}")
    encrypted_image = encrypt_matrix(HE, image.unsqueeze(0).numpy())

    print(f"encrypt_image is {encrypted_image}")

    decrypt_image = decrypt_matrix(HE, encrypted_image)
    print(f"decrypt_image is {decrypt_image}")


    for layer in model_encoded:
        s1 = time.time()
        if str(layer)[10:14] == "ReLU":

            decrypt_image = decrypt_matrix(HE, encrypted_image)

            print(f"the communication complexity between the CP with the client is {sys.getsizeof(decrypt_image) / (1024)} KB")

            tmp_encrypted_image = np.array(list(map(encrypt_decrypt_image, decrypt_image.flatten()))).reshape(decrypt_image.shape)
            print(f"the communication complexity between the client with the SP is {sys.getsizeof(tmp_encrypted_image) / (1024)} KB")

            sta_time = time.time()
            encrypted_image = np.array(list(map(replace_element, tmp_encrypted_image.flatten()))).reshape(tmp_encrypted_image.shape)
            print(f"the communication complexity between the SP with the CP is {sys.getsizeof(encrypted_image) / (1024)} KB")
            tmp_end_time = time.time()
            print(f"the time cost of relu layer is: {tmp_end_time - sta_time}s")
        else:
            encrypted_image = layer(encrypted_image)
            s2 = time.time()
            print(f"the time cost of {str(layer)[10:14]} layer is: {s2 - s1}s")

    result = decrypt_matrix(HE, encrypted_image)
    return result

def replace_element(element):
    if element in relu_HE_image:  # if the current element in relu_HE_image
        return data_dict[element]  # return the value of relu_HE_image
    else:
        return HE.encryptFrac(0.0)  # if the current value not in relu_HE_image, return the HE.encryptFrac(0.0)

filename = "shared_key.txt"
shared_key = util.load_from_file(filename)
iv=b'\xf2\xd6\xbei\xfc\x10;:(\xb6\x92\x7f2W\xbeR'
def encrypt_decrypt_image(element):
    ciphertext = util.aes_encrypt(shared_key, str(element), iv)[0]
    return ciphertext

image, lable = test_set[0]
print(lable)


data_dict = util.read_csv_to_dict(shared_key, HE, iv, "/home/lilvmy/paper-demo/paper7/CryptDNNExperiment/BFVSSE_MNIST/csv")
relu_HE_image = data_dict.data
# print(relu_HE_image)


starting_time = time.time()

res = enc_and_process(image)
print(f"results is {res}")

total_time = time.time() - starting_time

print(f"The total inference time costs is: {total_time}s")
print(res.argmax(1))
