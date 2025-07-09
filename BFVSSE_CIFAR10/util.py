import csv
import os

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad


# redefine a dict to store the value which type is PyCtxt
class PyCtxtDict(dict):
    def __init__(self, shared_key, HE, iv):
        self.HE = HE
        self.shared_key = shared_key
        self.iv = iv
        self.data = {}


    def __setitem__(self, key, value):
        key = aes_encrypt(self.shared_key, str(key), self.iv)[0]
        self.data[key] = self.HE.encryptFrac(value)

    def __getitem__(self, key):
        return self.data[key]


def read_csv_to_dict(shared_key, HE, iv, folder_path):
    data_dict = PyCtxtDict(shared_key, HE, iv)  # create a PyCtxtDict object to store the whole data
    for filename in os.listdir(folder_path):  # iterate through all files in a folder
        if filename.endswith('.csv'):  # if the file end with .csv
            file_path = os.path.join(folder_path, filename)  # obtain the complete route of the file
            with open(file_path, 'r') as file:  # open file
                reader = csv.reader(file)  # create csv reader
                next(reader)  # skip the first row
                for row_idx, row in enumerate(reader):  # iterate each row of the file
                    for col_idx, value_str in enumerate(row):  # iterate each value of the row
                        value = float(value_str)  # transverse value to float value
                        if value <= 0.0:
                            data_dict.__setitem__(value, 0.0)
                        else:
                            data_dict.__setitem__(value, value)
    return data_dict



def aes_encrypt(key, plaintext, iv):
    cipher = AES.new(key, AES.MODE_CBC, iv)
    ciphertext = cipher.encrypt(pad(plaintext.encode(), AES.block_size))
    return ciphertext, cipher.iv



def save_to_file(filename, ciphertext):
    with open(filename, 'wb') as file:
        file.write(ciphertext)
        file.close()

def load_from_file(filename):
    with open(filename, 'rb') as file:
        ciphertext = file.read()  # obtain ciphertext
    return ciphertext
