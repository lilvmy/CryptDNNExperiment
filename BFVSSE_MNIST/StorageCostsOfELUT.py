from Pyfhel import Pyfhel

from BFVSSE_MNIST import util
import time

import sys

p = 953983721
m = 4096

HE = Pyfhel()
HE.contextGen(p=p, m=m)
HE.keyGen()
relinKeySize = 3
HE.relinKeyGen(bitCount=5, size=relinKeySize)
filename = "shared_key.txt"
iv=b'\xf2\xd6\xbei\xfc\x10;:(\xb6\x92\x7f2W\xbeR'
shared_key = util.load_from_file(filename)



start_time = time.time()
data_dict = util.read_csv_to_dict(shared_key, HE, iv, "/home/cysren/Desktop/lilvmy/PPDLTest/BFVSSE_MNIST/csv")
end_time = time.time()
relu_HE_image = data_dict.data
print(relu_HE_image)
print(f"total time: {end_time - start_time}")
print(f"the storage is {sys.getsizeof(relu_HE_image) / (1024**2)} MB")