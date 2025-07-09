import numpy as np
from Pyfhel import Pyfhel


def test_bfv():
    p = 953983721
    m = 4096

    HE = Pyfhel()
    HE.contextGen(p=p, m=m)
    HE.keyGen()
    relinKeySize = 3
    HE.relinKeyGen(bitCount=5, size=relinKeySize)

    a = 1.987
    b = 2.345

    c_a = HE.encryptFrac(a)
    c_b = HE.encryptFrac(b)

    print(f"-----------the ciphertext a is: {c_a}")
    print(f"-----------the ciphertext b is: {c_b}")

    c_sum = c_a + c_b
    print(f"-----------the ciphertext sum b is: {c_sum}")

    p_sum = HE.decryptFrac(c_sum)
    print(f"-----------the decrypt sum b is: {p_sum}")




if __name__ == "__main__":
    test_bfv()