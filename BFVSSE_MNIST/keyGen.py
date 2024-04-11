from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend


def generate_ec_key():
    # the data owner's private key sk_DO
    k_d = ec.generate_private_key(ec.SECP256R1(), default_backend())
    sk_DO = k_d


    # the data owner's public key pk_DO
    g_k_d = k_d.public_key()
    pk_DO = g_k_d

    # the user's private key sk_U
    gamma = ec.generate_private_key(ec.SECP256R1(), default_backend())
    sk_U = gamma

    # the user's public key pk_U
    g_gamma = gamma.public_key()
    pk_U = g_gamma

    return sk_DO, pk_DO, sk_U, pk_U


def derive_shared_key(sk_DO, pk_U):
    # share encryption key of relu layer's image
    shared_relu_image_key = sk_DO.exchange(ec.ECDH(), pk_U)
    return shared_relu_image_key


# sk_do, pk_do, sk_U, pk_U = generate_ec_key()
# shared_key = derive_shared_key(sk_do, pk_U)
# print(f"shared_key: {shared_key}")
# file_name = "./shared_key.txt"
# util.save_to_file(file_name, shared_key)
# key = util.load_from_file(file_name)
# print(f"key loaded: {key}")


