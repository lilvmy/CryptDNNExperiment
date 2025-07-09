import warnings

import numpy as np
import onnx
import torch
import time

# Concrete-Numpy and Concrete-ML
from concrete.numpy.compilation import Configuration

# The QAT model
from zama.zaTFHEMnistModel import MNISTQATModel
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from tqdm import tqdm

from concrete.ml.torch.compile import compile_torch_model


def train(model, device, train_loader, optimizer, epoch, criterion):
    """Train the model."""

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data).squeeze()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 500 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx,
                    len(train_loader.dataset) // len(data),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, device, test_loader, criterion):
    """Test the model."""

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        f"Test set: Average loss: {test_loss:.4f}, "
        "Accuracy: "
        f"{correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset):.0f}%)"
    )

    return test_loss


def manage_dataset(train_kwargs, test_kwargs):
    """Get training and test parts of MNIST dataset."""

    # Pre-transform
    class ReshapeTransform:
        def __init__(self, new_size):
            self.new_size = new_size

        def __call__(self, img):
            return torch.reshape(img, self.new_size)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ReshapeTransform((28 * 28,)),
        ]
    )

    # Manage datasets
    dataset1 = datasets.MNIST("./data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("./data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader


def compile_and_test(
    model,
    use_virtual_lib,
    np_inputs,
    quantization_bits,
    test_data,
    test_data_length,
    test_target,
    show_mlir,
    current_index,
):
    # Compile the QAT model and test
    configuration = Configuration(
        enable_unsafe_features=True,  # This is for our tests only, never use that in prod
        use_insecure_key_cache=True,  # This is for our tests only, never use that in prod
        insecure_key_cache_location="/tmp/keycache",
        p_error=None,  # To avoid any confusion: we are always using kwarg p_error
        # global_p_error=None,  # To avoid any confusion: we are always using kwarg global_p_error
    )

    if use_virtual_lib:
        print(f"\n{current_index}. Compiling with the Virtual Library")
    else:
        print(f"\n{current_index}. Compiling in FHE")

    q_module = compile_torch_model(
        model,
        np_inputs,
        import_qat=True,
        configuration=configuration,
        # Note that in CML 0.4, fixing net_inputs and net_outputs to 5 will no more be needed,
        # since it will be the default
        n_bits={
            "net_inputs": 5,
            "op_inputs": quantization_bits,
            "op_weights": quantization_bits,
            "net_outputs": 5,
        },
        use_virtual_lib=use_virtual_lib,
        show_mlir=show_mlir,
    )

    # Check max bit width
    max_bit_width = q_module.forward_fhe.graph.maximum_integer_bit_width()

    if max_bit_width > 8:
        raise Exception(
            f"Too large bit-width ({max_bit_width}): training this network resulted in an "
            "accumulator size that is too large. Possible solutions are:"
            "    - this network should, on average, have 8bit accumulators. In your case an unlucky"
            f"initialization resulted in {max_bit_width} accumulators. You can try to train the "
            "network again"
            "    - reduce the sparsity to reduce the number of active neuron connexions"
            "    - if the weight and activation bitwidth is more than 2, you can try to reduce one "
            "or both to a lower value"
        )

    # Check the accuracy
    if use_virtual_lib:
        print(
            f"\n{current_index + 1}. Checking accuracy with the Virtual Library "
            f"(length {test_data_length})"
        )
    else:
        print(f"\n{current_index + 1}. Checking accuracy in FHE (length {test_data_length})")

    # Key generation
    if not use_virtual_lib:
        q_module.forward_fhe.keygen()

    correct_fhe = 0
    idx = 0

    # Reduce the test data, since very slow in FHE
    reduced_test_data = test_data[0:test_data_length, :]

    for idx, im in enumerate(tqdm(reduced_test_data)):
        target_np = test_target[idx]
        q_data = q_module.quantize_input(im)
        q_data = np.expand_dims(q_data, 0).astype(np.int64)

        prediction = q_module.forward_fhe.encrypt_run_decrypt(q_data)
        prediction = q_module.dequantize_output(prediction)

        if np.argmax(prediction) == target_np:
            correct_fhe += 1

    # Final accuracy
    return correct_fhe, reduced_test_data.shape[0], max_bit_width


def main():
    """Main."""

    warnings.filterwarnings("ignore")

    np.set_printoptions(threshold=1024)
    criterion = nn.CrossEntropyLoss()
    torch.autograd.set_detect_anomaly(True)

    # Options: the most important ones
    epochs = 1
    sparsity = 4
    quantization_bits = 2
    do_test_in_fhe = True
    do_training = True
    show_mlir = False

    # Options: can be changed
    lr = 0.02
    gamma = 0.33
    test_data_length_reduced = 1  # This is notably the length of the computation in FHE
    test_data_length_full = 10000

    # Options: no real reason to change
    batch_size = 32
    test_batch_size = 32
    use_cuda_if_available = True
    seed = None

    # Seeding
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)

    print(f"\nUsing seed {seed}\n")
    torch.manual_seed(seed)

    # Training and test arguments
    train_kwargs = {"batch_size": batch_size}
    test_kwargs = {"batch_size": test_batch_size}

    # Cuda management
    use_cuda = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Manage dataset
    train_loader, test_loader = manage_dataset(train_kwargs, test_kwargs)

    # Model definition
    model = MNISTQATModel(quantization_bits, quantization_bits)
    model = model.to(device)
    model.prune(sparsity, True)

    # Start
    print(
        f"Performing MNIST task with {quantization_bits}-bits in quantization and a "
        f"sparsity of {sparsity}"
    )

    if do_training:
        print("\n1. Training")
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
        test_loss = 1e10

        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer, epoch, criterion)
            cur_loss = test(model, device, test_loader, criterion)

            scheduler.step()

        model.prune(sparsity, False)

        # Export to ONNX
        print("\n2. Exporting to ONNX")
        input = torch.rand((1, 784)).to(device)
        torch.onnx.export(model, input, "mnist.qat.onnx", opset_version=14)

    else:
        print("\n1. Loading pre-trained model")

    # Reload the model
    model = onnx.load("mnist.qat.onnx")

    # Test in FHE
    st1 = time.time()
    if do_test_in_fhe:

        list_inputs = []

        for inputs in test_loader:
            inputs_var, targets = inputs
            list_inputs.append(inputs_var.detach().cpu().numpy())

        np_inputs = np.concatenate(list_inputs, axis=0)

        test_data = np.zeros((len(test_loader.dataset), 784))
        test_target = np.zeros((len(test_loader.dataset), 1))
        idx = 0

        for data, target in tqdm(test_loader):
            target_np = target.cpu().numpy()
            for idx_batch, im in enumerate(data.numpy()):
                test_data[idx] = im
                test_target[idx] = target_np[idx_batch]
                idx += 1

        accuracy = {}
        current_index = 3

        for use_virtual_lib, use_full_dataset in [(True, True), (True, False), (False, False)]:
            test_data_length = (
                test_data_length_full if use_full_dataset else test_data_length_reduced
            )

            correct_fhe, test_data_shape_0, max_bit_width = compile_and_test(
                model,
                use_virtual_lib,
                np_inputs,
                quantization_bits,
                test_data,
                test_data_length,
                test_target,
                show_mlir,
                current_index,
            )

            current_index += 2
            current_accuracy = correct_fhe / test_data_shape_0

            print(
                f"Accuracy in {'VL' if use_virtual_lib else 'FHE'} with length {test_data_length}: "
                f"{correct_fhe}/{test_data_shape_0} = "
                f"{current_accuracy:.4f}, in {max_bit_width} bits"
            )

            if (use_virtual_lib, use_full_dataset) == (True, True):
                accuracy["VL full"] = current_accuracy
            elif (use_virtual_lib, use_full_dataset) == (True, False):
                accuracy["VL short"] = current_accuracy
            else:
                assert (use_virtual_lib, use_full_dataset) == (False, False)
                accuracy["FHE short"] = current_accuracy

    # Check that accuracy in FHE and in VL is the same
    assert (
        accuracy["VL short"] == accuracy["FHE short"]
    ), "Error, accuracy in VL and in FHE are not the same"

    # Check that accuracy is random-looking
    assert accuracy["VL full"] > 0.8, "Error, accuracy is too bad"
    st2 = time.time()
    print(f"an image validation is {st2 - st1}")

    print()


if __name__ == "__main__":
    main()