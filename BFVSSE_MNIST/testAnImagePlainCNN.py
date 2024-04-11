import time

import torch
import torchvision
import torchvision.transforms as transforms


transform = transforms.ToTensor()
test_set = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

image, label = test_set[0]
print(f"image is {image}")
print(f"The label of image is {label}")
print(image.shape)
image = image.to('cuda')

# avoid RuntimeError: mat1 and mat2 shapes cannot be multiplied (12x16 and 192x10) (the second 1 is channel)
test_img = torch.reshape(image, (1, 1, 28, 28))


model = torch.load("/home/cysren/Desktop/lilvmy/PPDLTest/LeNet1_ReLU2.pt")
model.eval()
model.to('cuda')

with torch.no_grad():
    st1 = time.time()
    output = model(test_img)
    st2 = time.time()
    print(f"The test label is {output.argmax(1)}")
    print(f"The total time is {st2 - st1}")