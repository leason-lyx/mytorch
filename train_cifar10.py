import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from mytensor_autograd import (
    Tensor,
    Conv2d,
    MaxPool2d,
    ReLU,
    Linear,
    Flatten,
    SGD,
    cross_entropy_loss,
    Module,
)


class SimpleCNN(Module):
    def __init__(self, device="cuda"):
        self.conv1 = Conv2d(3, 32, kernel_size=3, stride=1, padding=1, device=device)
        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=1, padding=1, device=device)
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.relu = ReLU()
        self.flatten = Flatten()
        self.fc1 = Linear(64 * 8 * 8, 256, device=device)
        self.fc2 = Linear(256, 10, device=device)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def _to_mytensor(images, device):
    images_np = np.ascontiguousarray(images.numpy(), dtype=np.float32)
    return Tensor(images_np, device=device, requires_grad=False)


def _to_labels(labels):
    return np.ascontiguousarray(labels.numpy(), dtype=np.int32)


def train_one_epoch(model, optimizer, data_loader, device):
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, labels in data_loader:
        x = _to_mytensor(images, device)
        y = _to_labels(labels)
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_value = float(loss.data.numpy().reshape(-1)[0])
        total_loss += loss_value * y.shape[0]
        preds = np.argmax(logits.detach().data.numpy(), axis=1)
        total_correct += int((preds == y).sum())
        total += y.shape[0]
    return total_loss / total, total_correct / total


def evaluate(model, data_loader, device):
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, labels in data_loader:
        x = _to_mytensor(images, device)
        y = _to_labels(labels)
        logits = model(x)
        logits = logits.detach()
        loss = cross_entropy_loss(logits, y)
        loss_value = float(loss.data.numpy().reshape(-1)[0])
        total_loss += loss_value * y.shape[0]
        preds = np.argmax(logits.data.numpy(), axis=1)
        total_correct += int((preds == y).sum())
        total += y.shape[0]
    return total_loss / total, total_correct / total


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for mytensor conv/linear kernels.")

    device = "cuda"
    batch_size = 64
    epochs = 10
    lr = 0.01

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = SimpleCNN(device=device)
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
        )


if __name__ == "__main__":
    main()
