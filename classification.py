# %%
import copy
import sys

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from barcode_dataset import BarcodeDataset
import torch.nn as nn
import torch.optim as optim
from torchvision import models


def get_transform_functions():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.01, 0.01),
            scale=(0.98, 1.02),
            shear=2
        ),

        # Small, safe geometric changes
        transforms.RandomRotation(degrees=5),
        transforms.RandomHorizontalFlip(p=0.5),

        # Appearance changes (these are very safe for tight crops)
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),

        transforms.ToTensor(),
        transforms.Lambda(lambda img: img + 0.01 * torch.randn_like(img)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, test_transform

# %%
def validate(data_loader, model, device):
    # Set the model to evaluation mode
    model.eval()
    correct = 0
    total = 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        # Apply sigmoid to convert logits to probabilities, then threshold to binary predictions
        predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).to(images.dtype)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    model.train()
    return accuracy, total


def get_model():
    # 1. Load pre-trained ResNet50 model
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    # 3. Replace the final FC with a small binary head
    # ResNet50's penultimate layer outputs 2048 features
    model.fc = nn.Sequential(
        nn.Linear(2048, 64),
        nn.ReLU(),
        nn.Dropout(0.2),   # small dropout, dataset is tiny
        nn.Linear(64, 1)   # 1 logit for BCEWithLogitsLoss
    )

    return model

# %%
def main(root_directory, train_mode = True):
    """
    This is the main function where the program's primary logic resides.
    """

    # 2. Specify your root directory and class folders
    class_folders_list = ['strap', 'reflection']
    train_transform, test_transform = get_transform_functions()

    # 1. Load pre-trained ResNet50 model
    model = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 5. Define custom collate functions for each DataLoader to apply transforms
    def train_collate_fn(batch):
        # batch is a list of (image, label) tuples from the Subset
        images = []
        labels = []
        for img, label in batch:
            images.append(train_transform(img)) # Apply train_transform here
            labels.append(label)

        return torch.stack(images), torch.tensor(labels)

    def test_val_collate_fn(batch):
        images = []
        labels = []
        for img, label in batch:
            images.append(test_transform(img)) # Apply test_transform here
            labels.append(label)

        return torch.stack(images), torch.tensor(labels)
    
    if train_mode:
        # 3. Load the train dataset
        train_dataset = BarcodeDataset(
            root_dir=root_directory,
            class_folders=class_folders_list,
            train=train_mode,
        )

        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size

        # 3. Perform the random split
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        print(f"\nSplitting dataset using torch.utils.data.random_split:")
        print(f"  Training subset size: {len(train_dataset)} samples")
        print(f"  Test/Validation subset size: {len(val_dataset)} samples")

        # 6. Now use these subsets with PyTorch DataLoaders, applying transforms via collate_fn
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=train_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, collate_fn=test_val_collate_fn)
        print(f"Created DataLoader instances with distinct transforms via collate_fn:")
        print(f"  Train DataLoader batch size: {train_loader.batch_size}")
        print(f"  Test/Validation DataLoader batch size: {val_loader.batch_size}")

    else:
        # 3. Load the test dataset
        test_dataset = BarcodeDataset(
            root_dir=root_directory,
            class_folders=class_folders_list,
            train=train_mode,
        )
        print(f"  Test subset size: {len(test_dataset)} samples")
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=test_val_collate_fn)
        checkpoint = torch.load('./best_model.pth')
        model.load_state_dict(checkpoint)
        test_accuracy, test_total_samples = validate(test_loader, model, device)

        print( f"Total test samples = {test_total_samples},"
            f" Test accuracy {test_accuracy:.2f}%,"
            )

        return


    # 2. Define the loss function for binary classification
    criterion = nn.BCEWithLogitsLoss()

    # 3. Define the optimizer, Only optimize the parameters of the newly added classifier layers
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # define a scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,      # halve the LR
        patience=5,      # wait 5 epochs of no improvement
        min_lr=1e-5,
    )

    print(f"Model successfully loaded and modified. Using device: {device}")

    #########################################
    ## this is where the training starts! ###
    #########################################

    epochs = 30 # You might want to adjust the number of epochs
    best_model = None
    best_accuracy = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        model.train() # Set the model to training mode
        running_loss = 0.0
        training_samples = 0
        for _, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            training_samples += len(labels)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss. Squeeze outputs and cast labels to float for BCEWithLogitsLoss
            loss = criterion(outputs.squeeze(), labels.float())

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # validation across the validation set
        val_accuracy, total_val_samples = validate(val_loader, model, device)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = copy.deepcopy(model)
            best_epoch = epoch

        scheduler.step(val_accuracy)

        print(f"Epoch {epoch+1} finished, Training loss {running_loss/training_samples},"
            f"total validation samples = {total_val_samples},"
            f"Val accuracy {val_accuracy:.2f}%,"
            f"LR={optimizer.param_groups[0]['lr']:.6f}"
            )

    print(f"Training completed! best epoch = {best_epoch}")
    torch.save(best_model.state_dict(), './best_model.pth')
    print(f"Model Saved")


if __name__ == "__main__":
    if len(sys.argv) > 1:
            print(f"Arguments provided: {sys.argv[1:]}")

    else:
        print("No arguments provided.")
    
    root_directory = sys.argv[1] # the folder path containing strap/reflection folders
    phase = sys.argv[2] # phase
    main(root_directory, train_mode = (phase == "train"))

