# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.

## Problem Statement and Dataset
Develop an image classification model using transfer learning with the pre-trained VGG19 model.
Load and preprocess a labeled image dataset.
Image classification is a common task in computer vision where the goal is to automatically identify the category or class of an image. For example, given a photo of an animal, the model should correctly identify whether it’s a cat, dog, horse, etc.

1.Use the pre-trained VGG19 model for feature extraction or fine-tuning.

2.Build and train a classification model using the transferred features.

3.Evaluate the model based on accuracy, precision, recall, and other relevant metrics.

## DESIGN STEPS
### STEP 1:
Import required libraries, load the dataset, and define training & testing datasets.

### STEP 2:
Initialize the model, loss function, and optimizer. Use CrossEntropyLoss for multi-class classification and Adam optimizer for efficient training.

### STEP 3:
Train the model using the training dataset with forward and backward propagation.

### STEP 4:
Evaluate the model on the testing dataset to measure accuracy and performance.

### STEP 5:
Make predictions on new data using the trained model.

## PROGRAM

```python
# Load Pretrained Model and Modify for Transfer Learning

from torchvision.models import VGG19_Weights
model=models.vgg19(weights=VGG19_Weights.DEFAULT)

```
```python

# Modify the final fully connected layer to match the dataset classes

num_classes = len(train_dataset.classes)
model.classifier[6] = nn.Linear(4096, num_classes)

```
```python
# Include the Loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[-1].parameters(), lr=0.001)

```
```python

# Train the model

def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')
    # Plot training and validation loss
    print("Name:SWATHI D")
    print("Register Number: 212222230154")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot

![Screenshot 2025-04-16 124045](https://github.com/user-attachments/assets/d75cfed5-852a-4e0d-9bc7-6aa5e469c3c1)



### Confusion Matrix

![Screenshot 2025-04-16 123613](https://github.com/user-attachments/assets/a395cd4a-da38-4f23-a06b-e9f003216c26)



### Classification Report

![Screenshot 2025-04-16 123623](https://github.com/user-attachments/assets/77bd4545-078a-4371-8240-a7cbabc6f730)


### New Sample Prediction

```python
predict_image(model, image_index=55, dataset=test_dataset)
```
![download](https://github.com/user-attachments/assets/7033aeb3-04f1-4d3c-bd9e-0dd79cf47d6e)

```python
predict_image(model, image_index=5, dataset=test_dataset)
```
![download](https://github.com/user-attachments/assets/31dcd4e9-6072-4fc9-9d5a-60657488d113)


## RESULT
Thus, the Transfer Learning for classification using the VGG-19 architecture has been successfully implemented.
