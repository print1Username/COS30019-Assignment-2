import os
import torch

import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from Matplotlib import Matplotlib


class SwinTransformer:
    def __init__(self, image_size, model_path):
        # Check the device for the AI training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_path = model_path

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def PrepareModel(self, num_classes):
        print("Preparing SwinTransformer Model")

        self.model = models.swin_b(weights='IMAGENET1K_V1')
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

        # Put the model to the device
        self.model.to(self.device)
        print("Model prepared successfully.")

    def LoadData(self, data_path, split, batch_size):
        print(f"Load data from {data_path}")

        # Create the dataset
        dataset = ImageFolder(data_path, transform=self.transform)

        # Save the classes name
        self.class_names = dataset.classes
        print(f"Finding classes: {self.class_names}")
        print(f"Total have {len(dataset)} images.")

        # Split dataset to training and testing
        train_size = int(split * len(dataset))
        test_size = len(dataset) - train_size
        train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

        # Creating a Data Loader - Batch Processing Images
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        print(f"Training data: {train_size} images.")
        print(f"Testing data: {test_size} images.")

        return len(self.class_names)

    def TrainModel(self, epochs):
        print("Start training model...")

        # Setting the optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        chart = Matplotlib(8, 6, 'Training Accuracy', 'Epoch', 'Accuracy')

        # Start training loop
        for epoch in range(epochs):
            print(f"\nTraining {epoch + 1}/{epochs} round...")

            # Train mode
            self.model.train()
            train_correct = 0
            train_total = 0

            # Process each batch of training data
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)  # Putting data on GPU/CPU
                optimizer.zero_grad()  # Clear gradient
                outputs = self.model(images)  # Forward propagation
                loss = criterion(outputs, labels)  # Calculate loss
                loss.backward()
                optimizer.step()  # Backpropagation

                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()  # Calculate accuracy

                # Display progress every batches
                print(f">\tBatch {batch_idx + 1}/{len(self.train_loader)}, Loss: {loss.item():.4f}")

            train_accuracy = 100. * train_correct / train_total  # Calculate training accuracy
            print(f"Training Accuracy:\t{train_accuracy:.2f}%")
            self.TestModel()

            # Add the epoch and accuracy to the chart
            chart.AddData(epoch, train_accuracy)

        chart.CreateChart(3, False, "png", "results")
        self.SaveModel()

    def TestModel(self):
        self.model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():  # No gradient calculation is required during testing
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_accuracy = 100. * test_correct / test_total
        print(f"Testing accuracy: {test_accuracy:.2f}%")

    def SaveModel(self):
        torch.save({
            'model': self.model.state_dict(),
            'classes': self.class_names
        }, 'animal_model.pth')

    def LoadModel(self):
        print(f"Load Model: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Rebuilding the model structure
        num_classes = len(checkpoint['classes'])
        self.PrepareModel(num_classes)

        # Load model weights and class names
        self.model.load_state_dict(checkpoint['model'])
        self.class_names = checkpoint['classes']
        self.model.eval()
        print("Model loading completed")

    def PredictImage(self, image_path):
        # Load Image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs[0], dim=0)
            predicted_class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class_idx].item()

        predicted_animal = self.class_names[predicted_class_idx]
        return predicted_animal, confidence

    def PredictFolder(self, folder_path, result_folder=""):
        print(f"Analyzing folder: {folder_path}")

        # Supported image format
        image_formats = ['.jpg', '.jpeg', '.png']

        results = []
        chart = Matplotlib(8, 8, 'Detecting Animal Chart', 'Animal', '')
        dict = {}
        for filename in os.listdir(folder_path):
            # Check if it is an image file
            if any(filename.lower().endswith(fmt) for fmt in image_formats):
                image_path = os.path.join(folder_path, filename)
                try:
                    animal, confidence = self.PredictImage(image_path)
                    results.append({
                        'filename': filename,
                        'predicted_animal': animal,
                        'confidence': confidence
                    })

                    if animal in dict:
                        num = dict[animal]
                        dict.pop(animal)
                        dict[animal] = num + 1
                    else:
                        dict[animal] = 1

                    print(f"📷 {filename} -> {animal} (Accuracy: {confidence:.2%})")
                except Exception as e:
                    print(f"❌ {filename} Handling failure: {e}")

        chart.AddDictionaryList(dict)
        chart.ConvertDictionaryToList()
        chart.CreateChart(4, save_format="png", save_directory=result_folder)
        return results
