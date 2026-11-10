import torch
import platform

from src.SwinTransformer import SwinTransformer

model_path = "models/animal_model.pth"
train_folder = "../dataset/train"
test_folder = "../dataset/detect"
result_folder = "results"
image_size = 224
batch_size = 8
epochs = 5
split = 0.8


# 80% dataset for training, 20% dataset for validation

def PrintDevice():
    print(f"Python version:\t\t{platform.python_version()}")
    print(f"Pytorch version:\t{torch.__version__}")

    if torch.cuda.is_available() and torch.backends.cudnn.is_available():
        print(f"CUDA version:\t\t{torch.version.cuda}")
        print(f"cuDNN version:\t\t"
              f"{torch.backends.cudnn.version() // 10000}."
              f"{(torch.backends.cudnn.version() % 10000) // 100}."
              f"{(torch.backends.cudnn.version() % 100)}")
    print("\n")


if __name__ == "__main__":
    PrintDevice()

    # Prepared classifier and dataset
    classifier = SwinTransformer(image_size, model_path)

    num_classes = classifier.LoadData(train_folder, split, batch_size)
    classifier.PrepareModel(num_classes)

    # Start training model
    classifier.TrainModel(epochs)

    # Testing dataset
    classifier.LoadModel()
    results = classifier.PredictFolder(test_folder, result_folder)

    print(f"Total analyze {len(results)} images.")
