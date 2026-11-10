import torch
import platform
from datetime import datetime
from pathlib import Path

from src.SwinTransformer import SwinTransformer

MODEL_PATH = "models/animal_model.pth"
TRAIN_FOLDER = "dataset/train"
TEST_FOLDER = "dataset/detect"
RESULT_FOLDER = "results"
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 5
SPLIT = 0.8


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


def CheckDataset():
    for folder in (TRAIN_FOLDER, TEST_FOLDER):
        if not Path(folder).is_dir():
            print(f"Dataset folder not found: {folder}")
            return False

    print("Dataset folders found.")
    return True


def CheckModels():
    model_file = Path(MODEL_PATH)
    if not model_file.is_file():
        print(f"Model file not found: {MODEL_PATH}")
        return False

    model_info = model_file.stat()
    created_at = datetime.fromtimestamp(model_info.st_ctime)
    print(f"Model file found: {MODEL_PATH}")
    print(f"Model size: {model_info.st_size / (1024 * 1024):.2f} MB")
    print(f"Created: {created_at:%Y-%m-%d %H:%M:%S}")
    return True


if __name__ == "__main__":
    PrintDevice()
    CheckDataset()
    CheckModels()
    '''
    # Prepared classifier and dataset
    classifier = SwinTransformer(IMAGE_SIZE, MODEL_PATH)

    num_classes = classifier.LoadData(TRAIN_FOLDER, SPLIT, BATCH_SIZE)
    classifier.PrepareModel(num_classes)

    # Start training model
    classifier.TrainModel(EPOCHS)

    # Testing dataset
    classifier.LoadModel()
    results = classifier.PredictFolder(TEST_FOLDER, RESULT_FOLDER)

    print(f"Total analyze {len(results)} images.")
    '''
