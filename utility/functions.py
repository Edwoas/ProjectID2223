import pandas as pd
import os
import hopsworks
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')  # Change the backend to something compatible
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def loadReviews():
    data_df = pd.read_csv("/Users/edwardsjunnesson/PycharmProjects/ProjectID2223/data/Reviews.csv")

    return data_df

def hopsworks_connect():
    with open('/Users/edwardsjunnesson/PycharmProjects/ProjectID2223/data/hopsworks_api_key.txt', 'r') as file:
        os.environ["HOPSWORKS_API_KEY"] = file.read().rstrip()
    project = hopsworks.login()

    return project


def upload_model(model, name, project):
    torch.save(model.state_dict(), f"{name}.pth")
    path = f"/Users/edwardsjunnesson/PycharmProjects/ProjectID2223/models/{name}.pth"
    model_hops = project.get_model_registry()
    model_upload = model_hops.python.create_model(name=name, description="model X")
    model_upload.save(path)


def compute_cls_embeddings(texts, model, tokenizer):
    # Tokenize the input texts
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the [CLS] token embedding (first token in each sequence)
    embeddings = outputs.last_hidden_state[:, 0, :].flatten()  # Shape: [batch_size, hidden_size]
    return embeddings.numpy()

def create_model_feature_group(name, version, data, feature_store, model, tokenizer):
    data["cls_embeddings"] = data["Text"].apply(lambda x: compute_cls_embeddings(x, model, tokenizer))
    data.rename(columns={"Id": "review_id", "Score": "label"}, inplace=True)

    print("Uploading data")
    feature_group = feature_store.get_or_create_feature_group(
        name=name,
        version=version,
        description="Precomputed [CLS] embeddings for reviews",
        primary_key=["review_id"],
        online_enabled=True,
    )

    chunk_size = 5000
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i + chunk_size][["review_id", "cls_embeddings", "label"]]
        print(f"Uploading chunk {i // chunk_size + 1}")
        feature_group.insert(chunk)  # Upload the chunk


def clean_data_feature_group(name, version, data_raw, feature_store, model, tokenizer):
    data_raw.rename(columns={"Id": "review_id", "Score": "label", "Text": "raw_text"}, inplace=True)
    data_filtered = data_raw[data_raw['raw_text'].apply(lambda x: len(x) < 512)]
    data_cleaned = data_filtered[~data_filtered['raw_text'].duplicated(keep='first')]

    data_cleaned["cls_embeddings"] = data_cleaned["raw_text"].apply(lambda x: compute_cls_embeddings(x, model, tokenizer))

    print("Uploading data")
    feature_group = feature_store.get_or_create_feature_group(
        name=name,
        version=version,
        description="Precomputed [CLS] embeddings for reviews",
        primary_key=["review_id"],
        online_enabled=True,
    )

    chunk_size = 5000
    for i in range(0, len(data_cleaned), chunk_size):
        chunk = data_cleaned.iloc[i:i + chunk_size][["review_id", "cls_embeddings", "label"]]
        print(f"Uploading chunk {i // chunk_size + 1}")
        feature_group.insert(chunk)  # Upload the chunk


def dataset_balance(dataframe):
    target_samples_per_class = 50000
    balanced_dfs = []

    for label in dataframe['label'].unique():
        class_subset = dataframe[dataframe['label'] == label]
        # If the class has fewer rows than target_samples_per_class, keep all rows
        n_samples = min(target_samples_per_class, len(class_subset))

        # Subsample or keep all rows if class count is less than the target
        balanced_class_subset = class_subset.sample(n=n_samples, random_state=42, replace=False)
        balanced_dfs.append(balanced_class_subset)

    balanced_df = pd.concat(balanced_dfs)

    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # First, split into train and remaining (validation + test)
    train_df, remaining_df = train_test_split(
        balanced_df, stratify=balanced_df['label'], test_size=(1 - train_ratio), random_state=42
    )

    # Then split the remaining data into validation and test sets
    val_df, test_df = train_test_split(
        remaining_df, stratify=remaining_df['label'], test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42
    )

    return train_df, val_df, test_df


def dataset_ready_for_training(dataframe, device):
    # Prepare training data
    samples = torch.tensor(np.vstack(dataframe["cls_embeddings"].to_numpy()), dtype=torch.float32).to(device)
    labels = torch.tensor(dataframe["label"].to_numpy(), dtype=torch.long).to(device)  # Labels
    labels = labels - 1

    # Create dataset and dataloader
    dataset = ReviewDataset(samples, labels)
    dataloader = DataLoader(dataset, batch_size=200, shuffle=True) # 64

    return dataset, dataloader, samples.shape[1]


def set_seed(seed):
    torch.manual_seed(seed)  # Set seed for PyTorch
    np.random.seed(seed)  # Set seed for NumPy
    torch.use_deterministic_algorithms(True)  # Ensures deterministic operations


def compute_prediction_diff(predicted, labels_val, off_by):
    # Compute how off the predictions are
    errors = torch.abs(predicted - labels_val)

    # Count how many predictions are off by 1, 2, 3, and 4
    off_by[1] = (errors == 1).sum().item()
    off_by[2] = (errors == 2).sum().item()
    off_by[3] = (errors == 3).sum().item()
    off_by[4] = (errors == 4).sum().item()

    return off_by


def plots(epochs, train_losses, val_losses,
          precision_per_class, recall_per_class, fractions_off_by, conf):

    all_true, all_pred = conf[0], conf[1]
    epochs = np.arange(1, epochs+1)
    # Plot 1: Validation and Training Loss over Epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.savefig("train_val_loss.png")
    plt.close()

    # Plot 2: Precision per Class over Epochs
    plt.figure(figsize=(10, 6))
    for cls, values in precision_per_class.items():
        plt.plot(epochs, values, label=f"Class {cls}")
    plt.xlabel("Epochs")
    plt.ylabel("Precision")
    plt.title("Precision per Class over Epochs")
    plt.legend()
    plt.savefig("precision_per_class.png")
    plt.close()

    # Plot 3: Recall per Class over Epochs
    plt.figure(figsize=(10, 6))
    for cls, values in recall_per_class.items():
        plt.plot(epochs, values, label=f"Class {cls}")
    plt.xlabel("Epochs")
    plt.ylabel("Recall")
    plt.title("Recall per Class over Epochs")
    plt.legend()
    plt.savefig("recall_per_class.png")
    plt.close()

    # Plot 4: Fractions Off by Ratings over Epochs
    plt.figure(figsize=(10, 6))
    for off, values in fractions_off_by.items():
        plt.plot(epochs, values, label=f"Off by {off} Rating")
    plt.xlabel("Epochs")
    plt.ylabel("Fraction of Samples")
    plt.title("Fractions of Samples Off by Ratings over Epochs")
    plt.legend()
    plt.savefig("fractions_off.png")
    plt.close()

    conf_matrix = confusion_matrix(all_true, all_pred, labels=[0, 1, 2, 3, 4])
    ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[1, 2, 3, 4, 5]).plot()
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()


class ReviewDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class ClassifierHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassifierHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


class ClassifierHead2(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassifierHead2, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Dropout(0.1),
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


class ClassifierDistil(nn.Module):
    def __init__(self, input_dim=768, num_classes=5):
        super(ClassifierDistil, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Dropout(0.1),
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x):
        return self.fc(x)