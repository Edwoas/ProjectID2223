import hopsworks
import torch.nn as nn
import http.client
import torch
import os

conn = http.client.HTTPSConnection("real-time-amazon-data.p.rapidapi.com", timeout=12)
hops_api_key = os.environ["HOPSWORKS_API_KEY"]
rapid_api_key = os.environ["RAPID_API_KEY"]

headers = {
    f'x-rapidapi-key': f"{rapid_api_key}",
    'x-rapidapi-host': "real-time-amazon-data.p.rapidapi.com"
}


def get_model_from_hopsworks(project_name, model_name, version=None):
    host = "c.app.hopsworks.ai"
    connection = hopsworks.connection(host=host, api_key_value=hops_api_key)
    project = connection.get_project(project_name)
    model_registry = project.get_model_registry()
    model = model_registry.get_model(model_name, version=version)
    model_dir = model.download()

    return model_dir


def compute_cls_embeddings(texts, roberta_model, tokenizer):
    # Tokenize and process inputs
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Forward pass through the RoBERTa model
    with torch.no_grad():
        outputs = roberta_model(**inputs)

    # Extract the CLS token embedding (first token in each sequence)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]
    return cls_embeddings


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