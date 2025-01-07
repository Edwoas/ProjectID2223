from utility.functions import *
from transformers import AutoTokenizer, AutoModel
import torch
from torch.optim import Adam
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score


# if True: Creates a new feature group and uploads cleaned and processed data to it on hopsworks
new_feature_group = False

# if True: Uses MPS instead of CPU
mps = True

# if True: Generates plots and scores for test set
test_model = True

# if True: creates validation plots and saves model to hopsworks
create_plots_and_model = False

# Decides version on feature group
version = 1

if mps:
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
else:
    device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
model.to(device)
model.eval()

# Connect to Hopsworks
project = hopsworks_connect()
fs = project.get_feature_store()

# Your dataset
if new_feature_group:
    data = loadReviews()[:300000]
    clean_data_feature_group("cls_embeddings_distil", version, data, fs, model, tokenizer)

else:
    set_seed(5)
    # Fetch embeddings and labels from Hopsworks
    embeddings_fg = fs.get_feature_group("cls_embeddings_distil", version=version)
    data = embeddings_fg.read()

    print(data["label"].value_counts())

    train_df, val_df, test_df = dataset_balance(data)
    # Prepare training data
    print("#### TRAIN ####")
    print(train_df["label"].value_counts())
    print("#### VAL ####")
    print(val_df["label"].value_counts())
    print("#### TEST ####")
    print(test_df["label"].value_counts())


    train_dataset, train_loader, shape1 = dataset_ready_for_training(train_df, device)
    val_dataset, val_loader, shape2 = dataset_ready_for_training(val_df, device)
    test_dataset, test_loader, shape3 = dataset_ready_for_training(test_df, device)

    # Instantiate the classifier
    input_dim = shape1  # Dimension of embeddings (e.g., 768 for RoBERTa base)
    num_classes = 5  # Number of unique labels (e.g., 5 for 1-5 rating scale)
    classifier = ClassifierHead(input_dim, num_classes).to(device)

    # Step 4: Define the training loop
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(classifier.parameters(), lr=0.001)
    num_epochs = 8

    train_losses = []
    val_losses = []
    precision_per_class = {1:[], 2:[], 3:[], 4:[], 5:[]}
    recall_per_class = {1:[], 2:[], 3:[], 4:[], 5:[]}
    fractions_off_by_dict = {1:[], 2:[], 3:[], 4:[]}
    confusion_matrix_var = []


    # Training loop
    classifier.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        total_train_samples = 0
        for batch in train_loader:
            embeddings, labels = batch
            embeddings, labels = embeddings.to(device), labels.to(device)

            # Forward pass
            outputs = classifier(embeddings)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(labels)
            total_train_samples += len(labels)


        acc_loss_val = 0
        correct = 0
        total = len(val_dataset)
        all_true = []
        all_pred = []
        off_by = {1: 0, 2: 0, 3: 0, 4: 0}
        classifier.eval()
        with torch.no_grad():
            for batch_val in val_loader:
                embeddings_val, labels_val = batch_val
                embeddings_val, labels_val = embeddings_val.to(device), labels_val.to(device)

                # Forward pass
                outputs_val = classifier(embeddings_val)
                loss_val = criterion(outputs_val, labels_val)
                acc_loss_val += loss_val.item() * len(labels_val)

                _, predicted = torch.max(outputs_val, 1)
                correct += (predicted == labels_val).sum().item()

                errors = torch.abs(predicted - labels_val)

                # Increment counts for each "off by X" category
                for k in off_by.keys():  # k = 1, 2, 3, 4
                    off_by[k] += (errors == k).sum().item()

                # Collect true and predicted labels
                all_true.extend(labels_val.cpu().numpy())
                all_pred.extend(predicted.cpu().numpy())

        # Metrics
        accuracy = correct / total
        epoch_loss /= total_train_samples
        acc_loss_val /= total

        fraction_off_by = {k: v / total for k, v in off_by.items()}

        print(f"\nEpoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Validation Loss: {acc_loss_val:.4f} ,"
              f"Validation Acc: {accuracy:.4f}")

        train_losses.append(epoch_loss)
        val_losses.append(acc_loss_val)

        print(f"Validation Accuracy: {accuracy:.4f}")
        for k, v in fraction_off_by.items():
            print(f"Fraction Off By {k}: {v:.4f}")
            fractions_off_by_dict[k].append(v)

        # Precision and Recall
        precision = precision_score(all_true, all_pred, labels=[0, 1, 2, 3, 4], average=None)
        recall = recall_score(all_true, all_pred, labels=[0, 1, 2, 3, 4], average=None)

        for i, (p, r) in enumerate(zip(precision, recall)):
            print(f"Class {i + 1}: Precision = {p:.4f}, Recall = {r:.4f}")
            precision_per_class[i+1].append(p)
            recall_per_class[i+1].append(r)

        print("________________________\n")

        if epoch == num_epochs - 1:
            confusion_matrix_var.append(all_true)
            confusion_matrix_var.append(all_pred)

        classifier.train()

    if create_plots_and_model == True:
        plots(num_epochs, train_losses, val_losses, precision_per_class,
              recall_per_class, fractions_off_by_dict, confusion_matrix_var)

        upload_model(classifier, "fine_tune_bert_un", project)

    acc_loss_test = 0
    correct_test = 0
    total_test = len(test_dataset)
    all_true_test = []
    all_pred_test = []


    if test_model == True:
        classifier.eval()
        with torch.no_grad():
            for batch_test in test_loader:
                embeddings_test, labels_test = batch_test
                embeddings_test, labels_test = embeddings_test.to(device), labels_test.to(device)

                # Forward pass
                outputs_test = classifier(embeddings_test)
                loss_test = criterion(outputs_test, labels_test)
                acc_loss_test += loss_test.item() * len(labels_test)

                _, predicted = torch.max(outputs_test, 1)
                correct_test += (predicted == labels_test).sum().item()

                # Collect true and predicted labels
                all_true_test.extend(labels_test.cpu().numpy())
                all_pred_test.extend(predicted.cpu().numpy())

        # Metrics
        accuracy = correct_test / total_test
        acc_loss_test /= total_test

        print(f"\nTest Loss: {acc_loss_test:.4f} ,"
              f"Test Acc: {accuracy:.4f}")

        conf_matrix = confusion_matrix(all_true_test, all_pred_test, labels=[0, 1, 2, 3, 4])
        ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[1, 2, 3, 4, 5]).plot()
        plt.title("Confusion Matrix Test")
        plt.savefig("confusion_matrix_test.png")
        plt.close()

        print("________________________\n")


