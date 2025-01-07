# Project ID2223, Predicting Amazon Ratings Based on Reviews

Link to app (huggingface space with gradio): https://huggingface.co/spaces/edwoas/ProjectID2223_HF

_If you want to host the model you will have to enter your hopsworks key and also 
get an API key from the Real-Time Amazon API (link below) as environmental variables in the ProjectID2223_HF repository._

## Scope
The aim of this project was to given a number of Amazon reviews for a queried product predict a rating for it based on
the reviews. The final rating of a product is in this way based on a language model's interpretation of the reviews for a
product instead of the users' which I thought was interesting to try out.

## Datasets
The dataset used for training was the Amazon Product Reviews-dataset from Kaggle
(https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews?resource=download) where reviews served as inputs and
their respective rating as labels. This dataset consists of roughly 300 000 reviews, but only around 100 000 were used
in the training phase after data cleaning. During inference, the Real-Time Amazon Data API is used as input to the serverless
model which collects reviews for Amazon products (https://rapidapi.com/letscrape-6bRBa3QguO5/api/real-time-amazon-data).

## Model Selection
The approach used to predict the ratings based on reviews was to pick a pre-trained Language model that produces
rich embeddings and then fine-tune a smaller classifier head which takes the embeddings as input. The RoBERTa backbone
was first considered but due to its size the smaller distillation BERT model was used instead.

## Data Handling and Pre-processing
Upon inspection of the training data it turned out that there were many duplicates. To prevent data leaking
between training, validation, and test sets (which would have lead to skewed statistics for the model's performance) the
duplicates were removed. The number of reviews with a 5-star rating were also over-represented in the dataset so to make the
model less biased towards predicting 5's this class was thinned out. 

After cleaning the data it was tokenized and turned
into CLS-embeddings (processed by the BERT backbone) which then were uploaded to a feature group in hopsworks.
The CLS-embedding serves as a dense vector representation of the entire sequence's meaning or context and thus captures
semantics and sentiment useful for the classifier head to make good predictions. 

By uploading the embeddings it was easy to assure that any customized classifier head later trained would align with the
input to the final model (which also takes CLS-embeddings as input). Apart from aligning training and inference pipelines, 
tokenizing and creating CLS-embeddings are quite time-consuming. By already having the data cleaned and then processed by the
backbone more time could be spent on experimenting with hyperparameters and configurations of the classifier head.

## Training
During training different classifier architectures were experimented with. In the end the one that performed best during
the training/validation phase was a three layer classification head that uses a ReLU activation, batch normalization 
and dropout as regularization. If one chooses to not customize a classifier head, the default one used for distil-bert
with huggingface's AutoModelForSequenceClassification()-method only has an input layer and output layer without any
regularization or activation functions. This one was also tested but performed worse (albeit only slightly, 2-3% in
terms of accuracy).

Since the size of the dataset was diminshed after the data cleaning phase, the model was trained for only eight epochs.
Any further training quickly lead to overfitting (which one already can see is starting to accelerate when trained
only for eight epochs, shown in the evaluation plots later on). Furthermore, different learning rates and optimizers
were experimented with. In the end Adam with a 0.001 learning rate was used. 

## Evaluation

To evaluate the model, the following metrics were adopted:
- Precision
- Recall
- Loss
- Fractions off samples misclassified by 1, ranging up to 5, ratings
- Confusion Matrix

Comments on the plots and scores from the validation set are found above
each plot.

Distribution of classes across datasets:


| Classes | Train set      | Validation set | Test set |
|---------|----------------|----------------|----------|
| 1       | 14591 | 1824           | 1824     |
| 2       | 7950 | 993            | 994      |
| 3       | 10806 | 1351           | 1351     |
| 4       | 20954 | 2620           | 2619     |
| 5       | 40000 | 5000           | 5000     |


### Training and Validation Loss

As can be seen in the first plot,  the training loss quickly goes beneath the valdation loss. As earlier
mentioned the classifier head was for this reason only fine-tuned for eight epochs. The trend was similar despite
trying out different classifier architectures, dropout-rates, batch-sizes, etc. with the difference that this model
got the lowest loss.

![train_val_loss.png](..%2F..%2FDocuments%2FID2223_ProjPlots%2FUnbalanced%20dataset%2Ftrain_val_loss.png)

### Precision and Recall

When plotting the validation precision and recall for each class we can see that the 
class 5 is most stable and achieves the highest score on both metrics, not
surprisingly since it was over-represented in the dataset. Class 1 and 4 improve their precision score,
suggesting that they learn to avoid false positives while class 2 and 3 show no clear
trends.

![recall_per_class.png](..%2F..%2FDocuments%2FID2223_ProjPlots%2FUnbalanced%20dataset%2Frecall_per_class.png)

![precision_per_class.png](..%2F..%2FDocuments%2FID2223_ProjPlots%2FUnbalanced%20dataset%2Fprecision_per_class.png)

### Fraction of samples Off by Ratings over Epochs
Here we can see that the most common mistake the model makes when misclassifying 
is to predict the adjacent class (e.g. if the ground truth of a review was
2 it predicts 1 or 3). Then there is a large gap to the other misclassifications, 
with being off by two rating-points as the second most common one. Three and four
points were most rare. Since the models was fairly accurate in predicting the class 5
(which also was over-represented), this plot should be taken with a grain of salt. 
The confusion matrix below gives more insight into how good each class did.

![fractions_off.png](..%2F..%2FDocuments%2FID2223_ProjPlots%2FUnbalanced%20dataset%2Ffractions_off.png)

### Confusion Matrix
The confusion matrix gives a clear overview of the distribution of how the model performed
on different classes. We see that the weakest classes were 2 and 3 which also had the
least number of samples in them. This could have made the model more prone to predicting
a review as either having a very low or high rating, and we can see that quite many 2's and 3's
were incorrectly predicted as 1's or 5's. Due to limited time, augmentations techniques
were not looked in to which could have balanced the dataset and predictions more.

![confusion_matrix.png](..%2F..%2FDocuments%2FID2223_ProjPlots%2FUnbalanced%20dataset%2Fconfusion_matrix.png)

### Test Set

Test Loss: 0.9266

Test Accuracy: 0.6266

![confusion_matrix_test.png](..%2F..%2FDocuments%2FID2223_ProjPlots%2FUnbalanced%20dataset%2Fconfusion_matrix_test.png)


## UI and Inference Pipeline
The interface of the app was created with Gradio and is hosted as a Huggingface Space. The interface has a query field
where once can enter and search for a product. A dropdown menu will then show the alternatives retrieved by the API.
After choosing a product and confirming the selection, reviews for the five first pages of the product are collected
and inputted to the BERT-backbone + classifier head (which takes CLS-embeddings as input). A rating for each of the
reviews is predicted and the average of these plus the standard deviation is outputted in the UI. Along with the 
average and standard deviation is also a simple 1-5 rating bar outputted which shows the average rating and whiskers of
the deviation to give the user some intuition of the spread.

As it turned out, the API calls became a bottleneck in this project which collects reviews and fetches product search
results rather slowly. This is also the reason why not all product pages were used to compute the average rating since
each page of reviews for a product is a new API-call. A scraping method for getting product reviews was initially tried 
but abandoned since Amazon would detect and prompt log in during scraping. Due to the time limit of the project, the 
cheap and user-friendly API currently used was chosen instead.

