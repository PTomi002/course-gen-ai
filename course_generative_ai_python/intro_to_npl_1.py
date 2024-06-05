import pandas as pd
import torch
import torch.nn as nn

from torchviz import make_dot
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as DataSet, DataLoader

# Read content of the csv, dropping NaN values
pd.set_option('display.max_colwidth', 7)
csv = pd.read_csv('course_generative_ai_python/data/Tweets.csv').dropna()
print(csv)

# Define categories, best score is 2
category_id = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

# Map the sentiment table values into categories
csv['class'] = csv['sentiment'].map(category_id)
# print(csv)

# Input involves everything as a data.
# Features are the kind of specific properties of that input.
# Label is the target or the output.
# e.g.: image classification: every image of animals is considered input while the features are the ears, legs, etc the
#   model will prediction will based on
# In our case input is the whole csv, feature is the text we will predict on, label is the sentiment.
features = csv['text'].values
labels = csv['class'].values

# Split the input and expected output into corresponding pairs.
feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.5,
                                                                        random_state=123)
print(
    f"feature_train: {feature_train.shape}, label_train: {label_train.shape}, feature_test: {feature_test.shape}, "
    f"label_test: {label_test.shape}")

# One-Hot Encoding
# "I like this book, it is a good book.
# features/vocabulary:  ['book' 'good' 'is' 'it' 'like' 'this']
# array:                [[2 1 1 1 1 1]]
one_hot_encoder = CountVectorizer()
# fit_transform = learn vocabulary and return term-document matrix
feature_train_one_hot = one_hot_encoder.fit_transform(feature_train)
# transform = already know the vocabulary, return term-document matrix
feature_test_one_hot = one_hot_encoder.transform(feature_test)


# Tensor wrapper class around our data
class SentimentData(DataSet):
    def __init__(self, x, y):
        super().__init__()
        # tensor([[2., 1., 1., 1., 1., 1.]])
        self.x = torch.Tensor(x.toarray())
        # tensor([[2, 1, 1, 1, 1, 1]])
        self.y = torch.Tensor(y).type(torch.LongTensor)
        self.len = len(self.x)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index], self.y[index]


train_dataset = SentimentData(x=feature_train_one_hot, y=label_train)
test_dataset = SentimentData(x=feature_test_one_hot, y=label_test)

# Define the dataloaders for the datasets
BATCH_SIZE = 512

# While training a model, we typically want to pass samples in “minibatches”, reshuffle the data at every epoch to
#   reduce model overfitting.
# Model Overfitting = when the model gives accurate predictions for training data but not for new data
# Happens when: training data size is too small, training data contains large amounts of irrelevant information, etc...
# e.g.: dog recognisation, training data contains only images about dogs in parks, while it is not able to recognise
#   a dog in a room
# Model Underfitting =  when the model performs poorly on the training data
# Happens when: model is unable to capture the relationship between the input and output variables
trainer_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=15000)


# Define the neural network for training and test
# Explain me why we choose this composition of neurons?
class SentimentModel(nn.Module):
    def __init__(self, numOfFeatures, numOfClasses, hidden=10):
        super().__init__()
        self.linear = nn.Linear(numOfFeatures, hidden)
        self.linerTwo = nn.Linear(hidden, numOfClasses)
        self.relu = nn.ReLU()
        self.longSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.linerTwo(x)
        x = self.longSoftmax(x)
        return x


model = SentimentModel(numOfFeatures=feature_train_one_hot.shape[1], numOfClasses=3)

y = model(torch.randn(1, feature_train_one_hot.shape[1]))
dot = make_dot(y.mean(), params=dict(model.named_parameters())).render("rnn_torchviz", format="png")
print(dot)

if __name__ == '__main__':
    pass
