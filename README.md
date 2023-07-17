# Classifier_BERT
BERT-based text classification model implemented in TensorFlow and Transformers.

# BERT Classifier

This repository contains a Python script (`ClassifierBERT.py`) that implements a BERT-based text classification model using TensorFlow and Transformers. It allows you to train and evaluate the model on labeled data, make predictions on unlabeled data, and visualize the model's performance.

## Getting Started

To use the BERT Classifier, you'll need to follow these steps:

1. Install the required dependencies: TensorFlow, Transformers, scikit-learn, numpy, matplotlib, and regex.

2. Download a pre-trained BERT model from the Hugging Face model repository. The default model used is `bert-base-uncased`.

3. Prepare your labeled data by providing a DataFrame containing the text data and corresponding labels.

4. Instantiate the `ClassifierBERT` class with the labeled data, text column name, and label column name.

5. Train the model by calling the `train_model()` method, specifying the desired parameters such as epochs, batch size, and evaluation metric.

6. Evaluate the model's performance by calling the `show_model_performance()` method, which shows the confusion matrix and ROC curve.

7. Make predictions on unlabeled data using the `predict_labels()` method, which returns the predicted probabilities and labels.

8. Customize the script as needed for your specific use case, such as modifying the tokenization process or adding additional evaluation metrics.

## Example Usage

Here's an example of how to use the BERT Classifier in a Jupyter notebook:

```python
import pandas as pd
from ClassifierBERT import ClassifierBERT

# Load and preprocess the labeled data
labeled_data = pd.read_csv('labeled_data.csv')
text_varname = 'text'
labels_varname = 'label'

# Instantiate the BERT Classifier
classifier = ClassifierBERT(labeled_data, text_varname, labels_varname)

# Train the model
classifier.train_model(epochs=5, batch_size=32, train_size=0.8, test_size=0.1, val_size=0.1)

# Evaluate the model's performance
classifier.show_model_performance()

# Make predictions on unlabeled data
unlabeled_data = pd.read_csv('unlabeled_data.csv')
unlabeled_text_varname = 'text'
predictions, labels = classifier.predict_labels(unlabeled_data, unlabeled_text_varname)
```

