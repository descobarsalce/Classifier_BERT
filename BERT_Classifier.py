#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 23:19:23 2023
@author: descobarsalce
"""

import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from transformers import BertTokenizer
from sklearn.metrics import confusion_matrix
from transformers import TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import regex as re
import matplotlib.pyplot as plt

class StringPreprocessing:
    def __init__(self, model='bert-base-uncased'):
        """
        Initialize the ClassifierBERT object.

        Args:
            labeled (DataFrame): Labeled data for training the model.
            text_varname (str): Name of the column containing the text data.
            labels_varname (str): Name of the column containing the labels.
            model (str, optional): Pretrained BERT model to use. Defaults to 'bert-base-uncased'.
        """
        self.tokenizer = BertTokenizer.from_pretrained(model, do_lower_case=True)

    def tokenize(self, df, col_name):
        """
        Tokenize the text data using the BERT tokenizer.

        Args:
            df (DataFrame): DataFrame containing the text data.
            col_name (str): Name of the column containing the text data.
            unlabeled (bool, optional): Whether the data is unlabeled. Defaults to False.

        Returns:
            tuple: Tuple containing the tokenized input IDs, attention masks, and labels (if not unlabeled).
        """
        tokens = self.tokenizer.batch_encode_plus(df[col_name].tolist(),
                                                  add_special_tokens=True,
                                                  max_length=256,
                                                  padding='max_length',
                                                  return_attention_mask=True,
                                                  return_token_type_ids=False,
                                                  return_tensors='tf')
        input_ids = tf.constant(tokens['input_ids'])
        attention_masks = tf.constant(tokens['attention_mask'])
        
        return input_ids, attention_masks
    
    @staticmethod
    def clean_string(string):
        """
        Clean a string by applying various text transformations.

        Args:
            string (str): The input string to clean.

        Returns:
            str: The cleaned string.
        """
        string = str(string)
        string = string.encode("ascii", errors="ignore").decode()
        string = string.lower()
        chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'"]
        rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
        string = re.sub(rx, '', string)
        string = string.replace('&', 'and')
        string = string.replace(',', ' ')
        string = string.replace('-', ' ')
        string = re.sub(' +', ' ', string).strip()
        string = ' ' + string + ' '
        string = re.sub(r'[,-./]|\sBD', r'', string)
        string = re.sub('\s+', ' ', string)
        return string

class ClassifierBERT:
    """
    ClassifierBERT class for training and evaluating a BERT-based text classification model.
    """
    def __init__(self, labeled, text_varname, labels_varname, model='bert-base-uncased'):
        """
        Initialize the ClassifierBERT object.

        Args:
            labeled (DataFrame): Labeled data for training the model.
            text_varname (str): Name of the column containing the text data.
            labels_varname (str): Name of the column containing the labels.
            model (str, optional): Pretrained BERT model to use. Defaults to 'bert-base-uncased'.
        """
        try:
            assert isinstance(labeled, pd.DataFrame), "labeled must be a DataFrame."
            assert isinstance(text_varname, str), "text_varname must be a string."
            assert isinstance(labels_varname, str), "labels_varname must be a string."
        except AssertionError as e:
            print(f"Error during initialization: {e}")
            return
        self.TextPreprocessor = StringPreprocessing()
        self.model = TFBertForSequenceClassification.from_pretrained(model)
        self.labeled_data = labeled
        self.text_varname = text_varname
        self.labels_varname = labels_varname
        self.history = None

    
    def compile_model(self, metric='accuracy'):
        """
        Compile the BERT model with the specified metric.

        Args:
            metric (str, optional): Evaluation metric to use. Defaults to 'accuracy'.
        """
        num_labels = self.labeled_data[self.labels_varname].nunique()
        self.model.layers[-1].activation = None
        self.model.layers[-1].kernel_initializer = tf.keras.initializers.TruncatedNormal(
            stddev=self.model.config.initializer_range)
        self.model.classifier = tf.keras.layers.Dense(num_labels,
                                                      kernel_initializer=self.model.layers[-1].kernel_initializer,
                                                      name='classifier')
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        if metric == 'accuracy':
            metrics = tf.keras.metrics.SparseCategoricalAccuracy(metric)
        else:
            average = 'macro'  # Options: None, 'micro', 'macro', 'weighted'
            metrics = tfa.metrics.F1Score(num_classes=num_labels, average=average)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    def train_tst_val_split(self, train_size=0.8, test_size=0.1, val_size=0.1, random_state=111):
        """
        Split the labeled data into training, testing, and validation sets.

        Args:
            train_size (float, optional): Proportion of data for training. Defaults to 0.8.
            test_size (float, optional): Proportion of data for testing. Defaults to 0.1.
            val_size (float, optional): Proportion of data for validation. Defaults to 0.1.
            random_state (int, optional): Random seed for reproducibility. Defaults to 111.

        Returns:
            tuple: Tuple containing the training, validation, and testing DataFrames.
        """
        if sum([train_size, test_size, val_size]) != 1:
            print('Error: proportions do not add up to 1, please correct.')
        else:
            train_df, val_df = train_test_split(self.labeled_data, test_size=(test_size + val_size),
                                                random_state=random_state)
            val_df, test_df = train_test_split(val_df, test_size=(val_size / (val_size + test_size)),
                                               random_state=random_state)
        return train_df, val_df, test_df

    def train_model(self, epochs=2, batch_size=32, train_size=0.8, test_size=0.1, val_size=0.1,
                    show_confusion_matrix=True, show_roc_auc=True, seed=123, metric='accuracy'):
        """
        Train the BERT model on the labeled data.

        Args:
            epochs (int, optional): Number of training epochs. Defaults to 2.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            train_size (float, optional): Proportion of data for training. Defaults to 0.8.
            test_size (float, optional): Proportion of data for testing. Defaults to 0.1.
            val_size (float, optional): Proportion of data for validation. Defaults to 0.1.
            show_confusion_matrix (bool, optional): Whether to show the confusion matrix. Defaults to True.
            show_roc_auc (bool, optional): Whether to show the ROC curve and AUC score. Defaults to True.
            seed (int, optional): Random seed for reproducibility. Defaults to 123.
            metric (str, optional): Evaluation metric to use. Defaults to 'accuracy'.
        """
        train_df, val_df, test_df = self.train_tst_val_split(train_size=train_size, test_size=test_size,
                                                             val_size=val_size, random_state=seed)
        self.compile_model(metric=metric)
        
        # Transform datasets into correct format:
        datasets = [train_df, val_df, test_df]
        results = []
        for dataset in datasets:
            input_ids, attention_masks = self.TextPreprocessor.tokenize(dataset, self.text_varname)
            labels = tf.constant(dataset[self.labels_varname].tolist())
            results.append((input_ids, attention_masks, labels))        
        input_ids_tr, attention_masks_tr, labels_tr = results[0]
        input_ids_val, attention_masks_val, labels_val = results[1]
        input_ids_test, attention_masks_test, labels_test = results[2]

        self.history = self.model.fit(x=(input_ids_tr, attention_masks_tr),
                                       y=labels_tr,
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       validation_data=((input_ids_val, attention_masks_val), labels_val),
                                       )

        predictions = self.model.predict(input_ids_test)
        predicted_labels = np.argmax(predictions.logits, axis=1)

        self.show_model_performance(predictions, labels_test, predicted_labels,
                                     show_confusion_matrix=show_confusion_matrix, show_roc_auc=show_roc_auc)

    def show_model_performance(self, predictions, labels_test, predicted_labels, show_confusion_matrix=True,
                               show_roc_auc=True):
        """
        Show the performance of the trained model.

        Args:
            predictions (tf.Tensor): Predicted probabilities from the model.
            labels_test (tf.Tensor): True labels from the testing set.
            predicted_labels (np.ndarray): Predicted labels from the model.
            show_confusion_matrix (bool, optional): Whether to show the confusion matrix. Defaults to True.
            show_roc_auc (bool, optional): Whether to show the ROC curve and AUC score. Defaults to True.
        """
        if show_confusion_matrix:
            cm = confusion_matrix(labels_test, predicted_labels)
            print(cm)

        if show_roc_auc:
            y_pred_proba = tf.nn.softmax(predictions.logits, axis=-1)[:, 1]
            fpr, tpr, _ = roc_curve(labels_test, y_pred_proba)
            auc = roc_auc_score(labels_test, y_pred_proba)
            print(auc)

            plt.plot(fpr, tpr)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve (AUC = {:.3f})'.format(auc))
            plt.show()

    def predict_labels(self, unlabeled_df, unlabeled_text_varname, unlabeled=False):
        """
        Predict labels for unlabeled data.

        Args:
            unlabeled_df (DataFrame): Unlabeled data for prediction.
            unlabeled_text_varname (str): Name of the column containing the text data.
            unlabeled (bool, optional): Whether the data is unlabeled. Defaults to False.

        Returns:
            tuple: Tuple containing the predicted probabilities and labels (if not unlabeled).
        """
        input_ids_unlab, attention_masks_unlab, labels_unlab = self.TextPreprocessor.tokenize(unlabeled_df, unlabeled_text_varname)
        predictions_unlab = self.model.predict(input_ids_unlab)
        return predictions_unlab, np.argmax(predictions_unlab.logits, axis=1)

    def show_epoch_evolution(self):
        """
        Show the evolution of training and validation metrics across epochs.
        """
        train_loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        train_acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.plot(train_acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


