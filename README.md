# Sentiment Analysis using Support Vector Machines (SVM)

This code performs sentiment analysis using Support Vector Machines (SVM) on a dataset of sentiment tweets. It uses the TF-IDF vectorization technique to convert the text data into numerical features and trains an SVM classifier to predict the sentiment labels of test data.

## Prerequisites

- Python 3.x
- Google Colab environment
- Required libraries: zipfile, pandas, sklearn

## Setup

1. Mount Google Drive:
   - Make sure you have your dataset zip file and the code file in your Google Drive.
   - Mount your Google Drive using the `drive.mount('/content/drive')` command.

2. Set the paths:
   - Update the `zip_path` variable with the path to your dataset zip file in your Google Drive.
   - Update the `extract_path` variable with the desired path to extract the contents of the zip file in your Google Drive.

3. Extract the dataset:
   - The code will extract the contents of the zip file to the specified `extract_path` using the `zipfile.ZipFile` module.

4. Load and preprocess the dataset:
   - Update the `data_file` variable with the path to your sentiment dataset file in your Google Drive.
   - Split the dataset into training and testing sets using the `train_test_split` function from `sklearn.model_selection`.
   - Create TF-IDF vectors from the text data using the `TfidfVectorizer` from `sklearn.feature_extraction.text`.

5. Train the SVM classifier:
   - Train the SVM classifier using the training vectors and labels with `svm.fit(train_vectors, train_labels)`.

6. Evaluate the model:
   - Predict the sentiment labels for the test data using `svm.predict(test_vectors)`.
   - Calculate the accuracy of the model using `accuracy_score` from `sklearn.metrics`.

## Results

The accuracy of the SVM classifier on the sentiment analysis task is displayed at the end of the code execution.

Feel free to modify the code and experiment with different parameters to improve the performance or adapt it to your specific sentiment analysis task.

