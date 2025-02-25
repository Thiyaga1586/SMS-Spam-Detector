# SMS Spam Detection

## Overview
This project implements an SMS Spam Detection system using various machine learning models. The dataset is processed using text vectorization techniques, and multiple classifiers are trained to detect spam messages with high accuracy.

## Project Structure
- `spam_detection.ipynb` - Jupyter Notebook containing the full implementation of the SMS Spam Detection model.
- `sms-spam.csv` - Dataset containing labeled SMS messages for spam detection.
- `model.pkl` - Pre-trained machine learning model for spam classification.
- `vectorizer.pkl` - Saved TF-IDF vectorizer for text preprocessing.

## Prerequisites
Before running the project, ensure you have the following installed:
- Python (>=3.6)
- Jupyter Notebook
- Required Python libraries:
  ```sh
  pip install numpy pandas scikit-learn matplotlib seaborn wordcloud
  ```

## Features
### 1. Data Preprocessing
- Loads the SMS spam dataset.
- Cleans and preprocesses text data.
- Converts text into numerical format using `TfidfVectorizer`.

### 2. SMS Spam Detection Process
- **Data Collection**: Loads labeled SMS messages from `sms-spam.csv`.
- **Text Preprocessing**: Tokenization, stopword removal, and vectorization using TF-IDF.
- **Model Selection**: Multiple classifiers are trained and evaluated.
- **Model Training**: Training the models on preprocessed SMS data.
- **Evaluation & Validation**: Using accuracy, precision, recall, and confusion matrices.
- **Model Deployment**: Saving trained models and vectorizers for future use.

### 3. Machine Learning Models
The following classifiers are implemented:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Voting Classifier**
- **AdaBoost Classifier**
- **Bagging Classifier**
- **Extra Trees Classifier**

### 4. Model Training & Evaluation
- Splits the dataset into training and testing sets.
- Evaluates models using accuracy, precision, recall, and confusion matrices.
- Visualizes results with seaborn and matplotlib.

### 5. Model Persistence
- Saves the trained model (`model.pkl`) using `pickle`.
- Stores the trained `TfidfVectorizer` (`vectorizer.pkl`) for text transformation in future predictions.

## Installation & Usage
1. Install required dependencies:
   ```sh
   pip install numpy pandas scikit-learn matplotlib seaborn wordcloud
   ```
2. Open the `spam_detection.ipynb` notebook and run all cells.
3. Load the pre-trained model and vectorizer to classify new SMS messages.

## Results & Performance
- The models achieve high accuracy in classifying spam and non-spam messages.
- Word clouds visualize frequently occurring words in spam and ham messages.
- A confusion matrix is used to analyze model predictions.

## Conclusion
This project effectively classifies SMS messages as spam or non-spam using multiple machine learning models. The trained model and vectorizer can be reused for future predictions, making this system robust for real-world applications.

## License
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is licensed under the MIT License.


