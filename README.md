# NLP-Based-Product-Recommendation-Prediction
## Project Overview

This project focuses on predicting whether a customer would recommend a product based on various features, including review text, age, product category, and other relevant attributes. The dataset consists of women's e-commerce clothing reviews, and the objective is to build a machine learning model that provides accurate recommendations.

## Repository Structure

The repository contains the following files:

```bash
├── NLP-Based-Product-Recommendation-Prediction.py  # Main script for data processing & model training
├── reviews.csv                                     # Dataset containing clothing reviews
├── README.md                                       # Project overview and repository structure
├── Fashion Forward Forecasting.ipynb              # Jupyter Notebook with step-by-step analysis
```

## Dataset Information

The dataset consists of 18,442 rows and the following columns:

| Column Name               | Description |
|---------------------------|-------------|
| `Clothing ID`             | Unique identifier for each clothing item |
| `Age`                     | Age of the reviewer |
| `Title`                   | Short title of the review |
| `Review Text`             | Full customer review text |
| `Positive Feedback Count`  | Number of users who found the review helpful |
| `Division Name`           | High-level division of the product |
| `Department Name`         | Department category for the product |
| `Class Name`              | Specific class of the product |
| `Recommended IND`         | Target variable (1 if recommended, 0 if not) |

## Data Processing

- Missing values in the `Title` and `Review Text` columns are replaced with empty strings.
- The `Title` and `Review Text` columns are merged into `Full_Review` for text analysis.
- The text is processed using SpaCy, including lemmatization and stopword removal.
- Additional numerical features such as review length and sentiment indicators are extracted.

## Machine Learning Pipeline

- A `ColumnTransformer` is used to preprocess numeric, categorical, and text features.
- `SMOTE` is applied to handle class imbalance.
- A `RandomForestClassifier` is trained to make predictions.
- `GridSearchCV` is used for hyperparameter tuning.

## Model Performance

### Initial Model Performance:

| Metric        | Score  |
|--------------|--------|
| Accuracy     | **88.02%** |
| Precision    | **90.54%** |
| Recall       | **95.29%** |
| F1 Score     | **92.85%** |

### Tuned Model Performance:

| Metric        | Score  |
|--------------|--------|
| Accuracy     | **88.08%** |
| Precision    | **90.54%** |
| Recall       | **95.35%** |
| F1 Score     | **92.88%** |

The model shows strong recall for the recommended class while maintaining balanced performance across other metrics.

## How to Run

```bash
# Clone the repository
$ git clone https://github.com/your-username/NLP-Based-Product-Recommendation-Prediction.git

# Navigate to the project directory
$ cd NLP-Based-Product-Recommendation-Prediction

# Install dependencies
$ pip install -r requirements.txt

# Run the main script
$ python NLP-Based-Product-Recommendation-Prediction.py
```

## Future Improvements

- Experiment with additional NLP techniques such as sentiment scoring and topic modeling.
- Optimize hyperparameter tuning with a broader search space.
- Try other classifiers like XGBoost or deep learning models for better performance.
