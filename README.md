# NLP-Based-Product-Recommendation-Prediction
## Project Overview

This project focuses on predicting whether a customer would recommend a product based on various features, including review text, age, product category, and other relevant attributes. The dataset consists of women's e-commerce clothing reviews, and the objective is to build a machine learning model that provides accurate recommendations.

## Repository Structure

The repository contains the following files:

```bash
├── ecommerce_reviews_analysis.py  # Main script for data processing & model training
├── reviews.csv                    # Dataset containing clothing reviews
├── README.md                       # Project overview and repository structure
├── notebook.ipynb                  # Jupyter Notebook with step-by-step analysis
```

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
$ git clone https://github.com/your-username/fashion-forward-forecasting.git

# Navigate to the project directory
$ cd fashion-forward-forecasting

# Install dependencies
$ pip install -r requirements.txt

# Run the main script
$ python ecommerce_reviews_analysis.py
```

## Future Improvements

- Experiment with additional NLP techniques such as sentiment scoring and topic modeling.
- Optimize hyperparameter tuning with a broader search space.
- Try other classifiers like XGBoost or deep learning models for better performance.

## Contributors

👤 **Your Name** - Data Scientist

## License

📝 This project is licensed under the **MIT License**.

