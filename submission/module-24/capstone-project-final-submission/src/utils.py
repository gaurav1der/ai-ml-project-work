"""
Amazon Product Reviews Analysis - Utility Functions
===================================================

This module contains helper functions for data cleaning, feature engineering,
and analysis used in the Amazon Product Reviews capstone project.

Author: Gaurav Goel
Date: January 2025
UC Berkeley ML/AI Program
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import re
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    """
    A class for cleaning and preprocessing Amazon product review data.
    """

    def __init__(self):
        self.label_encoders = {}
        self.scaler = None

    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset using appropriate imputation strategies.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Cleaned dataframe with no missing values
        """
        df_clean = df.copy()

        # Handle missing prices - fill with median by category
        if 'price' in df_clean.columns and df_clean['price'].isnull().sum() > 0:
            df_clean['price'] = df_clean.groupby('category')['price'].transform(
                lambda x: x.fillna(x.median())
            )

        # Handle missing helpful_votes - fill with 0
        if 'helpful_votes' in df_clean.columns and df_clean['helpful_votes'].isnull().sum() > 0:
            df_clean['helpful_votes'] = df_clean['helpful_votes'].fillna(0)

        return df_clean

    def remove_duplicates(self, df):
        """
        Remove duplicate records from the dataset.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Dataframe with duplicates removed
        """
        return df.drop_duplicates().reset_index(drop=True)

    def detect_outliers_iqr(self, data, column):
        """
        Detect outliers using the Interquartile Range (IQR) method.

        Args:
            data (pd.DataFrame): Input dataframe
            column (str): Column name to analyze

        Returns:
            tuple: (outliers_df, lower_bound, upper_bound)
        """
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        return outliers, lower_bound, upper_bound


class FeatureEngineer:
    """
    A class for creating and engineering features from raw data.
    """

    @staticmethod
    def get_sentiment_score(text):
        """
        Calculate sentiment score for a given text using TextBlob.

        Args:
            text (str): Input text

        Returns:
            float: Sentiment polarity score (-1 to 1)
        """
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0

    @staticmethod
    def categorize_sentiment(score):
        """
        Categorize sentiment score into positive, negative, or neutral.

        Args:
            score (float): Sentiment polarity score

        Returns:
            str: Sentiment category
        """
        if score > 0.1:
            return 'Positive'
        elif score < -0.1:
            return 'Negative'
        else:
            return 'Neutral'

    @staticmethod
    def categorize_price(df, price_column='price'):
        """
        Categorize prices into quartile-based categories.

        Args:
            df (pd.DataFrame): Input dataframe
            price_column (str): Name of price column

        Returns:
            pd.Series: Price categories
        """
        quartiles = df[price_column].quantile([0.25, 0.5, 0.75]).values

        def price_category(price):
            if price <= quartiles[0]:
                return 'Low'
            elif price <= quartiles[1]:
                return 'Medium-Low'
            elif price <= quartiles[2]:
                return 'Medium-High'
            else:
                return 'High'

        return df[price_column].apply(price_category)

    def create_text_features(self, df):
        """
        Create text-based features from review text.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Dataframe with added text features
        """
        df_features = df.copy()

        # Basic text features
        df_features['review_length'] = df_features['review_text'].str.len()
        df_features['word_count'] = df_features['review_text'].str.split().str.len()

        # Sentiment analysis
        print("Calculating sentiment scores...")
        df_features['sentiment_score'] = df_features['review_text'].apply(self.get_sentiment_score)
        df_features['sentiment_category'] = df_features['sentiment_score'].apply(self.categorize_sentiment)

        # Text complexity
        df_features['avg_word_length'] = df_features['review_text'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
        )

        return df_features

    def create_temporal_features(self, df, date_column='review_date'):
        """
        Create temporal features from review date.

        Args:
            df (pd.DataFrame): Input dataframe
            date_column (str): Name of date column

        Returns:
            pd.DataFrame: Dataframe with added temporal features
        """
        df_temporal = df.copy()

        df_temporal['review_month'] = df_temporal[date_column].dt.month
        df_temporal['review_year'] = df_temporal[date_column].dt.year
        df_temporal['review_day_of_week'] = df_temporal[date_column].dt.day_name()
        df_temporal['review_quarter'] = df_temporal[date_column].dt.quarter
        df_temporal['is_weekend'] = df_temporal['review_day_of_week'].isin(['Saturday', 'Sunday'])

        return df_temporal

    def create_derived_features(self, df):
        """
        Create derived features combining multiple columns.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Dataframe with added derived features
        """
        df_derived = df.copy()

        # Price category
        df_derived['price_category'] = self.categorize_price(df_derived)

        # Engagement rate (helpful votes per review length)
        df_derived['engagement_rate'] = df_derived['helpful_votes'] / (df_derived['review_length'] + 1)

        # High sentiment flag
        df_derived['high_sentiment'] = df_derived['sentiment_score'] > df_derived['sentiment_score'].quantile(0.75)

        # Long review flag
        df_derived['long_review'] = df_derived['word_count'] > df_derived['word_count'].median()

        # Price-to-rating ratio
        df_derived['price_rating_ratio'] = df_derived['price'] / df_derived['rating']

        # Rating vs category average
        category_avg_rating = df_derived.groupby('category')['rating'].transform('mean')
        df_derived['rating_vs_category_avg'] = df_derived['rating'] - category_avg_rating

        return df_derived


class ModelEvaluator:
    """
    A class for evaluating model performance and providing business insights.
    """

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """
        Calculate comprehensive evaluation metrics.

        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values

        Returns:
            dict: Dictionary of evaluation metrics
        """
        return {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }

    @staticmethod
    def analyze_predictions_by_category(df, y_true, y_pred, category_column='category'):
        """
        Analyze model performance by product category.

        Args:
            df (pd.DataFrame): Input dataframe
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            category_column (str): Category column name

        Returns:
            pd.DataFrame: Performance metrics by category
        """
        results = []

        for category in df[category_column].unique():
            mask = df[category_column] == category
            if mask.sum() > 0:
                cat_true = y_true[mask]
                cat_pred = y_pred[mask]

                metrics = ModelEvaluator.calculate_metrics(cat_true, cat_pred)
                metrics['Category'] = category
                metrics['Count'] = mask.sum()

                results.append(metrics)

        return pd.DataFrame(results)

    @staticmethod
    def get_feature_importance_insights(feature_names, coefficients, top_n=10):
        """
        Generate business insights from feature coefficients.

        Args:
            feature_names (list): List of feature names
            coefficients (array-like): Model coefficients
            top_n (int): Number of top features to analyze

        Returns:
            pd.DataFrame: Feature importance with business insights
        """
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=False)

        return importance_df.head(top_n)


class DataGenerator:
    """
    A class for generating synthetic Amazon product review data for demonstration.
    """

    @staticmethod
    def generate_sample_data(n_samples=5000, random_seed=42):
        """
        Generate synthetic Amazon product review data.

        Args:
            n_samples (int): Number of samples to generate
            random_seed (int): Random seed for reproducibility

        Returns:
            pd.DataFrame: Generated dataset
        """
        np.random.seed(random_seed)

        # Product categories and names
        categories = ['Electronics', 'Books', 'Home & Kitchen', 'Clothing', 'Sports', 'Beauty', 'Toys', 'Health']
        category_weights = [0.25, 0.15, 0.15, 0.12, 0.1, 0.08, 0.08, 0.07]

        product_names = {
            'Electronics': ['Wireless Headphones', 'Smartphone Case', 'Laptop Stand', 'USB Cable', 'Bluetooth Speaker'],
            'Books': ['Python Programming', 'Data Science Guide', 'Machine Learning Handbook', 'AI Fundamentals', 'Statistics Textbook'],
            'Home & Kitchen': ['Coffee Maker', 'Kitchen Knife Set', 'Storage Containers', 'Blender', 'Cookware Set'],
            'Clothing': ['T-Shirt', 'Running Shoes', 'Jeans', 'Jacket', 'Dress Shirt'],
            'Sports': ['Yoga Mat', 'Dumbbells', 'Running Watch', 'Resistance Bands', 'Water Bottle'],
            'Beauty': ['Face Cream', 'Shampoo', 'Makeup Palette', 'Skincare Set', 'Hair Dryer'],
            'Toys': ['LEGO Set', 'Board Game', 'Action Figure', 'Puzzle', 'Building Blocks'],
            'Health': ['Vitamin Supplements', 'Thermometer', 'First Aid Kit', 'Protein Powder', 'Fitness Tracker']
        }

        # Review templates by rating
        review_templates = {
            5: ["Excellent product! Highly recommend.", "Amazing quality, exceeded expectations!", "Perfect, exactly what I needed.", "Outstanding value for money."],
            4: ["Good product, minor issues but overall satisfied.", "Works well, good quality.", "Happy with purchase, would buy again.", "Solid product, meets expectations."],
            3: ["Average product, nothing special.", "Okay quality, could be better.", "Mixed feelings about this purchase.", "Decent but not great."],
            2: ["Disappointed with quality.", "Not as described, issues with functionality.", "Poor build quality, regret buying.", "Would not recommend."],
            1: ["Terrible product, complete waste of money.", "Broke immediately, very poor quality.", "Absolutely awful, do not buy.", "Worst purchase ever made."]
        }

        data = []

        for i in range(n_samples):
            # Select category and product
            category = np.random.choice(categories, p=category_weights)
            product_name = np.random.choice(product_names[category])

            # Generate rating (skewed toward higher ratings)
            rating_probs = [0.05, 0.08, 0.15, 0.32, 0.40]
            rating = np.random.choice([1, 2, 3, 4, 5], p=rating_probs)

            # Generate review text
            base_review = np.random.choice(review_templates[rating])
            if rating >= 4:
                extra_text = [" Great customer service.", " Fast delivery.", " Would definitely buy again.", ""]
            elif rating == 3:
                extra_text = [" Could be improved.", " Average experience.", " Price is fair.", ""]
            else:
                extra_text = [" Very disappointed.", " Poor customer service.", " Will return if possible.", ""]

            review_text = base_review + np.random.choice(extra_text)

            # Generate other features
            helpful_votes = max(0, int(np.random.exponential(2 if rating >= 4 else 0.5)))
            verified_purchase = np.random.choice([True, False], p=[0.85, 0.15])

            # Generate price by category
            price_ranges = {
                'Electronics': (20, 500), 'Books': (10, 50), 'Home & Kitchen': (15, 200),
                'Clothing': (15, 100), 'Sports': (10, 150), 'Beauty': (8, 80),
                'Toys': (10, 100), 'Health': (12, 80)
            }
            min_price, max_price = price_ranges[category]
            price = round(np.random.uniform(min_price, max_price), 2)

            # Generate review date
            days_ago = np.random.randint(0, 730)
            review_date = pd.Timestamp.now() - pd.Timedelta(days=days_ago)

            data.append({
                'product_id': f'{category[:3].upper()}-{i//10:04d}',
                'product_name': product_name,
                'category': category,
                'rating': rating,
                'review_text': review_text,
                'helpful_votes': helpful_votes,
                'verified_purchase': verified_purchase,
                'price': price,
                'review_date': review_date,
                'reviewer_id': f'R{i:05d}'
            })

        df = pd.DataFrame(data)

        # Introduce missing values and duplicates
        missing_price_idx = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
        df.loc[missing_price_idx, 'price'] = np.nan

        missing_votes_idx = np.random.choice(df.index, size=int(0.01 * len(df)), replace=False)
        df.loc[missing_votes_idx, 'helpful_votes'] = np.nan

        # Add duplicates
        duplicate_idx = np.random.choice(df.index, size=int(0.005 * len(df)), replace=False)
        duplicates = df.loc[duplicate_idx].copy()
        df = pd.concat([df, duplicates], ignore_index=True)

        return df


def load_and_prepare_data(file_path=None, generate_sample=True, n_samples=5000):
    """
    Load or generate Amazon product review data and perform initial preparation.

    Args:
        file_path (str, optional): Path to existing data file
        generate_sample (bool): Whether to generate sample data if no file provided
        n_samples (int): Number of samples to generate

    Returns:
        pd.DataFrame: Prepared dataset
    """
    if file_path and pd.io.common.file_exists(file_path):
        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
    elif generate_sample:
        print(f"Generating sample dataset with {n_samples} records")
        generator = DataGenerator()
        df = generator.generate_sample_data(n_samples)
    else:
        raise ValueError("No data file provided and sample generation disabled")

    print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
    return df


def prepare_features_for_modeling(df):
    """
    Prepare all features for machine learning modeling.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        tuple: (X, y, feature_columns, label_encoders)
    """
    # Initialize processors
    cleaner = DataCleaner()
    engineer = FeatureEngineer()

    # Clean data
    df_clean = cleaner.handle_missing_values(df)
    df_clean = cleaner.remove_duplicates(df_clean)

    # Engineer features
    df_features = engineer.create_text_features(df_clean)
    df_features = engineer.create_temporal_features(df_features)
    df_features = engineer.create_derived_features(df_features)

    # Encode categorical variables
    categorical_features = ['category', 'price_category', 'sentiment_category', 'review_day_of_week']
    label_encoders = {}

    for feature in categorical_features:
        if feature in df_features.columns:
            le = LabelEncoder()
            df_features[f'{feature}_encoded'] = le.fit_transform(df_features[feature])
            label_encoders[feature] = le

    # Convert boolean features to numeric
    boolean_features = ['verified_purchase', 'is_weekend', 'high_sentiment', 'long_review']
    for feature in boolean_features:
        if feature in df_features.columns:
            df_features[feature] = df_features[feature].astype(int)

    # Select features for modeling
    feature_columns = [
        'helpful_votes', 'price', 'review_length', 'word_count', 'sentiment_score',
        'review_month', 'review_quarter', 'category_encoded', 'price_category_encoded',
        'sentiment_category_encoded', 'verified_purchase', 'is_weekend', 'high_sentiment',
        'long_review', 'engagement_rate', 'price_rating_ratio', 'rating_vs_category_avg',
        'avg_word_length'
    ]

    # Filter to existing columns
    feature_columns = [col for col in feature_columns if col in df_features.columns]

    X = df_features[feature_columns]
    y = df_features['rating']

    return X, y, feature_columns, label_encoders, df_features


# Export key functions for easy import
__all__ = [
    'DataCleaner',
    'FeatureEngineer',
    'ModelEvaluator',
    'DataGenerator',
    'load_and_prepare_data',
    'prepare_features_for_modeling'
]

if __name__ == "__main__":
    print("Amazon Reviews Analysis Utilities")
    print("=================================")
    print("This module provides utility functions for the capstone project.")
    print("Import the required classes and functions in your analysis notebook.")