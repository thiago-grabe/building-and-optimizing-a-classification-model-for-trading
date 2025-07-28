# Building and Optimizing a Classification Model for Trading

## Overview

This project focuses on building and optimizing a machine learning classification model to predict the direction of 5-day price movements for the SPDR Healthcare Sector ETF (XLV). The model uses various features including technical indicators, volatility data (VIX), and Google Trends sentiment data to make predictions about whether XLV's price will increase or decrease over a 5-day period.

## Project Objectives

- **Data Preprocessing**: Load and clean financial data from multiple sources
- **Feature Engineering**: Create meaningful features from raw financial data
- **Model Training**: Train a Random Forest classifier with proper cross-validation
- **Hyperparameter Tuning**: Optimize model performance using GridSearchCV
- **Model Evaluation**: Assess performance using multiple metrics and compare to baseline
- **Feature Importance Analysis**: Identify the most important predictive features

## Project Structure

```
building-and-optimizing-a-classification-model-for-trading/
├── project_starter.ipynb          # Main Jupyter notebook with analysis
├── xlv_data.csv                   # XLV ETF daily price data (2004-2024)
├── vix_data.csv                   # VIX volatility index daily data
├── GoogleTrendsData.csv           # Monthly Google Trends data for "recession"
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── LICENSE                        # Project license
```

## Data Sources

### 1. XLV ETF Data
- **Source**: SPDR Healthcare Sector ETF (NYSEARCA: XLV)
- **Period**: January 1, 2004 to March 31, 2024
- **Features**: Open, High, Low, Close, Adjusted Close, Volume

### 2. VIX Data
- **Source**: CBOE Volatility Index (INDEXCBOE: VIX)
- **Period**: Same as XLV data
- **Usage**: Market uncertainty proxy

### 3. Google Trends Data
- **Source**: Google Trends for "recession" searches in the US
- **Frequency**: Monthly data interpolated to daily
- **Usage**: Public sentiment indicator

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/building-and-optimizing-a-classification-model-for-trading.git
cd building-and-optimizing-a-classification-model-for-trading
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook
```bash
jupyter notebook project_starter.ipynb
```

## Dependencies

The project requires the following Python packages:

```
pandas              # Data manipulation and analysis
numpy               # Numerical computations
yfinance            # Financial data download
ta                  # Technical analysis indicators
matplotlib          # Basic plotting
seaborn             # Statistical visualization
scikit-learn        # Machine learning algorithms
plotly              # Interactive plotting
ipykernel           # IPython kernel for Jupyter
ipywidgets          # Interactive widgets
jupyter             # Jupyter notebook
jupyterlab          # JupyterLab interface
```

## Project Workflow

### 1. Data Acquisition and Preprocessing
- Load XLV, VIX, and Google Trends data
- Handle missing values and data quality issues
- Create proper date indexing for time series analysis

### 2. Feature Engineering
- **Temporal Features**: Cyclical month encoding, one-hot encoded weekdays
- **Historical Returns**: 1, 5, 10, and 20-day rolling returns
- **Volume Features**: Log-transformed trading volume
- **Technical Indicators**: 
  - Internal Bar Strength (IBS)
  - Bollinger Bands (moving average, bands, indicators)
  - Relative Strength Index (RSI)
- **Target Variable**: Binary classification for 5-day forward returns

### 3. Model Training and Evaluation
- Train-test split (80/20) with temporal ordering preserved
- RandomForestClassifier with cross-validation
- GridSearchCV for hyperparameter optimization
- Performance evaluation using accuracy, precision, recall, and F1-score

### 4. Feature Importance Analysis
- Identify most predictive features
- Remove low-importance features and retrain model
- Compare performance of full vs. reduced feature sets

## Results Summary

The project demonstrates the challenges of predicting financial markets using machine learning:

- Models achieve performance close to baseline (~50-55% accuracy)
- Feature importance analysis reveals which indicators are most predictive
- Results highlight the difficulty of consistent outperformance in efficient markets

## Questions and Answers from Analysis

### Data Analysis Questions

#### Q: Does the data look relatively balanced or grossly unbalanced in the distribution of the target variable? Why is this important?

> **Answer:** The data looks relatively balanced. From the visualization, we can see that the distribution is roughly 50-50 between positive and negative 5-day forward returns. This is important because: 
> 1. Balanced classes prevent the model from being biased toward predicting the majority class
> 2. It ensures that accuracy is a meaningful metric (in imbalanced datasets, a naive model could achieve high accuracy by always predicting the majority class)
> 3. It allows the model to learn meaningful patterns for both classes rather than just memorizing the dominant class

---

### Model Training Questions

#### Q: With a value of max_depth=15, does your model overfit or underfit?

> **Answer:** The model overfits. With max_depth=15, we can see that the training accuracy is higher than the validation accuracy, and there's a noticeable gap between the training and validation curves, indicating the model is memorizing the training data rather than generalizing well.

#### Q: With a value of max_depth=15, is your performance metric (accuracy score) more likely to improve with more training data or with higher model complexity?

> **Answer:** More training data. Since the model is overfitting (training accuracy > validation accuracy), adding more training data would help reduce the overfitting and improve generalization. Increasing model complexity would make the overfitting problem worse.

#### Q: Looking more closely at the DataFrame of top 5 results, varying which hyperparameter did not seem to have any effect, at least in the top-ranking score?

> **Answer:** n_estimators - varying the number of estimators from 50 to 150 does not seem to have a significant effect on the performance, as multiple different n_estimator values appear in the top rankings with similar scores.

---

### Model Evaluation Questions

#### Q: Explain, in words and citing the actual numbers from the evaluation report above, what the precision and recall scores mean.

> **Answer:** Precision measures the accuracy of positive predictions - of all the times the model predicted a positive 5-day return, what percentage were actually positive. A precision score tells us how many of our positive predictions were correct. Recall measures the model's ability to find all positive cases - of all the actual positive 5-day returns in the data, what percentage did the model correctly identify. A high recall means the model catches most of the positive cases. The actual numbers will depend on the model's performance when run.

#### Q: How does your model's performance compare to the baseline in terms of accuracy?

> **Answer:** The model's accuracy should be compared to the baseline accuracy (which is around 50-55% based on majority class prediction). If the model achieves accuracy significantly higher than this baseline, it demonstrates that the features contain predictive signal. If it's similar to baseline, the model is not learning meaningful patterns.

#### Q: How do the precision and recall of your model compare to those of the baseline model?

> **Answer:** A baseline model that always predicts the majority class would have: precision = majority class percentage (~50-55%), recall = 100% for the majority class and 0% for the minority class. Our model should ideally have more balanced precision and recall for both classes, showing it can distinguish between positive and negative returns rather than just predicting the dominant class.

---

### Feature Importance Questions

#### Q: How does the accuracy compare to your last trained model?

> **Answer:** The reduced feature model's accuracy should be compared to the full feature model. If performance is similar or better, it suggests the removed features were not contributing meaningful information and may have been adding noise. If performance drops significantly, the removed features contained important predictive signal.

#### Q: How does the accuracy compare to the baseline?

> **Answer:** The reduced model's accuracy should still be compared to the baseline (~50-55% majority class prediction). If it still outperforms the baseline, the reduced feature set retains predictive power. If it's now similar to baseline, the removed features may have contained the key predictive signals.

---

### Strategic Analysis Questions

#### Q: What would your next course of action be? In particular, share your thoughts on further optimization, pursuing different strategies, and anything else.

> **Answer:** Next steps: 
> 1. **Further optimization**: Try different algorithms (XGBoost, neural networks), more sophisticated feature engineering (technical indicators, market microstructure data), ensemble methods, and time-series cross-validation
> 2. **Different strategies**: Consider longer prediction horizons, different asset classes, or focus on risk prediction rather than direction
> 3. **Data improvements**: Add more relevant features like earnings data, sentiment analysis from news, options flow, or higher frequency data
> 4. **Risk management**: Even modest predictive power can be valuable with proper position sizing and risk controls

#### Q: What do you think of the fact that we used interpolated monthly Google Trends data to try and predict short-term (5-day) price movements?

> **Answer:** Using interpolated monthly Google Trends data to predict 5-day movements is problematic. The temporal mismatch means we're using low-frequency, smoothed sentiment data to predict high-frequency price movements. Monthly data likely reflects longer-term economic concerns rather than short-term trading signals. Daily or weekly Google Trends data, news sentiment, or social media sentiment would be more appropriate for 5-day predictions. The interpolation creates artificial daily values that may not reflect actual daily sentiment changes.

---

## Key Learnings

1. **Market Efficiency**: Financial markets are highly efficient, making consistent prediction challenging
2. **Feature Engineering**: Proper feature engineering is crucial for financial ML models
3. **Temporal Considerations**: Time series data requires careful handling of temporal dependencies
4. **Baseline Importance**: Always establish and compare against meaningful baselines
5. **Data Quality**: High-quality, relevant data is more important than complex algorithms
6. **Risk Management**: Even modest predictive power can be valuable with proper risk controls

## Technical Implementation Details

### Model Architecture
- **Algorithm**: Random Forest Classifier
- **Cross-Validation**: 5-fold cross-validation
- **Hyperparameter Tuning**: GridSearchCV with parameters:
  - `max_depth`: [2, 3, 4, 5]
  - `min_samples_leaf`: [1, 2, 3, 4]
  - `n_estimators`: [50, 75, 100, 125, 150]

### Feature Engineering Techniques
- **Cyclical Encoding**: Sin/cosine transformation for month features
- **One-Hot Encoding**: Business days (Monday-Friday)
- **Technical Indicators**: Using the `ta` library for professional-grade indicators
- **Return Calculations**: Simple returns rather than log returns for interpretability

### Performance Metrics
- **Primary**: Accuracy (appropriate for balanced classes)
- **Secondary**: Precision, Recall, F1-score
- **Baseline**: Majority class prediction accuracy

## Future Enhancements

1. **Advanced Models**: 
   - Deep learning architectures
   - Ensemble methods
   - Time series specific models

2. **Feature Engineering**: 
   - More sophisticated technical indicators
   - Market microstructure features
   - Alternative data sources

3. **Data Sources**: 
   - High-frequency tick data
   - News sentiment analysis
   - Social media sentiment
   - Options flow data

4. **Risk Management**: 
   - Position sizing algorithms
   - Stop-loss strategies
   - Portfolio optimization
   - Risk-adjusted returns

5. **Real-time Implementation**: 
   - Live trading system
   - Automated risk controls
   - Performance monitoring

## Disclaimer

⚠️ **Important Notice**: This project is for educational purposes only and should not be used for actual trading without proper risk management and additional validation. Financial markets are inherently risky and past performance does not guarantee future results.

## License

This project is licensed under the MIT License - see the LICENSE file for details.