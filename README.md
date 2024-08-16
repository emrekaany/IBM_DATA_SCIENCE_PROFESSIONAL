**Project Overview**
This project is a practice exercise that I created while completing the IBM Data Science Professional Certificate. The goal is to apply various data science techniques learned during the course to predict insurance charges using real-world data. The dataset contains information such as age, gender, BMI, smoking habits, region, and number of children, with the target variable being charges (medical insurance costs).

This project covers a complete end-to-end data science workflow, including:

Data Quality Checks
Feature Engineering
Data Preprocessing
Model Training and Optimization
Model Evaluation
Visualization and Reporting
Project Components
Data Quality Checks

We begin by checking the dataset for missing values, duplicate records, and outliers.
Invalid values (such as negative ages or extreme BMI values) are detected and removed from the dataset.
Feature Engineering

Categorical columns (sex, smoker, region) are encoded into numerical values.
A new feature number_of_times_gave_birth is generated, based on the sex and children columns, to account for gender-specific childbearing impacts on insurance charges.
Data Preprocessing

The dataset is scaled using MinMaxScaler to prepare the features for model training.
The data is split into training, validation, and test sets for reliable model evaluation.
Model Training and Optimization

Fully Connected Neural Network: A neural network model is constructed using TensorFlow/Keras, and Bayesian Optimization is used to fine-tune hyperparameters like the number of units in each layer, dropout rate, and optimizer selection.
Linear Regression and Random Forest Regressor models are also trained to compare with the neural network performance.
Model Evaluation

The models are evaluated using MAPE (Mean Absolute Percentage Error) on training, validation, and test sets to measure their performance on predicting insurance charges.
The results for each model are stored in a DataFrame for comparison.
Visualization

Several visualizations are generated to better understand the relationship between features and the target variable (charges). These include:
Scatter plots (e.g., Age vs. Charges, BMI vs. Charges)
Box plots (e.g., Charges by Region)
Correlation Heatmap
Line plots (e.g., Average Charges by Age for Smokers vs. Non-Smokers)
The visualizations and performance results are saved as an HTML file for easy review.
Tools and Libraries Used
Python for scripting and model development
Pandas for data manipulation and preprocessing
Seaborn and Matplotlib for data visualization
Scikit-learn for feature scaling, data splitting, and model evaluation
TensorFlow/Keras for building and training the neural network model
Bayesian Optimization from the bayes_opt library for hyperparameter tuning
RandomForestRegressor and LinearRegression from Scikit-learn for comparison models
mpld3 for saving plots as interactive HTML visualizations
How to Run the Project
Data: Ensure the dataset (insurance.csv) is placed in the appropriate directory (e.g., 'C:\\Users\\ky4642\\Downloads\\insurance.csv').
Dependencies: Install the required Python libraries using pip:
bash
Copy code
pip install matplotlib seaborn pandas scikit-learn tensorflow bayesian-optimization mpld3
Execution: Run the Python script in your preferred environment (e.g., Jupyter Notebook, VS Code, or any Python IDE).
Output: The model performance results and visualizations will be saved to an HTML file in your specified output directory (e.g., 'C:\\Users\\ky4642\\Pictures\\visualizations.html').
Future Enhancements
Additional Feature Engineering: Adding more complex features such as interaction terms or polynomial features may improve model accuracy.
Advanced Model Tuning: Exploring more advanced hyperparameter tuning techniques, such as Grid Search or Genetic Algorithms, could optimize the models further.
Explainability: Implementing SHAP or LIME to better understand model decisions and improve interpretability.
Acknowledgements
This project was developed as part of the learning process during the IBM Data Science Professional Certificate program. Special thanks to IBM for the comprehensive course content and practical guidance.
