# CustomerChurn_G15
WQD7001 PDS_Group 15 Project Bank Customer Churn

**OBJECTIVE**
1. To identify the factors influencing customer churn using historical interaction and behavioural data. 

2. To develop a machine learning model to predict the likelihood of customer churn. 

3. To compare the performance of at least three machine learning algorithms (e.g., logistic regression, decision trees, and random forests) in predicting customer churn.

**DATASET**
Dataset comprises of 10,000 samples and 18 features.

Dataset contains 3 primary types of information:
1. User Data that contains key demographic and personal information about customer: Customer Id, Surname, Credit Score, Geography, Gender, Age, Has Credit Card, Estimated Salary, Exited
2. Card Data that tell information about and product usage: Balance, Number of Products, Card Type, Point Earned
3. Sentiment Driver which reflects customer satisfaction and engagement: Complain, Satisfaction Score, Is Active Member
   
12 features with an integer data type, 4 features with an object data type, and 2 features with a float data type. Conversion of data type to int format needed.

**PROCEDURE**
Visualized and analyzed data related to customer churn by using CORRELATION HEATMAP
Preprocessed and transforms categorical data for Machine Learning model training using pandas toolkit and standardization techniques
Explored 6 various data modelling, followed by hypertune the best models and tested it on test data to identify the best performing model.
Evaluated Models using confusion matrix, ROC,AUC.
Feature selection was done to identify the most imprtance feature and used it for data product development.
