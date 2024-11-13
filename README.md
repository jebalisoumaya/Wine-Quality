Decision Tree Classification for Wine Quality Prediction
This project utilizes a Decision Tree Classifier to predict the quality of wines based on various chemical features. 
The dataset used is the Wine Quality Dataset, which includes information about both red and white wines, along with their corresponding quality ratings.
The objective is to classify the wines into three categories: Low, Medium, and High quality.


2/ Dataset
The dataset used in this project is the Wine Quality Dataset (WineQT.csv).

The quality column contains a score ranging from 0 to 10, which is converted into three categories for classification:

* Low (0): For quality <= 4 
* Medium (1): For quality between 5 and 6
* High (2): For quality > 6
3/ Code Overview
The code follows these main steps:

* Data Exploration and Preprocessing:
* Load the Dataset: The dataset is loaded into a Pandas DataFrame.
* Display Basic Information: Basic statistics are displayed, including the first few rows of data, data types, and summary statistics.
* Check for Missing Values and Duplicates: Ensure the dataset is clean by checking for any missing values or duplicate entries.
* Calculate the Correlation Matrix: Identify key features most correlated with the target variable (quality).
4/ Model Building:
* Feature Selection: Based on the correlation matrix, the most important features related to wine quality are selected for training the model.
* Categorize Quality: Convert the continuous quality ratings into three categories: Low, Medium, and High.
* Train-Test Split: Split the data into training and testing sets using train_test_split.
* Model Training: A Decision Tree Classifier is trained with a maximum depth of 3 and a minimum sample leaf of 5.
* Model Evaluation:
Accuracy: Evaluate the model using accuracy, which measures the percentage of correct predictions.
Confusion Matrix: Display a confusion matrix to visualize the performance of the model across the different wine quality categories.
Classification Report: Generate a classification report that includes precision, recall, and F1-score for each class (Low, Medium, High).
6/ Visualization:
* Visualize the Decision Tree: The trained decision tree is visualized using plot_tree to provide insights into the decision-making process and the feature importance.


The  accuracy of the Decision Tree classifier is 84.84%, which means that the model is correctly predicting the wine quality category about 85% of the time.
However, the confusion matrix and classification report indicate areas for improvement, particularly in the prediction of some classes.

![image](https://github.com/user-attachments/assets/2c90e0ce-9554-402f-bd39-94c7dc614688)

Class 0 (Low): There are 9 wines that the model predicted as belonging to Medium quality, but none of the wines predicted as Low quality were actually Low. This indicates the model is struggling to identify wines with low quality.
Class 1 (Medium): The model performed well for Medium quality wines, with 279 correctly predicted and only 10 misclassified as High quality.
Class 2 (High): For High quality wines, only 12 wines were correctly classified, while 33 wines were misclassified as Medium

From the clasification report :
![image](https://github.com/user-attachments/assets/8e4e129a-0afc-4fdf-bbb6-4fece3daa894)

Class 0 (Low): The precision, recall, and F1-score for the Low class are all 0.00, indicating that the model does not predict Low wines effectively. 
This is likely because there is a small number of Low wines (9 instances), and the model fails to classify them.

Class 1 (Medium): The precision (0.87) and recall (0.97) for Medium quality wines are quite high.
The model is very effective at predicting this class, with an F1-score (0.91) indicating strong performance for Medium wines.

Class 2 (High): The precision (0.55) for High quality wines is relatively low, while the recall (0.27) is even lower, meaning the model struggles to detect High quality wines and often misclassifies them as Medium. 
The F1-score (0.36) reflects this poor performance.




