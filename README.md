## GOVERNMENT PROMOTE OR CONTROL DEVELOPMENTS IN MACHINE LEARNING AND AI IN CREDIT CARD FRAUD

## ABSTRACT
 The technology of machine learning and artificial intelligence in the world is rapidly increasing daily. This will be done by making use of a dataset from Kaggle,saving the Microsoft Excel to CSV, Jupyter Notebook and Python to select some data analysis of an online fraud transaction by grouping them into different categories and then knowing the total entities, statistics, machine learning algorithms, the data quantity of the fraudster and non-fraudster. The confusion matrix for quality of the data of a given sample. The result and discussion represent the visualization of data by using graphs to describe the machine learning and AI algorithm using the ROC curve of the fraud and non-fraud online transaction.

## MACHINE LEARNING ALGORITHM STEP

1. Get the dataset from the Kaggle.

2. Import the libraries of the dataset.

3. Divide the dataset into two part i.e,. Train dataset and Test dataset.

4. Calcuate the accuracy and performance metrics of each algorithm using confusion matrix.

## DATA SET
 Credit card fraud dataset can be gotten from Kaggle that containing the combination of fraud and non-fraud transaction (Sheo , et al., 2022). CSV files are mostly used format for machine learning data. The total number of transaction for credit card detection is 284807 and the number of fraud is 492 and non fraud is 284315. The dataset is highly imbalanced; the fraud have a percentage of 0.2% and non fraud have a percentage of 99.8%.
  The dataset features are transformed using the principal component analysis (PCA) such as V1, V2, V3 …….. V28 are PCA features and Time, Amount and Class are non-PCA features.

## CONFUSION MATRIX

1. TRUE POSITIVE (TP): 
  This show that the actual number of fraud transactions predicted as fraud.

2. TRUE NEGATIVE (TN): 
  This show that the actual number of fraud transactions predicted 
as not fraud.

3. FALSE POSITIVE (FP): 
 The number of not fraud transactions predicted as fraud.

4. FALSE NEGATIVE (FN):
 The number of fraud transactions predicted as not fraud.

## PERFORMANCE MATRIX

# 1. Accuracy:
 The Random Forest demonstrated the highest accuracy at 99.96%, closely followed by KNN with 99.95%, the lowest accuracy is K-means. These algorithms demonstrated that random forest is mostly used for detection for credit card fraud.

# 2. Precision: 
  Random forest exhibited notable precision of 97.44% which is the highest 
among the five machine learning algorithms and ensuring comprehensive fraud 
detection.

# 3. Recall Analysis:
 Random forest and KNN exhibited notable recall of 77.55%,indicating its capability to minimize false positives while capturing a higher proportion of actual fraudulent transactions. 

# 4. F1_Score: 
Random Forest achieved an  F1-Score of 86.36%, reflecting its balanced 
performance in harmonizing Precision and Recall. This balanced measure demonstrates 
the algorithm's ability to maintain a reasonable trade-off between false positives and false negatives in identifying fraudulent activities.

# 5. K- Means (Clustering): 
  While K-means clustering demonstrated slightly lower metrics across Accuracy, Precision, and F1-Score compared to supervised learning algorithms, it's important to note that clustering algorithms like K-means can provide insights into data patterns but might not be as effective for precise fraud identification.

## CONCLUSION
   The credit card fraud detection has been in existence for years and the researcher has a 
lot of work to deal with and this has make it difficult for them to detect the fraudulent 
transaction, with the used of machine learning algorithm they can find the solution after theyhave happened that this can be stressful to locate them. There are lot of numerous transactionthat occur in fraudulent transaction.
  From this research we have examined and deal with the issues as it pertained to the 
credit card fraud detection transaction (Kazeem, 2023) , so that the government can promote and control the development of machine learning (ML) and Artificial Intelligence (AI) with the used of algorithms to stop the fraud that is affecting the society in data ecology, this can be done when the government or credit card firm can able to find the solution to detect fraudulent transaction (Muhammad , et al., n.d.). Also, to evaluate the sample dataset from the Kagglefrom the credit card detection we have observed that there are two hundred and eighty-four thousand eight hundred and seven (284807) rows and thirty-one (31) columns transaction. The fraud have four hundred and ninety-two (492) and non_fraud (does not have fraud) of two hundred and eighty-four thousand three hundred and fifteen (284315) transactions and there is the small whole number to separate the one that have the fraud and non-fraud transaction by the used of binary integer to represent which are 0 and 1. 0 is for those that do not have fraud and 1 is for those who have fraud transaction and the pie chat was used to know the percentage of the fraudulent transaction which are 99.8% and 0.2%. A confusion matrix is a major problem in calculating the data which are divided into four parts: True positive, True negative, False positive, and false negative.
  There are different machine learning techniques like logistic regression, Decision Tree and Random forest were used to detect fraud in credit card systems. Also, the accuracy was used to detect the data quality of the fraudulent transaction that can be used to evaluate the performance of the system. The accuracy for logistic regression, Decision tree and Random forest classifier are 90.0, 94.3 and 95.5 respectively, by comparing all the three methods, the random forest technique or classifier is the best than that of logistic regression and Decision tree (Andrea , et al., 2014). The above research has demonstrated to us that machine learning techniques or Artificial Intelligence techniques are capable of handling fraudulent cases in the government, society and the environment.
  We have used the graphical representation of machine learning algorithms to predict the sample of data analysis to split the one that has the most fraudulent and non-fraudulent transaction using the ROC curve, train, test, Logistic Regression, Support Vector Machine 
(SVM), Decision Tree Classifier, Random Forest to tackle the detection problem in the society.










