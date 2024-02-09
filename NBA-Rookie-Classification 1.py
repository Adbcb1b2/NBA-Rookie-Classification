# Import necessary libraries
import numpy as np  # Used for mathematical operations
import pandas as pd  # Used for data manipulation and analysis
import matplotlib.pyplot as plt  # Used for plotting graphs
import seaborn as sns  # Used for more sophisticated visualizations
from matplotlib.colors import Normalize, ListedColormap

# Import machine learning modules from sklearn
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.neural_network import MLPClassifier  # For creating a Neural Network
from sklearn.naive_bayes import GaussianNB  # For Naive Bayes classification
from sklearn.linear_model import LogisticRegression  # For Logistic Regression classification
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # For model evaluation
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn import metrics # For score functions, performance metrics and pairwise metrics and distance computations.
from sklearn.decomposition import PCA

# Read the dataset into a DataFrame
df = pd.read_csv('./nba_rookie_data.csv')

# Display the first few rows of the DataFrame to understand the data
df.head()

# Check the dimensions of the DataFrame (rows, columns)
df.shape

# Get the information on DataFrame structure and data types
df.info()

# Count the number of missing values in each column
missing_values = df.isnull().sum()
missing_values

# Fill missing values in '3 Point Percent' with the mean of the column
df['3 Point Percent'] = df['3 Point Percent'].fillna(df['3 Point Percent'].mean())

# Recheck for any remaining missing values
missing_values_c = df.isnull().sum()
missing_values_c

# Compute the correlation matrix to understand relationships between features
corr_matrix = df.corr()

# Plot the heatmap of the correlation matrix for visual inspection
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation between all Features")
plt.show()

# Group the data by the target variable and count the number of games played
print(df.groupby('TARGET_5Yrs')['Games Played'].count())

# Select features and target variable for modeling
X = df.iloc[:, [1, 3, 10, 15]].values  # Select relevant columns for features
y = df.iloc[:, 20].values  # Select the target column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Scale the features
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)

# Logistic Regression Model
logre = LogisticRegression()
logre.fit(X_train, y_train)  # Train the model
y_pred = logre.predict(X_test)  # Make predictions
# logre_accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy

# Print the model's accuracy and confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.show()
# Print classification report for Log Regression model
print(classification_report(y_test, y_pred))



# Gaussian Naive Bayes Model
gnb = GaussianNB()
gnb.fit(X_train, y_train)  # Train the model
y_gnb_pred = gnb.predict(X_test)  # Make predictions
# gnb_accuracy = accuracy_score(y_test, y_gnb_pred)  # Calculate accuracy

# Print the model's confusion matrix
gnb_cm = confusion_matrix(y_test, y_gnb_pred)
sns.heatmap(gnb_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.show()

# Print classification report for Gaussian Naive Bayes model
print(classification_report(y_test, y_gnb_pred))

# Neural Network Model (Multi-layer Perceptron)
mlp = MLPClassifier(hidden_layer_sizes=(50, 30), activation="tanh", max_iter=2000, random_state=0)
mlp.fit(X_train, y_train)  # Train the model
y_mlp_pred = mlp.predict(X_test)  # Make predictions
mlp_accuracy = accuracy_score(y_test, y_mlp_pred)  # Calculate accuracy

# Print the model's accuracy and confusion matrix
print('Our Accuracy is %.3f' % mlp_accuracy)
mlp_cm = confusion_matrix(y_test, y_mlp_pred)
sns.heatmap(mlp_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.show()

# Print classification report for Neural Network model
print(classification_report(y_test, y_mlp_pred))

# Running the models for all features

# Select all features (except the target variable) for modeling
X1 = df.iloc[:, 1:19].values
# Select the target column for modeling
y1 = df.iloc[:, 20].values

# Split the dataset into training and testing sets (1/3rd of the data for testing)
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=1/3, random_state=0)

# Scale the features using StandardScaler for normalization
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)

# Initialize the Logistic Regression model
logre = LogisticRegression()
# Fit the Logistic Regression model to the training data
logre.fit(X_train, y_train)

# Make predictions on the test data
y_pred = logre.predict(X_test)


# Evaluate the accuracy of the Logistic Regression model
logre_accuracy = accuracy_score(y_test, y_pred)
# Print the accuracy
print('Our Accuracy is %.3f' % logre_accuracy)

# Calculate the number of mislabeled points for the Logistic Regression model
print('Number of LOGre Mislabeled points out of a total %d points : %d'
      % (X_test.shape[0], (y_test != logre.predict(X_test)).sum()))

# Print classification report for all feature log regression model
print(classification_report(y_test, y_pred))

# Generate the confusion matrix for the Logistic Regression model
cm = metrics.confusion_matrix(y_test, y_pred)
# Plot the confusion matrix using seaborn's heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Gaussian Naive Bayes

# Initialize the Gaussian Naive Bayes model
gnb = GaussianNB()
# Fit the model to the training data
gnb.fit(X_train, y_train)

# Make predictions on the test data
y_gnb_pred = gnb.predict(X_test)

# Evaluate the accuracy of the Gaussian Naive Bayes model
gnb_accuracy = accuracy_score(y_test, y_gnb_pred)
# Print the accuracy
print('Our Accuracy is %.3f' % gnb_accuracy)

# Calculate the number of mislabeled points for the Gaussian Naive Bayes model
print('Number of GNB Mislabeled points out of a total %d points : %d'
      % (X_test.shape[0], (y_test != gnb.predict(X_test)).sum()))

# Generate the confusion matrix for the Gaussian Naive Bayes model
gnb_cm = metrics.confusion_matrix(y_test, y_gnb_pred)
# Plot the confusion matrix using seaborn's heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(gnb_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title('Gaussian Naive Bayes Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print a classification report to evaluate the Gaussian Naive Bayes model
print(classification_report(y_test, y_gnb_pred))

# Neural Network

# Initialize the Neural Network (Multi-layer Perceptron) model with specified hidden layers and activation function
mlp = MLPClassifier(hidden_layer_sizes=(50, 30, 100), activation="tanh", max_iter=2000, random_state=0)
# Fit the Neural Network model to the training data
mlp.fit(X_train, y_train)

# Make predictions on the test data
y_mlp_pred = mlp.predict(X_test)

# Evaluate the accuracy of the Neural Network model
mlp_accuracy = accuracy_score(y_test, y_mlp_pred)
# Print the accuracy
print('Our Accuracy is %.3f' % mlp_accuracy)

# Calculate the number of mislabeled points for the Neural Network model
print('Number of NN Mislabeled points out of a total %d points : %d'
      % (X_test.shape[0], (y_test != mlp.predict(X_test)).sum()))

# Generate the confusion matrix for the Neural Network model
mlp_cm = metrics.confusion_matrix(y_test, y_mlp_pred)


# Plot the confusion matrix using seaborn's heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(mlp_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title('Neural Network Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print a classification report to evaluate the Neural Network model
print(classification_report(y_test, y_mlp_pred))



# Apply PCA and reduce the data to two principal components for visualization
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Initialize and train the MLPClassifier on the reduced data
mlp = MLPClassifier(hidden_layer_sizes=(50, 30), activation="tanh", max_iter=2000, random_state=0)
mlp.fit(X_train_pca, y_train)

# Evaluate the classifier
y_mlp_pred = mlp.predict(X_test_pca)
mlp_accuracy = accuracy_score(y_test, y_mlp_pred)
print('Our Accuracy is %.3f' % mlp_accuracy)

# Create a mesh grid for the two principal components
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
h = 0.02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict using the trained MLPClassifier directly in the PCA-transformed space
Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Set up the color and symbol encoding
nm = Normalize(vmin=0, vmax=1)
cm = ListedColormap(["blue", "red"])

# Plot the contour and the test points in the PCA-transformed space
fig, ax = plt.subplots()
ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=cm, norm=nm, edgecolors='k')

# Show the plot
plt.show()

print('Number of NN Mislabeled points out of a total %d points : %d'
      % (X_test_pca.shape[0], (y_test != mlp.predict(X_test_pca)).sum()))
# Print classification report for PCA NN model
print(classification_report(y_test, y_mlp_pred))