# CarPricePrediction

### **1. Importing Libraries**
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Lasso
from sklearn import metrics
```
- `pandas` is used for data manipulation and analysis.
- `matplotlib.pyplot` and `seaborn` are used for visualization.
- `sklearn.model_selection` helps in splitting the dataset for training and testing.
- `LinearRegression` and `Lasso` are machine learning models used for price prediction.
- `metrics` is used to evaluate model performance.

---

### **2. Loading the Dataset**
```python
car_dataset = pd.read_csv('car data.csv')
```
- Reads the dataset from a CSV file into a Pandas DataFrame.

---

### **3. Exploring the Dataset**
```python
car_dataset.head()
```
- Displays the first 5 rows of the dataset.

```python
car_dataset.shape
```
- Shows the number of rows and columns in the dataset.

```python
car_dataset.info()
```
- Provides information about the dataset, including column names, data types, and non-null values.
Here’s a step-by-step explanation of the remaining code in your **Car Price Prediction** notebook:

---

## **4. Checking Categorical Data Distribution**
```python
print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Selling_type.value_counts())
print(car_dataset.Transmission.value_counts())
```
- Displays the count of unique values in categorical columns like **Fuel_Type**, **Selling_type**, and **Transmission**.

---

## **5. Encoding Categorical Data**
```python
# Encoding the "Fuel_Type" column
car_dataset.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)

# Encoding the "Selling_type" column
car_dataset.replace({'Selling_type': {'Dealer': 0, 'Individual': 1}}, inplace=True)

# Encoding the "Transmission" column
car_dataset.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)
```
- Converts categorical variables into numerical values:
  - **Fuel_Type**: Petrol → 0, Diesel → 1, CNG → 2  
  - **Selling_type**: Dealer → 0, Individual → 1  
  - **Transmission**: Manual → 0, Automatic → 1  

```python
car_dataset.head()
```
- Displays the modified dataset after encoding.

---

## **6. Splitting Features and Target Variable**
```python
X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
Y = car_dataset['Selling_Price']
```
- **X**: Independent variables (features) after dropping **Car_Name** and **Selling_Price**.
- **Y**: Dependent variable (**Selling_Price**) to be predicted.

```python
print(X)
print(Y)
```
- Displays feature matrix (`X`) and target variable (`Y`).

---

## **7. Splitting Training and Testing Data**
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)
```
- Splits the dataset into **90% training** and **10% testing**.

---

## **8. Model Training - Linear Regression**
```python
# Loading Linear Regression model
lin_reg_model = LinearRegression()
```
- Creates a **Linear Regression** model.

```python
lin_reg_model.fit(X_train, Y_train)
```
- Trains the model using training data.

---

## **9. Model Evaluation - Linear Regression**
```python
# Prediction on Training Data
training_data_prediction = lin_reg_model.predict(X_train)
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error (Training Data): ", error_score)
```
- Predicts prices on training data.
- Evaluates model accuracy using **R² score**.

```python
# Prediction on Test Data
test_data_prediction = lin_reg_model.predict(X_test)
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error (Test Data): ", error_score)
```
- Predicts on **test data** and calculates **R² score**.

---

## **10. Model Training - Lasso Regression**
```python
# Loading Lasso Regression model
lasso_reg_model = Lasso()
```
- Creates a **Lasso Regression** model.

```python
lasso_reg_model.fit(X_train, Y_train)
```
- Trains the **Lasso Regression** model.

---

## **11. Model Evaluation - Lasso Regression**
```python
# Prediction on Training Data
training_data_prediction = lasso_reg_model.predict(X_train)
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error (Training Data - Lasso): ", error_score)
```
- Predicts on **training data** using **Lasso Regression**.
- Calculates **R² score**.

```python
# Prediction on Test Data
test_data_prediction = lasso_reg_model.predict(X_test)
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error (Test Data - Lasso): ", error_score)
```
- Predicts on **test data** using **Lasso Regression**.
- Calculates **R² score**.
