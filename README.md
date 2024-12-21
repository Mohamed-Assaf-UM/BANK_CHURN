# BANK_CHURN_MODEL

This video transcript introduces the process of building an end-to-end project using deep learning, with a focus on developing and deploying a binary classification model using Artificial Neural Networks (ANNs). Here's a simplified breakdown of the key points:  

---

### **Aim**  
To create an end-to-end deep learning project for predicting customer churn in a bank using TensorFlow and Keras.

---

### **Theory**  

1. **Frameworks and Libraries**:
   - **TensorFlow**: An open-source library for building deep learning models, capable of implementing various neural networks like ANN, RNN, LSTM, GRU, and Transformers.
   - **Keras**: An API integrated with TensorFlow, simplifying model creation with less code.
   - Additional Libraries: NumPy, Pandas, Scikit-learn, Matplotlib, and Streamlit.

2. **Problem Statement**:
   - Predict if a customer will exit the bank based on features like credit score, geography, gender, age, balance, etc.
   - Use the *Churn_Modelling.csv* dataset, which includes input features and a binary target column (`Exited`).

3. **Steps Involved**:
   - **Data Preprocessing**:
     - Handle categorical variables by encoding them numerically.
     - Perform standardization or normalization.
   - **Model Development**:
     - Build an ANN with an input layer (11 features), hidden layers, and an output layer (binary classification).
     - Implement techniques like dropout to avoid overfitting.
   - **Training and Evaluation**:
     - Use forward and backward propagation to train the model.
     - Optimize the model using loss functions and optimizers.
   - **Model Serialization**:
     - Save the model in `.h5` or `.pkl` formats for deployment.
   - **Deployment**:
     - Develop a web app using Streamlit to integrate and deploy the trained model.

---

### **Inferences**  

1. **Advantages of Keras**:
   - Reduces boilerplate code when building and training models.
   - Allows seamless implementation of complex neural network architectures.

2. **Importance of Deployment**:
   - Storing the model's weights and architecture for real-world usage.
   - Provides an accessible interface for users through a web app.

---

### **Result**  

By the end of the project:
- A functional ANN model will classify bank customers based on their likelihood of leaving.
- The model will be integrated into a Streamlit web app and deployed on the Streamlit cloud.

---

### Next Steps  

1. Set up the project environment:
   - Use `conda` to create a virtual environment.
   - Install dependencies listed in `requirements.txt`.
2. Preprocess the dataset and implement basic feature engineering.
3. Start building the ANN model using TensorFlow and Keras.

---
Here's an explanation of your code step by step, with the purpose of each block:

---

### **Installing `ipykernel`**
```python
# Installed ipykernel to enable using Jupyter Notebook-specific features
```
- **Purpose**: Ensures the Jupyter environment can handle kernel-related operations like saving sessions or switching environments effectively. It is often required when working with notebooks in a virtual environment.

---

### **Importing Libraries**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
```
- **Purpose**:
  - `pandas`: Handles data manipulation and analysis, especially with DataFrames.
  - `train_test_split`: Splits the data into training and testing sets.
  - `StandardScaler`: Scales numeric features to have zero mean and unit variance.
  - `LabelEncoder` and `OneHotEncoder`: Encode categorical variables.
  - `pickle`: Saves Python objects like models or scalers to a file for reuse.

---

### **Loading the Dataset**
```python
data = pd.read_csv("Churn_Modelling.csv")
data.head()
```
- **Purpose**: 
  - Loads the dataset into a DataFrame.
  - Displays the first 5 rows to verify the structure.

---

### **Dropping Irrelevant Columns**
```python
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
```
- **Purpose**: 
  - Removes columns that do not contribute to the model (e.g., IDs or names).

---

### **Encoding Gender**
```python
label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
```
- **Purpose**:
  - Converts `Gender` from a categorical variable (`Male`, `Female`) to numerical values (`0` for Female, `1` for Male), making it machine-readable.

---

### **One-Hot Encoding Geography**
```python
onehot_encoder_geo = OneHotEncoder()
geo_encoder = onehot_encoder_geo.fit_transform(data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoder, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
data = pd.concat([data.drop('Geography', axis=1), geo_encoded_df], axis=1)
```
- **Purpose**:
  - Converts the `Geography` column into three binary columns (`Geography_France`, `Geography_Germany`, `Geography_Spain`) for each region using one-hot encoding.
  - Adds these new columns back to the DataFrame after dropping the original `Geography`.

---
In this line:

```python
geo_encoder = onehot_encoder_geo.fit_transform(data[['Geography']]).toarray()
```

The `.toarray()` method is used to convert the output of `fit_transform`, which is a **sparse matrix**, into a **dense NumPy array**.

### Why do we use `.toarray()`?
1. **Efficiency**:
   - `OneHotEncoder` by default outputs a **sparse matrix** to save memory when dealing with a large number of categories, especially if most of the data in the matrix is zero (which is common in one-hot encoding).
   - A sparse matrix stores only the non-zero values and their positions, rather than the entire matrix.

2. **Conversion to Dense Format**:
   - `.toarray()` converts the sparse matrix into a dense format (a NumPy array) where all values, including zeros, are stored explicitly. 
   - This makes it easier to work with the data in subsequent steps, such as creating a pandas DataFrame.

3. **Integration with pandas**:
   - To use the one-hot encoded data in a pandas DataFrame, you need a dense array format. The `pd.DataFrame` constructor does not directly accept sparse matrices, so the `.toarray()` ensures compatibility.

### Without `.toarray()`
If you skip `.toarray()`, the variable `geo_encoder` will remain a sparse matrix. While sparse matrices can be useful for saving memory, they are not as straightforward to manipulate and integrate with pandas DataFrames or other libraries that expect dense data.

### Example
Here’s how the data looks before and after `.toarray()`:

#### Output with `.toarray()`:
```python
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
```

#### Output without `.toarray()` (sparse matrix):
```python
<3x3 sparse matrix of type '<class 'numpy.float64'>'
	with 3 stored elements in Compressed Sparse Row format>
```

The sparse matrix representation is more memory-efficient but less convenient for DataFrame creation and further processing.

In short, you use `.toarray()` for **compatibility with pandas and ease of manipulation**, while accepting a slight increase in memory usage.

---

### Code:
```python
geo_encoded_df = pd.DataFrame(geo_encoder, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
data = pd.concat([data.drop('Geography', axis=1), geo_encoded_df], axis=1)
```

---

### Step 1: Understand the Data Before Encoding

Assume you have this `data` DataFrame:

| CustomerID | Geography | Age | Salary |
|------------|-----------|-----|--------|
| 1          | France    | 30  | 50000  |
| 2          | Germany   | 40  | 60000  |
| 3          | Spain     | 35  | 55000  |

Here, the `Geography` column has categorical values: **France**, **Germany**, and **Spain**.

---

### Step 2: One-Hot Encode `Geography`

The `OneHotEncoder` transforms `Geography` into three separate columns (one for each category). After encoding, you get:

```python
geo_encoder = [[1, 0, 0],  # France
               [0, 1, 0],  # Germany
               [0, 0, 1]]  # Spain
```

---

### Step 3: Convert to DataFrame with Column Names

The line:
```python
geo_encoded_df = pd.DataFrame(geo_encoder, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
```

Converts the encoded data into a **DataFrame** and assigns proper column names. You get:

| Geography_France | Geography_Germany | Geography_Spain |
|-------------------|-------------------|-----------------|
| 1                 | 0                 | 0               |
| 0                 | 1                 | 0               |
| 0                 | 0                 | 1               |

---

### Step 4: Combine with the Original DataFrame

The line:
```python
data = pd.concat([data.drop('Geography', axis=1), geo_encoded_df], axis=1)
```

1. Removes the original `Geography` column using `data.drop('Geography', axis=1)`.
2. Combines the remaining columns with the newly created `geo_encoded_df`.

After this step, the final DataFrame looks like:

| CustomerID | Age | Salary | Geography_France | Geography_Germany | Geography_Spain |
|------------|-----|--------|-------------------|-------------------|-----------------|
| 1          | 30  | 50000  | 1                 | 0                 | 0               |
| 2          | 40  | 60000  | 0                 | 1                 | 0               |
| 3          | 35  | 55000  | 0                 | 0                 | 1               |

---

### In Simple Words:

1. **Convert encoded data into a DataFrame**: Turn the matrix of 0s and 1s into a table with proper column names (`Geography_France`, `Geography_Germany`, etc.).
2. **Drop the old column**: Remove the `Geography` column from the original data.
3. **Combine**: Add the new one-hot encoded columns to the original DataFrame.

This ensures you replace the categorical `Geography` column with its one-hot encoded representation in the data.                   

---
### **Saving Encoders**
```python
with open('label_encoder_gender.pkl', 'wb') as file:
    pickle.dump(label_encoder_gender, file)

with open('onehot_encoder_geo.pkl', 'wb') as file:
    pickle.dump(onehot_encoder_geo, file)
```
- **Purpose**:
  - Saves the encoders for `Gender` and `Geography` to files so they can be reused later, ensuring consistency between training and testing.

---
Let's break down the steps where you save the encoders into files using **pickle** in Python.

---

### Code:
```python
with open('label_encoder_gender.pkl', 'wb') as file:
    pickle.dump(label_encoder_gender, file)

with open('onehot_encoder_geo.pkl', 'wb') as file:
    pickle.dump(onehot_encoder_geo, file)
```

---

### Step 1: What is Happening?
1. **Purpose**:  
   - You are saving two objects, `label_encoder_gender` and `onehot_encoder_geo`, to files using the **pickle** library. 
   - Pickling serializes Python objects so they can be stored and reloaded later.

2. **Why Save These Encoders?**  
   - When you preprocess your data (like label encoding or one-hot encoding), you want to reuse the same encoders during inference or deployment.  
   - By saving them, you don’t need to re-fit the encoders on the same data every time.

---

### Step 2: Breaking It Down

#### **a) First Line**
```python
with open('label_encoder_gender.pkl', 'wb') as file:
    pickle.dump(label_encoder_gender, file)
```

- `open('label_encoder_gender.pkl', 'wb')`:  
   - Opens a file named `label_encoder_gender.pkl` in **write-binary** (`wb`) mode.  
   - `.pkl` is a common extension for pickled files, but it’s not mandatory.

- `pickle.dump(label_encoder_gender, file)`:  
   - **Serializes** the `label_encoder_gender` object (which was created earlier for encoding gender) and writes it into the file.  
   - This means the object is saved in a format that can be loaded back later.

#### **b) Second Line**
```python
with open('onehot_encoder_geo.pkl', 'wb') as file:
    pickle.dump(onehot_encoder_geo, file)
```

- Similar to the first line but saves the `onehot_encoder_geo` object, which was used to encode the `Geography` column.

---

### Step 3: Example Use Case

Imagine you trained your encoders as part of a machine learning pipeline for customer data. You save them now and can later reload them for use without retraining.

---

### Step 4: Why Use `with open()`?
- The `with` statement ensures that the file is **automatically closed** after saving the object, even if an error occurs.

---

### Real-Life Example

Suppose you have the following objects before saving:
```python
label_encoder_gender = LabelEncoder()
onehot_encoder_geo = OneHotEncoder()
```

After you save them using `pickle.dump`, two files are created:
1. `label_encoder_gender.pkl`
2. `onehot_encoder_geo.pkl`

Later, when you need to use these encoders, you can load them back like this:
```python
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)
```

This ensures consistency in preprocessing both during training and deployment.

---

### In Simple Words:
- **Save objects for later use**: You save the `label_encoder_gender` and `onehot_encoder_geo` objects into `.pkl` files so they can be reused later.
- **Reusable in deployment**: When your model runs in production, it can use the same encoders you trained with.
- **Efficient**: No need to retrain or redefine the encoders; just load the saved files.
---
You **can** use a new `OneHotEncoder` or `LabelEncoder` whenever needed, but there are strong reasons why you should save them and reuse the same encoders, especially in **real-world machine learning workflows**.

---

### Why Save the Encoders?

#### 1. **Ensures Consistency**
- If you train a machine learning model with specific encodings, the exact same encodings must be used during deployment or inference.  
- **Example**:  
   Suppose `LabelEncoder` encoded `Gender` as:
   - `Male -> 0`  
   - `Female -> 1`  

   If you train your model with this mapping but later re-encode the column with a fresh `LabelEncoder`, it might generate:
   - `Female -> 0`  
   - `Male -> 1`  

   This mismatch will confuse the model and lead to **incorrect predictions**.

---

#### 2. **Avoids Data Dependency**
- Encoders rely on the **data they are trained on**.  
  - For example, if `OneHotEncoder` sees `['France', 'Germany', 'Spain']`, it generates:
    ```
    France -> [1, 0, 0]
    Germany -> [0, 1, 0]
    Spain -> [0, 0, 1]
    ```

- If later you use a fresh `OneHotEncoder` on new data, it may generate inconsistent results or throw errors if some categories are missing.

---

#### 3. **Handles Missing Categories in Inference**
- If your model encounters a category in test or deployment data that wasn’t in the training data, reusing the saved encoder avoids errors.  
- A new encoder will not "know" how to handle such cases, while a saved encoder will have a proper configuration to manage them (e.g., by ignoring or setting unknown categories to `0`).

---

#### 4. **Saves Time and Effort**
- Re-training encoders is unnecessary when you’ve already done the work once.  
- Loading a pre-saved encoder is much faster and avoids repeating the preprocessing steps.

---

#### 5. **Production/Deployment Needs**
- In a production environment (e.g., deploying a model as an API), you often cannot retrain the encoders. Instead, you must use the encoders that were trained with the model to ensure compatibility.
- If you don't save the encoders, your deployed pipeline may fail.

---

### Real-World Scenario Example

**Training Pipeline**:
1. Your training data contains `Gender` and `Geography` columns.
2. You encode them with `LabelEncoder` and `OneHotEncoder`.
3. You save the model **and** the encoders (`.pkl` files).

**Deployment Pipeline**:
1. During deployment, you load the saved model and encoders.
2. When new data arrives (e.g., `Gender = Male, Geography = France`), you reuse the saved encoders to preprocess the input data **exactly as during training**.

---

### What Happens if You Don't Save?
1. **Encoding Mismatch**: The new encoder may create different mappings, breaking the model.  
2. **Errors**: If the new data contains unseen categories, the model won’t work correctly.  
3. **Redundant Work**: You’ll waste time recreating the encoders every time.  

---

### When Can You Skip Saving?
If your application is small-scale or exploratory, and you don’t need consistent preprocessing across training and inference, you can skip saving encoders. However, for anything beyond basic experimentation, saving encoders is a **best practice**.

---

### **Separating Independent and Dependent Features**
```python
X = data.drop('Exited', axis=1)
y = data['Exited']
```
- **Purpose**:
  - Splits the data into:
    - `X`: Features used for prediction (independent variables).
    - `y`: The target variable (`Exited`, representing customer churn).

---

### **Splitting Data into Training and Testing Sets**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- **Purpose**:
  - Divides the dataset into training (80%) and testing (20%) subsets.
  - `random_state=42` ensures consistent results across runs.

---

### **Scaling Features**
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
- **Purpose**:
  - Standardizes the numerical features in `X_train` and `X_test` to ensure they are on the same scale, improving model performance.
  - `fit_transform`: Calculates the mean and variance on the training set, and scales it.
  - `transform`: Uses the same scaling parameters to scale the test set.

---

### **Saving the Scaler**
```python
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
```
- **Purpose**: Saves the scaler object for reuse during prediction on new data.

---

### **Final Processed Data**
```python
data.head()
```
- **Purpose**: Displays the first 5 rows of the final processed DataFrame for verification.

---

### **Key Purpose of the Entire Code**
1. **Prepares the Dataset for Machine Learning**:
   - Encodes categorical variables.
   - Scales numerical variables.
   - Splits data into training and testing sets.

2. **Saves Preprocessing Tools**: Saves the encoders and scaler for consistent preprocessing of new data during model evaluation or deployment.

---
This block of code is setting up a neural network using TensorFlow and Keras. Here's a detailed explanation of each part with its purpose, followed by a visual and simple explanation:

---

### **1. Imports**:
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime
```

- **TensorFlow**: A library for building and training machine learning and deep learning models.
- **Sequential**: A Keras model type where layers are added in sequence, from input to output.
- **Dense**: A layer type where every neuron in one layer is connected to every neuron in the next layer.
- **EarlyStopping**: A callback to stop training when the model performance stops improving, preventing overfitting.
- **TensorBoard**: A tool for visualizing training metrics like loss and accuracy over time.
- **datetime**: Used for generating timestamps to organize TensorBoard logs.

---

### **2. Neural Network Model**:
```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  ## Hidden Layer 1
    Dense(32, activation='relu'),                                  ## Hidden Layer 2
    Dense(1, activation='sigmoid')                                ## Output Layer
])
```

#### Explanation of each line:
1. **`Dense(64, activation='relu', input_shape=(X_train.shape[1],))`**:
   - **Dense**: A fully connected layer with 64 neurons.
   - **`activation='relu'`**: Applies the ReLU activation function, which outputs 0 for negative inputs and the input itself for positive inputs, making it useful for non-linear problems.
   - **`input_shape=(X_train.shape[1],)`**: Specifies the number of features in the input data so the model knows the input size.

2. **`Dense(32, activation='relu')`**:
   - A second hidden layer with 32 neurons and ReLU activation.

3. **`Dense(1, activation='sigmoid')`**:
   - The output layer with 1 neuron. 
   - **`activation='sigmoid'`**: Outputs a value between 0 and 1, making it suitable for binary classification.

---

### **Visualizing the Neural Network**:

Imagine your neural network as a flow of information:

- **Input Layer**:
  - Takes the input data (features from `X_train`).
  - Shape: Number of features in `X_train`.

- **Hidden Layer 1 (HL1)**:
  - 64 neurons fully connected to the input layer.
  - ReLU activation introduces non-linearity.

- **Hidden Layer 2 (HL2)**:
  - 32 neurons fully connected to HL1.
  - ReLU activation applied again.

- **Output Layer**:
  - 1 neuron predicts the final output.
  - Sigmoid activation converts the output into a probability (0 or 1 for binary classification).

---
![image](https://github.com/user-attachments/assets/61bbcc60-bab8-425a-a578-8fb325bd3968)

### **Purpose of Each Layer**:
1. **Hidden Layers**:
   - Extract complex patterns from the input data.
   - The first layer processes the raw data, and subsequent layers refine it.

2. **Output Layer**:
   - Provides the prediction (e.g., a probability of belonging to one class).

---

### **Simple Analogy**:

Think of it as an assembly line:

1. **Input Layer**: Raw materials (your data).
2. **HL1**: First processing step, where the main assembly is done.
3. **HL2**: Refines and polishes the output from the first step.
4. **Output Layer**: The final product (classification or prediction).

---

### **Diagram of the Neural Network**:

```plaintext
Input Layer       Hidden Layer 1       Hidden Layer 2       Output Layer
     |                  |                     |                    |
[Feature 1]        (64 neurons)          (32 neurons)          (1 neuron)
[Feature 2]        ReLU Activation      ReLU Activation     Sigmoid Activation
  ... 
[Feature n]        Fully Connected      Fully Connected     Final Probability
```

The illustration shows a simple neural network:

- **Input Layer**: Contains nodes for each feature in the dataset. If your dataset has `X_train.shape[1]` features (e.g., 5), the input layer will have 5 nodes.
- **Hidden Layer 1**: The first dense layer with 64 neurons and the ReLU activation function, connected to the input layer.
- **Hidden Layer 2**: The second dense layer with 32 neurons and the ReLU activation function, connected to the first hidden layer.
- **Output Layer**: A single neuron with the Sigmoid activation function, used for binary classification.

### Why `input_shape=(X_train.shape[1],)` is a Tuple
- **`input_shape` Parameter**: Specifies the shape of the input data. Here, `X_train.shape[1]` gives the number of features (columns) in the training data.
- **Tuple Format**: TensorFlow requires the input shape to be in tuple format because it supports multi-dimensional data. Even for a single-dimensional input (e.g., a flat array of features), it needs the tuple structure for consistency and scalability to higher dimensions.
---

### Explanation of `model.summary()` Output

1. **Model Name**: `"sequential"`  
   Indicates that the model is a **Sequential Model**, meaning the layers are stacked linearly one after the other.

---

### Layers and Parameters

| **Layer**          | **Output Shape** | **Params (#)** | **Explanation**                                                                                         |
|---------------------|------------------|----------------|---------------------------------------------------------------------------------------------------------|
| **dense (Dense)**   | `(None, 64)`     | 832            | - **Input to Hidden Layer 1**  
  - `(None, 64)` means the batch size is dynamic (`None`) and each batch has 64 neurons.  
  - Parameters = **(Input Features + 1 Bias)** × **Neurons** = `(13 + 1) × 64 = 832`. |
| **dense_1 (Dense)** | `(None, 32)`     | 2,080          | - **Hidden Layer 1 to Hidden Layer 2**  
  - Parameters = **(64 + 1 Bias)** × **32 Neurons** = `65 × 32 = 2,080`.                                     |
| **dense_2 (Dense)** | `(None, 1)`      | 33             | - **Hidden Layer 2 to Output Layer**  
  - Parameters = **(32 + 1 Bias)** × **1 Neuron** = `33`.                                                   |

---

### Key Terms
- **Output Shape**: The shape of the data after passing through the layer. `None` represents the batch size, which can vary.
- **Param #**: The total number of trainable parameters (weights and biases) in the layer.

---

### Total Parameters
- **Trainable Parameters**: `2,945`  
  These are the weights and biases that will be updated during training.
- **Non-trainable Parameters**: `0`  
  These are fixed parameters (e.g., frozen weights) and do not update during training.

---
### Explanation of the Code

1. **`import tensorflow`**  
   - Importing the TensorFlow library to access its functionalities.

---

2. **Optimizer: `Adam`**  
   ```python
   opt = tensorflow.keras.optimizers.Adam(learning_rate=0.01)
   ```
   - **Adam** is an adaptive optimization algorithm that combines the benefits of:
     - **Momentum-based Gradient Descent** (for stability).
     - **Adaptive Learning Rates** (for efficient convergence).  
   - **Learning Rate**: `0.01` specifies how fast the model learns during training.  
   - This optimizer adjusts weights based on the gradients of the loss function.

---

3. **Loss Function: `BinaryCrossentropy`**  
   ```python
   loss = tensorflow.keras.losses.BinaryCrossentropy()
   ```
   - This loss function is used for **binary classification** problems.  
   - Measures the difference between the predicted probabilities and the actual class labels (0 or 1).  

   **Attributes of the Loss Function**:
   - `from_logits`: `False` means the model outputs probabilities (via the sigmoid function). If `True`, the model outputs raw logits.  
   - `label_smoothing`: Default is `0.0`. If set, it smooths the labels, making them less confident to prevent overfitting.  
   - `axis`: Specifies the axis for computation; default is `-1` (last axis).

---

4. **Output: `<LossFunctionWrapper>`**
   - A wrapper around the `binary_crossentropy` function, indicating it's ready to compute the loss during training.

---

### Why These Choices?

- **Adam**: Efficient and widely used for deep learning tasks. Works well with sparse data and large models.  
- **Binary Crossentropy**: Ideal for binary classification problems as it penalizes predictions that deviate from the actual labels.  
---

