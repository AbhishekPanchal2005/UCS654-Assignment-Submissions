# Statistical Modeling of Transformed NO₂ Data using Maximum Likelihood Estimation

This repository contains a Python implementation for modeling a probability density function of **transformed NO₂ (Nitrogen Dioxide)** air quality data using **Maximum Likelihood Estimation (MLE)**.

The solution is **roll-number specific**, dataset-driven, and implemented using standard scientific Python libraries.

---

## 📌 Objective

The objectives of this assignment are:

- To apply a roll-number–dependent non-linear transformation to NO₂ data  
- To model the transformed data using a Gaussian-like probability density function  
- To estimate the parameters **μ (mean)**, **λ (lambda)**, and **c (normalization constant)** using MLE  

---

## 📂 Dataset

- **Dataset:** India Air Quality Dataset  
- **File Used:** `data.csv`  
- **Feature Selected:** `no2`  

Only the `no2` column is used. Missing values are removed before analysis.

---

## 🔢 Roll Number Based Parameters

University Roll Number: r = 102317167

The transformation constants are defined as:

\[
a_r = 0.05 \times (r \bmod 7)
\]

\[
b_r = 0.3 \times (r \bmod 5 + 1)
\]

Computed values:

- `a_r = 0.05`
- `b_r = 0.9`

---

## 🔁 Data Transformation

Let `x` denote the original NO₂ concentration values.

The transformed variable `z` is defined as:

\[
z = x + a_r \cdot \sin(b_r \cdot x)
\]

This transformation introduces a smooth non-linear perturbation while preserving the overall structure of the data.

---

## 📐 Probability Density Function

The transformed data is modeled using the following probability density function:

\[
\hat{p}(z) = c \cdot e^{-\lambda (z - \mu)^2}
\]

Where:
- **μ** is the mean of the transformed data  
- **λ** controls the spread of the distribution  
- **c** is the normalization constant  

---

## 📊 Parameter Estimation using MLE

Using Maximum Likelihood Estimation:

\[
\mu = \frac{1}{n} \sum_{i=1}^{n} z_i
\]

\[
\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (z_i - \mu)^2
\]

\[
\lambda = \frac{1}{2\sigma^2}
\]

\[
c = \frac{1}{\sigma \sqrt{2\pi}}
\]

---

## 🛠️ Implementation

### Requirements
- Python 3.x  
- pandas  
- numpy  

### Python Code

```python
import math
import pandas as pd
import numpy as np

r = 102317167

data = pd.read_csv(r"C:\Users\ASUS\Downloads\data.csv", encoding="latin1")

no2 = data["no2"].dropna()

Ar = 0.05 * (r % 7)
Br = 0.3 * ((r % 5) + 1)

x = no2.values
z = x + Ar * np.sin(Br * x)

mean = np.mean(z)
var = np.var(z)
std = np.sqrt(var)

lambda1 = 1.0 / (2.0 * var)
c = 1.0 / (std * np.sqrt(2.0 * np.pi))

print(f"Lambda  : {lambda1}")
print(f"Mu      : {mean}")
print(f"c       : {c}")
```

## ✅ Output

After executing the Python program on the provided dataset, the following parameter values were obtained for the modeled probability density function:

- **μ (Mean)** ≈ 25.80  
- **λ (Lambda)** ≈ 0.00146  
- **c (Normalization Constant)** ≈ 0.02155  

These values represent the Maximum Likelihood Estimates of the parameters for the transformed NO₂ data.

> Note: Slight variations in the output may occur depending on floating-point precision or dataset preprocessing, which is expected in real-world data analysis.

---

## 🧠 Key Observations

- The roll-number–dependent sinusoidal transformation introduces controlled non-linearity into the NO₂ data without significantly altering its overall distribution.
- The transformed data follows a Gaussian-like pattern, making it suitable for modeling using a continuous probability density function.
- Maximum Likelihood Estimation provides stable and interpretable parameter estimates for real-world environmental data.
- The estimated value of λ indicates a relatively wide spread of the transformed NO₂ distribution.
- The step-wise implementation improves clarity and makes the solution easy to understand, reproduce, and explain during evaluation or viva.

