# **Car-Value-Decoding-Engine**

<p align="center">

  <img src="https://img.shields.io/badge/Project-Car--Value--Decoding--Engine-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Domain-Auto%20Pricing%20Intelligence-4CAF50?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Focus-Explainable%20Machine%20Learning-FFC107?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10-yellow?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-UI%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/Model-RandomForestRegressor-006400?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Explainability-Value%20Decomposition-9C27B0?style=for-the-badge" />

</p>
 
### *A Transparent Machine Learning System for Understanding Car Prices Like a Human Appraiser*
     
---

# Table of Contents

To help navigation, this README includes:

1. **Introduction**
2. **Why Explainable Pricing Matters**
3. **Dataset Source + Owner Explanation**
4. **Project Philosophy & Design Goals**
5. **High-Level Summary of the System**
6. **Architecture**
7. **Data Pipeline**
8. **Feature Engineering**
9. **Model Training**
10. **Value Decomposition Theory**
11. **Mathematical Formulation**
12. **Example: Decomposed Pricing Story**
13. **Command-Line Interface (CLI Guide)**
14. **Streamlit App Walkthrough (With Screenshots)**
15. **Explainability Tools**
16. **Dataset Insights**
17. **Future Enhancements**
18. **Real Business Use Cases**

---

# **1. Introduction**

Most machine learning projects stop at prediction.
This one does not.

**Car-Value-Decoding-Engine** goes beyond predicting car prices, it *explains* them.

It behaves like a **professional human appraiser**, breaking a car’s predicted price into meaningful and interpretable components:

* How much the **brand** adds
* How much **age** subtracts
* How **mileage** influences resale value
* How **engine size** affects base value
* How **condition** affects desirability
* How **fuel type** shifts market expectation
* How **transmission** affects demand

Instead of offering a mysterious ML-generated price, it offers:

> “This price makes sense because each part contributes fairly and logically.”

This is what **true explainable machine learning** looks like.

---

# **2. Why Explainable Pricing Matters**

Car pricing is not random, it is a function of measurable and emotional components.
But real-world ML pricing models often behave like opaque black boxes, making them:

* hard to trust
* hard to understand
* hard to debug
* hard to justify

This project solves that by making predictions **transparent, interpretable, and auditable**.

For businesses, explainability helps:

* build consumer trust
* improve regulatory acceptance
* assist negotiation
* support fairness & compliance
* enable strategic decisions

For developers, explainability helps:

* verify model logic
* detect data bias
* validate assumptions
* discover feature interactions
* avoid model hallucinations

---

# **3. Dataset Source**

The dataset comes from **Abdullah Meo (Kaggle)**:

[https://www.kaggle.com/datasets/abdullahmeo/car-price-pridiction](https://www.kaggle.com/datasets/abdullahmeo/car-price-pridiction)

### Why this dataset is valuable:

* It contains **realistic automotive attributes**
* It reflects genuine **market patterns**
* It includes both **numeric and categorical data**
* It features **non-linear interactions** (perfect for Random Forests)
* It enables creation of **interpretable pricing logic**

Every attribute connects directly to a real-world pricing factor:

| Dataset Column | Real-World Meaning        |
| -------------- | ------------------------- |
| Brand          | Reputation, luxury factor |
| Model          | Design variant, trims     |
| Engine Size    | Performance & spec level  |
| Mileage        | Wear and tear             |
| Year           | Depreciation factor       |
| Condition      | Market-readiness          |
| Transmission   | Market demand             |
| Fuel Type      | Running cost perception   |

This allows a rich, insightful ML system.

---

# **4. Project Philosophy & Design Principles**

This project follows five guiding principles:

### **Transparency over accuracy**

A pricing model must explain itself, not just output numbers.

### **Components reflect human reasoning**

Instead of predicting blindly, the model simulates:

* brand uplift
* mileage penalty
* aging depreciation
* spec-based value

### **Modularity**

Every part of the system, data, model, decomposition and UI is separated.

### **Real-world usability**

Designed to be used by:

* dealerships
* pricing analysts
* buyers/sellers
* researchers

### **Production-readiness**

Architecture mirrors real ML systems used in industry.

---

# **5. High-Level System Summary**

The system has four major components:

### **Data Pipeline**

Cleans raw input, engineers features, handles missingness, creates baseline rows.

### **Model Pipeline**

Trains a Random Forest with encoded categorical variables.

### **Value Decomposition Engine**

Simulates feature-group replacement to compute contributions.

### **Interactive Dashboard**

Lets users experiment with values and view explanations.

---

# **6. Architecture**

```
┌────────────────────┐
│  Raw Kaggle Data   │
└───────┬────────────┘
        │ data_prep.py
        ▼
┌────────────────────┐
│ Cleaned Dataset     │
│ + Engineered Fields │
└───────┬────────────┘
        │ features.py
        ▼
┌────────────────────┐
│ Preprocessing      │
│ (Scaling + OHE)    │
└───────┬────────────┘
        │ train_model.py
        ▼
┌────────────────────┐
│ Trained Model      │
│ + Baseline Stats   │
└───────┬────────────┘
        │ value_decomposition.py
        ▼
┌────────────────────┐
│ Decomposition      │
│ Explanation Engine │
└───────┬────────────┘
        │ app/app.py
        ▼
┌────────────────────┐
│ Streamlit App      │
│ (Decoder + Tools)  │
└────────────────────┘
```

Each piece is independently testable and replaceable.

---

# **7. Data Pipeline**

The pipeline handles messy real-world data gracefully.

### Steps:

### **1. Load raw CSV**

from `data/raw/car_price_prediction.csv`.

### **2. Clean string columns**

Trims whitespace, normalizes values.

### **3. Convert numeric fields**

Ensures Year, Mileage, Engine Size, and Price are valid numbers.

### **4. Remove impossible values**

(negative mileage, zero-age cars, etc.)

### **5. Compute engineered features:**

#### `car_age = reference_year - year`

Human-friendly depreciation measure.

#### `km_per_year = mileage / car_age`

Normalizes mileage intensity.

### **6. Handle missing categoricals**

Replaces missing values with `"Unknown"` to avoid model crashes.

---

# **8. Feature Engineering**

### **Numeric Features:**

* Car age
* Engine size
* Mileage
* Mileage intensity (km per year)

Why numeric scaling matters:
Random Forests don’t require scaling, but scaling helps decomposition consistency.

### **Categorical Features:**

* Brand
* Fuel type
* Transmission
* Condition

OneHotEncoder ensures that each category becomes its own dimension.

---

# **9. Model Training**

### Why RandomForestRegressor?

Because it offers:

* **robustness**
* **nonlinearity**
* **feature interaction learning**
* **stability under perturbations** (important for decomposition)
* **simplicity of deployment**

The model pipeline:

1. Preprocess numerics & categoricals
2. Fit Random Forest
3. Save model and training stats
4. Store baseline feature row for decomposition
5. Save metrics (MAE, RMSE, R²)

---

# **10. Value Decomposition Theory**

This module (`value_decomposition.py`) is the **heart** of the project.

It solves the hardest ML problem:

> "Given a prediction, how do we determine how much each factor contributed?"

### Why not SHAP?

SHAP is powerful, but:

* difficult to explain to non-experts
* computationally expensive
* unstable across model changes
* not grouped by human-friendly categories

Our method is:

* deterministic
* stable
* grouped by meaningful components
* mathematically sound
* domain-aligned

---

# **11. Mathematical Formulation**

Let:

* **X_base** = baseline feature vector
* **X_groupi** = baseline with group i replaced
* **f()** = trained model
* **price_base** = f(X_base)
* **price_groupi** = f(X_groupi)

Contribution of group *i*:

```
C_i = price_groupi, previous_price
```

Final predicted price:

```
P = price_base + Σ C_i
```

This ensures explainability is:

* **additive**
* **human-readable**
* **consistent**

---

# **12. Story Example of Decomposed Pricing**

Imagine a car predicted at $44,210.

The model might say:

* **+472** because it's Audi (brand premium)
* **–914** because it's older
* **+13,378** because of high-distribution mileage pattern
* **–15,497** because automatic transmission is penalized in this dataset
* **+255** because of good condition
* **+1657** from engine size

This creates a **transparent story**, not just a number.

---

# **13. CLI Guide, What Each Command Does**

### Prepare Data

```bash
python -m src.cli prepare-data
```

Cleans and caches the dataset.

### Train Model

```bash
python -m src.cli train
```

Trains the ML engine.

### Evaluate Model

```bash
python -m src.cli evaluate
```

Computes metrics and saves them.

### Decode Car

```bash
python -m src.cli decode-car --index n
```

Prints a detailed decomposition story.

---

# **14. Streamlit App, Full Walkthrough**

---

## Single Car Decoder

<img width="1262" height="472" alt="Screenshot 2025-12-09 at 16-54-15 Car Value Decoding Engine" src="https://github.com/user-attachments/assets/6dc677e5-437d-43c0-9f8b-099ca1ba179e" />

### What this tool does:

* Lets you simulate any car configuration
* Predicts the price
* Shows value breakdown
* Visualizes contributions
* Outputs explainable JSON

Great for:

* buyers
* sellers
* ML explainability demos
* pricing analysts

---

## Raw Decomposition

<img width="409" height="296" alt="Screenshot 2025-12-09 at 16-54-41 Car Value Decoding Engine" src="https://github.com/user-attachments/assets/b693d382-e3e7-42ef-849d-fd5530741415" />

Outputs full machine-readable breakdown.

---

## Decomposition Chart

<img width="1262" height="387" alt="Screenshot 2025-12-09 at 16-54-30 Car Value Decoding Engine" src="https://github.com/user-attachments/assets/56a7fb41-81fe-4db9-adce-dc8779abc8bd" />

Color-coded interpretation of increases and penalties.

---

## Compare Two Cars

<img width="657" height="325" alt="Screenshot 2025-12-09 at 16-56-05 Car Value Decoding Engine" src="https://github.com/user-attachments/assets/116c45e4-3ce2-486a-acf6-5d9324573fa3" />

Lets you compare:

* Specs
* Final predictions
* Contribution profiles

---

## Market Explorer

<img width="1099" height="447" alt="Screenshot 2025-12-09 at 16-55-49 Car Value Decoding Engine" src="https://github.com/user-attachments/assets/0456f0d2-ecd3-476e-90e0-1b15a0e23e8b" />

Visual dataset understanding.

---

# **15. Explainability Tools**

## Permutation Importance

Shows global influence of features.

## Bias Detection

Shows if the model systematically favors or penalizes:

* transmissions
* fuel types
* conditions
* brands

This ensures fairness in pricing systems.

---

# **16. Dataset Insights**

dataset exhibits interesting behaviors:

### Mileage sometimes increases price

Why?

* Some high-mileage cars belong to luxury brands
* Mileage clusters may correlate with engine size
* Market bias encoded in source data

### Automatic transmission decreases price

Possible reasons:

* Region prefers manual cars
* Dataset bias
* Engine-transmission pairs not uniformly distributed

Understanding these patterns is vital when applying ML to economics.

---

# **17. Future Enhancements**

### Add SHAP to complement deterministic decomposition

### Build full negotiation simulator

### Deploy model via FastAPI

### Add “depreciation forecast curve”

### Add “overpriced/underpriced car detector”

### Use LLM to generate natural-language valuation reports

### Integrate image-based classification for brand detection

### Add full AutoML pipeline

### Build model confidence interval estimator

---

# **18. Real Business Use Cases**

### Car dealerships

Price appraisal accuracy + explainability builds trust.

### Marketplaces

Show transparent pricing breakdown to buyers.

### Inspectors

Use decomposition to justify assessments.

### Researchers

Study economic patterns in vehicle markets.

### Pricing analysts

Validate pricing strategies using feature importances.

---

# How This System Generalizes to Other Domains**

This decomposition framework can also be used for:

* home pricing engines
* insurance risk scoring
* credit scoring
* e-commerce price optimization
* medical diagnosis attribution
* HR salary intelligence

Anywhere you need:

* prediction + explanation
* transparency + trust
