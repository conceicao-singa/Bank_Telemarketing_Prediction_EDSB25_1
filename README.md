# Bank_Telemarketing_Prediction_EDSB25_1

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Goals](Project-Goals)
3. [Models and Architecture](#Models-and-Architecture)
4. [Installation](#installation)
5. [Directory Structure](#directory-structure)
6. [Credits](#credits)
7. [License](#license)

## 1. Project Overview

A comprehensive machine‑learning project focused on predicting whether a customer will subscribe to a term deposit following a bank’s telemarketing campaign, with strong emphasis on reducing false negatives, extracting campaign insights, and ensuring robust model evaluation. The project addresses this challenge through a structured approach that includes understanding customer behavior, preparing clean and leakage‑free data, engineering meaningful features, and developing models that generalize reliably in real‑world campaign environments.

## 2. **Project Goals**

* Build predictive models to classify customer subscription likelihood.

* Understand customer behavior using deep EDA.

* Engineer domain‑specific features.

* Optimize models through hyperparameter tuning.

* Minimize false negatives, improving telemarketing efficiency.

* Produce actionable insights for strategic decision‑making.

## 4. Installation

### 4.1. Prerequisites
- Python 3.12 +

### 4.2. Steps

- **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/Bank_Telemarketing_Prediction_EDSB25_1.git
   cd Bank_Telemarketing_Prediction_EDSB25_1
    ```
## 5. Directory Structure

- **Data Directory** (`Bank_Telemarketing_Prediction_EDSB25_1/data`)
  - **`data_prep.py`**: Includes functions for preprocessing the data.

- **Models Directory** (`Bank_Telemarketing_Prediction_EDSB25_1/models`)
  - **`training.py`**: Defines all trained models, including their architectures, and handles the processes of building, and training.


- **Utilities Directory** (`Bank_Telemarketing_Prediction_EDSB25_1/scripts/utils`)
  - **`utils.py`**: Loading and model saving functions and other essentials

- **Notebooks Directory** (`Bank_Telemarketing_Prediction_EDSB25_1/notebooks`)
  - **`data_preparation.ipynb`**: Notebook for data exploration and preprocessing.
  - **`training.ipynb`**: Notebook for training the models, making predictions on test data and evaluating the final models.
  - **`evaluation.ipynb`**: Notebook for SHAP and Feature importance analysis 


- **Scripts Directory** (`Bank_Telemarketing_Prediction_EDSB25_1/scripts`)
  - Contains various `.py` files for testing functions and methods before integrating them into the notebooks.

## 6. Credits

- **EDSB25_1**

- **Contributors:**
  * Conceição Singa (20231109)
  * Gonçalo Abel Gonçalves (20242016)
  * João André Neves (20241487)
  * Md Haque (20241356)

## 7. License
This project is licensed under the [Apache] http://www.apache.org/licenses/