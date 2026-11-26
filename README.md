# HW-05: Unsupervised Learning & Clustering with OWID Energy Data

## Objectives

The learning objectives of this assignment are to:

1. Perform **unsupervised learning** using real-world global energy data.
2. Apply **data cleaning, filtering, and feature selection** to prepare data for clustering.
3. Implement **feature standardization** for distance-based models.
4. Apply and compare multiple **clustering algorithms**, including:
   - K-Means
   - Hierarchical (Agglomerative) Clustering
   - Gaussian Mixture Models (GMM)
5. Evaluate clustering quality using:
   - **Calinski–Harabasz Score**
   - **Silhouette Score**
6. Perform **probability-based outlier detection** using GMMs.
7. Interpret clusters in the context of:
   - Energy systems
   - Decarbonization
   - Global inequality and policy
8. Write **clean, testable, modular Python code**.
9. Follow professional **software quality standards** using automated testing and linting.

---

## Dataset

This assignment uses the **Our World in Data (OWID) global energy dataset**:

```

data/owid-energy.csv

````

This dataset includes country-level energy indicators such as:

- Fossil energy per capita
- Renewable energy per capita
- Share of low-carbon energy
- Electricity production sources
- Population and GDP (for some years)

You will use a subset of these features to identify **latent energy system patterns across countries and time**.

---

## Setup Your Environment

Before starting, you must have the following installed:

- [Python 3.9+](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installation/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)
- [pytest](https://docs.pytest.org/)
- [pylint](https://pylint.pycqa.org/)

---

### Installing Dependencies

Install all required packages with:

```bash
pip install -r requirements.txt
````

---

## Check Out the Starter Code

Once you accept the assignment, GitHub Classroom will create your repository at:

```
https://github.com/INFO-511-S25/hw-05-clustering-<your-username>
```

Your repository includes:

* `ds.py` — Your implementation file ✅
* `test_ds.py` — The autograder ❌ (DO NOT MODIFY)
* `data/owid-energy.csv` — The dataset
* `hw-05-clustering.ipynb` — Analysis notebook
* `requirements.txt`
* `.gitignore`

A **`feedback` branch and pull request** will also be created automatically for grading.

---

## Codespaces (Recommended)

**Codespaces are enabled.**

To launch:

1. Click the green **Code** button
2. Select the **Codespaces** tab
3. Click **Create codespace on main**

This gives you a fully configured Python environment with all dependencies pre-installed.

---

## Local Machine (Alternative)

If working locally:

```bash
git clone https://github.com/INFO-511-S25/hw-05-clustering-<your-username>.git
cd hw-05-clustering-<your-username>
```

All work must be done in the `main` branch.

---

## File Descriptions

### ✅ `ds.py` (**ONLY FILE YOU MODIFY**)

This file contains all required functions for the HW-05 clustering workflow.

### **Part 1: Data Preparation**

* `load_energy_data`
* `initial_summary`
* `clean_energy_data`
* `filter_year_range`
* `select_clustering_features`

---

### **Part 2: Feature Standardization & K-Means**

* `standardize_features`
* `compute_calinski_harabasz_scores`
* `run_kmeans`

---

### **Part 3: Hierarchical & GMM Clustering**

* `run_hierarchical_clustering`
* `fit_gmm`
* `gmm_cluster_labels`
* `gmm_outlier_mask`

---

### **Part 4: Cluster Validation**

* `silhouette_for_labels`

Each function has a **strict docstring defining expected behavior**.

---

### ✅ `test_ds.py` (**DO NOT MODIFY**)

* Runs all automated tests for your clustering pipeline.
* Passing locally means your GitHub autograder will pass.

---

### ✅ `hw-05-clustering.ipynb`

You will:

* Explore the dataset
* Visualize standardized features
* Run and compare all clustering models
* Interpret clusters & outliers
* Reflect on ethical and policy implications

---

### ✅ `.gitignore`

Prevents committing:

* Virtual environments
* Cache files
* OS junk
* Large generated artifacts

---

## Instructions

1. Open `ds.py`
2. Implement **every required function**
3. Do **NOT**:

   * Rename functions
   * Modify `test_ds.py`
   * Modify `requirements.txt`
   * Modify docstrings

---

## Running Tests

Run the full test suite:

```bash
pytest
```

Expected output when correct:

```plaintext
============================= test session starts ==============================
collected XX items

test_ds.py .................                               [100%]

============================== XX passed in 0.10s ===============================
```

---

## Code Quality Checks (Required)

Run pylint:

```bash
pylint ds.py
```

A perfect score:

```plaintext
--------------------------------------------------------------------
Your code has been rated at 10.00/10
```

---

## Submission Instructions

1. Commit your completed `ds.py`:

```bash
git add ds.py
git commit -m "Completed HW-05 Clustering"
git push
```

2. **DO NOT:**

* Modify `test_ds.py`
* Modify `requirements.txt`
* Merge the `feedback` pull request

3. Confirm ✅ green GitHub Actions checks.

---

## Grading

### ✅ Points Allocation

* **80 points** — All automated correctness (`pytest`) tests pass
* **10 points** — All automated style (`pylint`) checks pass
* **10 points** — Overall code quality:

  * Clean logic
  * No hardcoding
  * Sensible variable names
  * Proper use of NumPy, Pandas, and scikit-learn
  * Clear modular structure

---

### ❌ Deductions (–5 points each)

* Modifying any file other than `ds.py`
* Renaming required functions
* Deleting or altering docstrings
* Hardcoding values to pass tests
* Merging the `feedback` pull request

---

## Why This Assignment Matters

This assignment mirrors how clustering is used in:

* Energy systems analysis
* Climate policy modeling
* Market segmentation
* Global development analytics
* Infrastructure transition planning

It also teaches:

* How unsupervised learning can **reveal structure without labels**
* Why validation and interpretation matter just as much as model fitting
* How misleading clusters can be without domain awareness

---

✅ **This assignment builds directly toward real-world analytics, sustainability modeling, and future AI system design.**

Good luck—and enjoy discovering global energy structure through unsupervised learning! ⚡🌍📊
