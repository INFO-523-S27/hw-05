---
name: HW-05 Feedback
about: Feedback rubric for HW-05 (Clustering with OWID Energy Data)
title: "HW-05 Feedback"
labels: "grading"
assignees: []
---

Dear [STUDENT GITHUB USERNAME],  
Below is your feedback for **HW-05: Clustering with OWID Energy Data**. Please review carefully and reach out during office hours if you have questions.

---

# ✅ Final Score: **[_ / 100]**

---

## Part 1: Data Loading, Cleaning & Feature Prep (**30 points**)

### 1. `load_energy_data` (**5 points**)
- [ ] Correctly loads CSV using `pd.read_csv`
- [ ] Returns a DataFrame
- Score: **[_ / 5]**
- Feedback:

---

### 2. `initial_summary` (**5 points**)
- [ ] Correct row and column counts
- [ ] Correct separation of numeric vs categorical columns
- [ ] No hardcoding
- Score: **[_ / 5]**
- Feedback:

---

### 3. `clean_energy_data` (**10 points**)
- [ ] Drops duplicates
- [ ] Removes missing values correctly using `subset_cols`
- [ ] Does NOT modify original DataFrame in-place
- Score: **[_ / 10]**
- Feedback:

---

### 4. `filter_year_range` (**5 points**)
- [ ] Correct inclusive filtering
- [ ] No mutation of original DataFrame
- Score: **[_ / 5]**
- Feedback:

---

### 5. `select_clustering_features` (**5 points**)
- [ ] Returns only selected columns
- [ ] Preserves row count
- [ ] Does NOT modify input DataFrame
- Score: **[_ / 5]**
- Feedback:

---

## Part 2: Standardization & K-Means (**30 points**)

### 6. `standardize_features` (**10 points**)
- [ ] Mean ≈ 0 for all columns
- [ ] Std ≈ 1 for all columns
- [ ] Correct return of `(df_scaled, means, stds)`
- Score: **[_ / 10]**
- Feedback:

---

### 7. `compute_calinski_harabasz_scores` (**10 points**)
- [ ] Correct loop over cluster_range
- [ ] Uses sklearn CH score correctly
- [ ] Returns dictionary with correct keys and values
- Score: **[_ / 10]**
- Feedback:

---

### 8. `run_kmeans` (**10 points**)
- [ ] Correct label shape
- [ ] Correct cluster center shape
- [ ] Deterministic with `random_state`
- Score: **[_ / 10]**
- Feedback:

---

## Part 3: Hierarchical Clustering, GMM & Outliers (**25 points**)

### 9. `run_hierarchical_clustering` (**5 points**)
- [ ] Correct number of clusters returned
- [ ] Correct linkage handling
- Score: **[_ / 5]**
- Feedback:

---

### 10. `fit_gmm` + `gmm_cluster_labels` (**10 points**)
- [ ] GMM properly fit
- [ ] Correct label prediction
- [ ] Deterministic with `random_state`
- Score: **[_ / 10]**
- Feedback:

---

### 11. `gmm_outlier_mask` (**10 points**)
- [ ] Correct probability-based outlier detection
- [ ] Logical thresholding behavior
- [ ] Returns boolean mask
- Score: **[_ / 10]**
- Feedback:

---

## Part 4: Cluster Validation (**10 points**)

### 12. `silhouette_for_labels` (**10 points**)
- [ ] Correct use of `silhouette_score`
- [ ] Handles single-cluster case by returning `np.nan`
- Score: **[_ / 10]**
- Feedback:

---

## Code Quality & Professionalism (**5 points**)

- [ ] Clean, readable logic
- [ ] No hardcoded values
- [ ] No unnecessary duplication
- [ ] Good variable naming
- Score: **[_ / 5]**
- Feedback:

---

# ✅ Automated Checks

- **Pytest:** ✅ Pass / ❌ Fail  
- **Pylint:** ✅ Pass / ❌ Fail  

---

# ⚠️ Deductions Checklist (–5 each if repeated)

- [ ] Modified `test_ds.py`
- [ ] Modified `requirements.txt`
- [ ] Renamed required functions
- [ ] Deleted or modified docstrings
- [ ] Merged the `feedback` pull request
- [ ] Hardcoded values to match test cases

---

# 📝 Final Instructor Comments

