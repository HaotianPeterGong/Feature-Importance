# Feature-Importance
This repo explores various feature importance and selection techniques.

Feature importance is widely used to interpret machine learning models. Here, I explore commonly used techniques like:

1. Spearman's rank correlation coefficient
2. Principle Component Analysis
3. Minimal Redundancy Maximal Relevance (mRMR)
4. Permutation Importance
5. Drop Column Importance
6. Shap Importance

Next, I train models using 1,2,3.. most important feature(s), as predicted by each of above techniques, and compare losses to identify which technique is more efficient. I have also implemented a simple Automatic Feature Selection technique which iteratively finds the best (and simpler) model.

Finally, I discuss two statistical technqiues that give a deeper understanding of the generated scores:

1. Variance in Feature Importances
2. Empirical P-Values

Thanks to [Prof. Terence Parr](https://github.com/parrtfor) his guidance and support in this school project.
