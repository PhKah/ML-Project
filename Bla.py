import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.stats import pearsonr

a = [9,2,8,9,1,1,5,1,5,6,9,1,10,10,9,8,1]
b = [9,2,1,10,9,9,3,3,10,8,10,9,9,8,10,10,3]
tba = 0
tbb = 0
for i in range(0,len(a) - 1):
    tba += a[i]
    tbb += b[i]
tba /= len(a)
tbb /= len(a)
tu = 0
mau1 = mau2 = 0
for i in range(0, len(a) - 1):
    tu += (a[i] - tba) * (b[i] - tbb)
    mau1 += pow(a[i] - tba, 2)
    mau2 += pow(b[i] - tbb, 2)
print(tu/pow(mau1 * mau2,1/2))

r_score, _ = pearsonr(a, b)

print(f"Hệ số Pearson dùng SciPy: {r_score:.4f}")