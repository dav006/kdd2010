# kdd2010

Python code to solve the KDD Cup 2010 using different models:

1. af.py <file>
Additive Factors Model, missing kcs are filled with Problem Hierchym, Sparse Data

2. afKMeans.py <file>
Similar to AFM, but instead of using KCs we use kmeans to generate more abstract KCs. To generate the centroids run kmeansKCS.py <file>

3. featureCombination.py Based on http://pslcdatashop.org/KDDCup/workshop/papers/kdd2010ntu.pdf that is using :
  CFARs for student name, step name, problem name, KC, (problem name, step name), (student name, problem name), (student name, unit name) and (student name, KC)
    •Eight CFARs
    •Two scaled numerical features for opportunity and problem view.
