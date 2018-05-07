# kdd2010

Python code to solve the KDD Cup 2010 using different models:

1. af.py <file>
Additive Factors Model, missing kcs are filled with Problem Hierchym, Sparse Data

2. afKMeans.py <file>
Similar to AFM, but instead of using KCs we use kmeans to generate more abstract KCs. To generate the centroids run kmeansKCS.py <file>
