from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from sklearn.feature_extraction import DictVectorizer
import argparse
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score

import time

def read_datashop_student_step(step_file, model_id=None):

    header = {v: i for i,v in enumerate(step_file.readline().rstrip().split('\t'))}
   
    kc_mods = [v[3:-1] for v in header if v[0:2] == "KC"]
    kc_mods.sort()

    model_id = 0
    model = "KC(%s)" % (kc_mods[model_id])
    opp = "Opportunity(%s)" % (kc_mods[model_id])

    kcs = []
    opps = []
    y = []
    stu = []
    student_label = []
    item_label = []
    mydict = {}
    allLabels = []

    for line in step_file:
        data = line.rstrip().split('\t')

        student = data[header['Anon Student Id']]
        steps = data[header['Problem Hierarchy']].split(",")
        
        for step in steps:
            key = student+step

            if key in mydict:
                mydict[key]=mydict[key]+1
            else:
                mydict[key]=1

        if len(data) <= header[model]:
            kc_labels = [kc for kc in steps]
            kcs.append({kc: 1 for kc in steps})
            opps.append({kc: int(mydict[student+kc]) for i,kc in enumerate(kc_labels)})
        else:
            kc_labels = [kc for kc in data[header[model]].split("~~") if kc != ""]
            allLabels.extend(kc_labels)

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(allLabels)
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("Finished saving vectorizer")

    true_k = 80
    model = MiniBatchKMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1)
    model.fit(X)
    print("Finished saving k-means model")
    '''
    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster")
        for ind in order_centroids[i, :10]:
            print (terms[ind])
        print
    '''
    joblib.dump(model, 'kcsKmeans.pkl')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process datashop file.')
    parser.add_argument('student_data', type=argparse.FileType('r'),
                        help="the student data file in datashop format")
    args = parser.parse_args()

    ssr_file = args.student_data
    read_datashop_student_step(ssr_file)