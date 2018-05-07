from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from sklearn.feature_extraction import DictVectorizer
import argparse
import numpy as np
from scipy.sparse import hstack
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.sparse import hstack
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

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
    lineas=0
    student_label = []
    item_label = []
    mydict = {}
    clf = joblib.load('kcsKmeans.pkl')
    vectorizer = joblib.load('vectorizer.pkl') 

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
            kc_labels = [clf.predict(vectorizer.transform([kc]))[0] for kc in data[header[model]].split("~~") if kc != ""]

            if not kc_labels:
                continue

            kcs.append({kc: 1 for kc in kc_labels})
            opps.append({})
            for kc in kc_labels:
                key = student+str(kc)

                if key in mydict:
                    mydict[key]=mydict[key]+1
                else:
                    mydict[key]=1

                opps[-1][kc]=mydict[key]-1

        if data[header['Correct First Attempt']] == "1":
            y.append(1)
        else:
            y.append(0)

        student = data[header['Anon Student Id']]
        stu.append({student: 1})
        student_label.append(student)

        item = data[header['Problem Name']] + "##" + data[header['Step Name']]
        item_label.append(item)
        lineas+=1
        print(lineas)

    return (kcs, opps, y, stu, student_label, item_label)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process datashop file.')
    parser.add_argument('student_data', type=argparse.FileType('r'),
                        help="the student data file in datashop format")
    args = parser.parse_args()

    ssr_file = args.student_data
    kcs, opps, y, stu, student_label, item_label = read_datashop_student_step(ssr_file)
    print("Finished processing")
    sv = DictVectorizer(sparse=True)
    qv = DictVectorizer(sparse=True)
    ov = DictVectorizer(sparse=True)

    S = sv.fit_transform(stu)
    print("Number of students :%d"%(S.shape[1]))
    Q = qv.fit_transform(kcs)
    print("Number of KCS :%d"%(Q.shape[1]))
    O = ov.fit_transform(opps)
    max_abs_scaler = preprocessing.MaxAbsScaler()
    O = max_abs_scaler.fit_transform(O)
    print("Number of KCS operations :%d"%(O.shape[1]))
    print("Finished Fit Transform")

    X = sparse.hstack((S,Q,O))
    print("Finished hstack")

    y = np.array(y)

    entrenamiento = []
    lr= LogisticRegression()
    start_time = time.time()
    lr.fit(X,y)
    print("--- Took %s seconds to fit data ---" % (time.time() - start_time))

    train_pred = lr.predict(X)
    print("Training Correct :%2.2f"%(np.mean(train_pred == y) * 100))
    errorEntrenamiento = np.sqrt(mean_squared_error(y, train_pred))
    print("RMSE :%0.4f" % (errorEntrenamiento))
