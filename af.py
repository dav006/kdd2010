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
    problemV =[]
    y = []
    stu = []
    student_label = []
    item_label = []
    mydict = {}
    emptyKCs = 0

    for line in step_file:
        data = line.rstrip().split('\t')

        student = data[header['Anon Student Id']]
        pH = data[header['Problem Hierarchy']].split(",")
        problemView = data[header['Problem View']]

        problemV.append(problemView)
        
        for problemH in pH:
            key = student+problemH

            if key in mydict:
                mydict[key]=mydict[key]+1
            else:
                mydict[key]=1

        if len(data) <= header[model]:
            emptyKCs+=1
            kc_labels = [kc for kc in pH]
            kcs.append({kc: 1 for kc in pH})
            opps.append({kc: int(mydict[student+kc]) for i,kc in enumerate(kc_labels)})
        else:
            kc_labels = [kc for kc in data[header[model]].split("~~") if kc != ""]

            if not kc_labels:
                continue

            kcs.append({kc: 1 for kc in kc_labels})
            for kc in pH:
                kcs[-1][kc]=1

            kc_opps = [o for o in data[header[opp]].split("~~") if o != ""]
            opps.append({kc: int(kc_opps[i])-1 for i,kc in enumerate(kc_labels)})
            for kc in pH:
                opps[-1][kc]=mydict[student+kc]-1

        if data[header['Correct First Attempt']] == "1":
            y.append(1)
        else:
            y.append(0)

        student = data[header['Anon Student Id']]
        stu.append({student: 1})
        student_label.append(student)

        item = data[header['Problem Name']] + "##" + data[header['Step Name']]
        item_label.append(item)

    return (kcs, opps, y, stu, student_label, item_label, problemV)

if __name__ == "__main__":

    file = 'algebra_2005_2006_train.txt'
    ssr_file = open(file,'r')

    kcs, opps, y, stu, student_label, item_label, problemV = read_datashop_student_step(ssr_file)

    ##Data set de validacion
    file = 'algebra_2005_2006_master.txt'
    ssr_file = open(file,'r')

    kcsv, oppsv, yv, stuv, student_labelv, item_labelv, problemVv = read_datashop_student_step(ssr_file)

    print("Finished processing")
    sv = DictVectorizer(sparse=True)
    qv = DictVectorizer(sparse=True)
    ov = DictVectorizer(sparse=True)

    sv.fit(np.append(stu,stuv))
    qv.fit(np.append(kcs,kcsv))
    ov.fit(np.append(opps,oppsv))

    S = sv.transform(stu)
    print("Number of students :%d"%(S.shape[1]))
    Q = qv.transform(kcs)
    print("Number of KCS :%d"%(Q.shape[1]))
    O = ov.transform(opps)
    max_abs_scaler = preprocessing.MaxAbsScaler()
    O = max_abs_scaler.fit_transform(O)
    print("Number of KCS operations :%d"%(O.shape[1]))

    problemV = np.asarray(problemV)
    problemV = problemV[:,np.newaxis]
    max_abs_scaler = preprocessing.MaxAbsScaler()
    PV = max_abs_scaler.fit_transform(problemV)

    print("Finished Fit Transform")

    X = sparse.hstack((S,Q,O,PV))
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
    print("Training RMSE :%0.4f" % (errorEntrenamiento))

    print("Now processing validation data")    
    SV = sv.transform(stuv)
    QV = qv.transform(kcsv)
    OV = ov.transform(oppsv)
    max_abs_scaler = preprocessing.MaxAbsScaler()
    OV = max_abs_scaler.fit_transform(OV)
    print("Number of KCS operations :%d"%(OV.shape[1]))

    problemV = np.asarray(problemVv)
    problemV = problemV[:,np.newaxis]
    max_abs_scaler = preprocessing.MaxAbsScaler()
    PVV = max_abs_scaler.fit_transform(problemV)

    print("Finished Fit Transform")

    X = sparse.hstack((SV,QV,OV,PVV))
    print("Finished hstack")

    y = np.array(yv)

    test_pred = lr.predict(X)
    print("Test Correct :%2.2f"%(np.mean(test_pred == y) * 100))
    errorVal = np.sqrt(mean_squared_error(y, test_pred))
    print("Test RMSE :%0.4f" % (errorVal))

