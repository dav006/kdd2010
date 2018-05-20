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

totalRows = 809694
totalRowsVal = 3967
studentDictCount = {}
stepNameDictCount = {}
problemNameDictCount = {}
kcDictCount = {}

problemStepDictCount = {}
studentProblemDictCount = {}
studentUnitDictCount = {}
studentKCDictCount = {}

studentDict = {}
stepNameDict = {}
problemNameDict = {}
kcDict = {}

problemStepDict = {}
studentProblemDict = {}
studentUnitDict = {}
studentKCDict = {}

def read_master_file(model_id=None):
    file = 'algebra_2005_2006_master.txt'
    step_file = open(file,'r')

    header = {v: i for i,v in enumerate(step_file.readline().rstrip().split('\t'))}
   
    kc_mods = [v[3:-1] for v in header if v[0:2] == "KC"]
    kc_mods.sort()

    model_id = 0
    model = "KC(%s)" % (kc_mods[model_id])
    opp = "Opportunity(%s)" % (kc_mods[model_id])

    opps = np.zeros((totalRowsVal,1))
    problemV =np.zeros((totalRowsVal,1))
    y = []

    emptyKCs = 0
    indexRow = 0

    stu = np.zeros((totalRowsVal,1))
    stepName = np.zeros((totalRowsVal,1))
    problemName = np.zeros((totalRowsVal,1))
    kcs = np.zeros((totalRowsVal,1))

    problemStep = np.zeros((totalRowsVal,1))
    studentProblem = np.zeros((totalRowsVal,1))
    studentUnit = np.zeros((totalRowsVal,1))
    studentKC = np.zeros((totalRowsVal,1))

    for line in step_file:
        data = line.rstrip().split('\t')

        student = data[header['Anon Student Id']]
        pN = data[header['Problem Name']]
        problemView = data[header['Problem View']]
        step = data[header['Step Name']]
        unit = data[header['Problem Hierarchy']]
        student = data[header['Anon Student Id']]

        if data[header['Correct First Attempt']] == "1":
            y.append(1)
        else:
            y.append(0)

        problemV[indexRow,0]=problemView

        if student not in studentDict:
            studentDict[student]=0.0
            studentDictCount[student] = 1.0
        stu[indexRow,0]= (studentDict[student]) / (studentDictCount[student])

        if step not in stepNameDict:
            stepNameDict[step]=0.0
            stepNameDictCount[step]=1.0
        stepName[indexRow,0]=(stepNameDict[step])/(stepNameDictCount[step])

        if pN not in problemNameDict:
            problemNameDict[pN]=0.0
            problemNameDictCount[pN]=1.0
        problemName[indexRow,0]=(problemNameDict[pN])/(problemNameDictCount[pN])

        key =pN+step
        if key not in problemStepDict:
            problemStepDict[key]=0.0
            problemStepDictCount[key]=1.0
        problemStep[indexRow,0]=(problemStepDict[key])/(problemStepDictCount[key])

        key =student+pN
        if key not in studentProblemDict:
            studentProblemDict[key]=0.0
            studentProblemDictCount[key]=1.0
        studentProblem[indexRow,0]=(studentProblemDict[key])/(studentProblemDictCount[key])

        key =student+unit
        if key not in studentUnitDict:
            studentUnitDict[key]=0.0
            studentUnitDictCount[key]=1.0
        studentProblem[indexRow,0]=(studentUnitDict[key])/(studentUnitDictCount[key])

        if len(data) <= header[model]:
            emptyKCs+=1
            kcs[indexRow,0]=0.0
            studentKC[indexRow,0]=0.0
            opps[indexRow,0]=0.0
        else:
            kc_label = data[header[model]]
            if kc_label not in kcDict:
                kcDict[kc_label]=0.0
                kcDictCount[kc_label]=1.0
            kcs[indexRow,0]=(kcDict[kc_label])/(kcDictCount[kc_label])

            kc_opps = data[header[opp]].split("~~")
            sumOpps = 0.0
            for kc_opp in kc_opps:
                sumOpps+=float(kc_opp)
            opps[indexRow,0]=sumOpps

            key =student+kc_label
            if key not in studentKCDict:
                studentKCDict[key]=0.0
                studentKCDictCount[key]=1.0
            studentKC[indexRow,0]=(studentKCDict[key])/(studentKCDictCount[key])
        indexRow+=1

    print("Empty KCs : %d"%(emptyKCs))

    return (stu,stepName,problemName,kcs,problemStep,studentProblem,studentUnit,studentKC,problemV,opps,y)

def read_datashop_student_step(step_file, model_id=None):

    header = {v: i for i,v in enumerate(step_file.readline().rstrip().split('\t'))}
   
    kc_mods = [v[3:-1] for v in header if v[0:2] == "KC"]
    kc_mods.sort()

    model_id = 0
    model = "KC(%s)" % (kc_mods[model_id])
    opp = "Opportunity(%s)" % (kc_mods[model_id])
    totalRows = 809694

    opps = np.zeros((totalRows,1))
    problemV =np.zeros((totalRows,1))
    y = []

    emptyKCs = 0
    indexRow = 0

    stu = np.zeros((totalRows,1))
    stepName = np.zeros((totalRows,1))
    problemName = np.zeros((totalRows,1))
    kcs = np.zeros((totalRows,1))

    problemStep = np.zeros((totalRows,1))
    studentProblem = np.zeros((totalRows,1))
    studentUnit = np.zeros((totalRows,1))
    studentKC = np.zeros((totalRows,1))

    for line in step_file:
        data = line.rstrip().split('\t')

        student = data[header['Anon Student Id']]
        pN = data[header['Problem Name']]
        problemView = data[header['Problem View']]
        step = data[header['Step Name']]
        unit = data[header['Problem Hierarchy']]
        student = data[header['Anon Student Id']]

        correct = 0
        if data[header['Correct First Attempt']] == "1":
            y.append(1)
            correct = 1
        else:
            y.append(0)

        problemV[indexRow,0]=problemView

        if student in studentDict:
            studentDict[student]=studentDict[student]+correct
            studentDictCount[student]+=1.0
        else:
            studentDict[student]=correct
            studentDictCount[student] = 1.0
        stu[indexRow,0]= (studentDict[student]-correct) / (studentDictCount[student])

        if step in stepNameDict:
            stepNameDict[step]=stepNameDict[step]+correct
            stepNameDictCount[step]=stepNameDictCount[step]+1
        else:
            stepNameDict[step]=correct
            stepNameDictCount[step]=1.0
        stepName[indexRow,0]=(stepNameDict[step]-correct)/(stepNameDictCount[step])

        if pN in problemNameDict:
            problemNameDict[pN]=problemNameDict[pN]+correct
            problemNameDictCount[pN]=problemNameDictCount[pN]+1
        else:
            problemNameDict[pN]=correct
            problemNameDictCount[pN]=1.0
        problemName[indexRow,0]=(problemNameDict[pN]-correct)/(problemNameDictCount[pN])

        key =pN+step
        if key in problemStepDict:
            problemStepDict[key]=problemStepDict[key]+correct
            problemStepDictCount[key]=problemStepDict[key]+1
        else:
            problemStepDict[key]=correct
            problemStepDictCount[key]=1.0
        problemStep[indexRow,0]=(problemStepDict[key]-correct)/(problemStepDictCount[key])

        key =student+pN
        if key in studentProblemDict:
            studentProblemDict[key]=studentProblemDict[key]+correct
            studentProblemDictCount[key]=studentProblemDictCount[key]+1
        else:
            studentProblemDict[key]=correct
            studentProblemDictCount[key]=1.0
        studentProblem[indexRow,0]=(studentProblemDict[key]-correct)/(studentProblemDictCount[key])

        key =student+unit
        if key in studentUnitDict:
            studentUnitDict[key]=studentUnitDict[key]+correct
            studentUnitDictCount[key]=studentUnitDictCount[key]+1
        else:
            studentUnitDict[key]=correct
            studentUnitDictCount[key]=1.0
        studentProblem[indexRow,0]=(studentUnitDict[key]-correct)/(studentUnitDictCount[key])

        if len(data) <= header[model]:
            emptyKCs+=1
            kcs[indexRow,0]=0.0
            studentKC[indexRow,0]=0.0
            opps[indexRow,0]=0.0
        else:
            kc_label = data[header[model]]
            if kc_label in kcDict:
                kcDict[kc_label]=kcDict[kc_label]+correct
                kcDictCount[kc_label]=kcDictCount[kc_label]+correct
            else:
                kcDict[kc_label]=correct
                kcDictCount[kc_label]=1.0
            kcs[indexRow,0]=(kcDict[kc_label]-correct)/(kcDictCount[kc_label])

            kc_opps = data[header[opp]].split("~~")
            sumOpps = 0.0
            for kc_opp in kc_opps:
                sumOpps+=float(kc_opp)
            opps[indexRow,0]=sumOpps

            key =student+kc_label
            if key in studentKCDict:
                studentKCDict[key]=studentKCDict[key]+correct
                studentKCDictCount[key]=studentKCDictCount[key]+1
            else:
                studentKCDict[key]=correct
                studentKCDictCount[key]=1.0
            studentKC[indexRow,0]=(studentKCDict[key]-correct)/(studentKCDictCount[key])
        indexRow+=1

    print("Empty KCs : %d"%(emptyKCs))

    return (stu,stepName,problemName,kcs,problemStep,studentProblem,studentUnit,studentKC,problemV,opps,y)

if __name__ == "__main__":

    file = 'algebra_2005_2006_train.txt'
    ssr_file = open(file,'r')

    stu,stepName,problemName,kc,problemStep,studentProblem,studentUnit,studentKC,problemV,opps,y = read_datashop_student_step(ssr_file)
    print("Finished processing")

    max_abs_scaler = preprocessing.MaxAbsScaler()
    opps = max_abs_scaler.fit_transform(opps)
    
    max_abs_scaler = preprocessing.MaxAbsScaler()
    problemV = max_abs_scaler.fit_transform(problemV)
    print("Finished Fit Transform")

    X = np.concatenate((stu,stepName,problemName,kc,problemStep,studentProblem,studentUnit,studentKC,problemV,opps),axis=1)
    print(X.shape)
    print("Finished concatenate")

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

    stu,stepName,problemName,kc,problemStep,studentProblem,studentUnit,studentKC,problemV,opps,y1 = read_master_file()
    print("Finished processing")

    max_abs_scaler = preprocessing.MaxAbsScaler()
    opps = max_abs_scaler.fit_transform(opps)
    
    max_abs_scaler = preprocessing.MaxAbsScaler()
    problemV = max_abs_scaler.fit_transform(problemV)
    print("Finished Fit Transform")

    X1 = np.concatenate((stu,stepName,problemName,kc,problemStep,studentProblem,studentUnit,studentKC,problemV,opps),axis=1)
    print(X1.shape)
    print("Finished concatenate")

    y1 = np.array(y1)

    test_pred = lr.predict(X1)
    print("Test Correct :%2.2f"%(np.mean(test_pred == y1) * 100))
    errorVal = np.sqrt(mean_squared_error(y1, test_pred))
    print("Test RMSE :%0.4f" % (errorVal))