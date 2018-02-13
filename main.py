#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
#from features_extraction.dataset import Dataset

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print('Usage: python3.5 main.py extracted_features.csv')
        exit(1)

    # create dataset frame
    #df = Dataset(filename=sys.argv[1], verbose=True).data
    #df = shuffle(df)

    # eliminating non numeric columns
    #df = df._get_numeric_data()
    #df.to_csv('extracted_features.csv', encoding='utf-8', index=False)
    
    # import data
    df = pd.read_csv(sys.argv[1])
    
    # training/test splitting
    y = np.array(df['class'])
    X = np.array(df.ix[:, df.columns != 'class'])
    predictors = df.ix[:, df.columns != 'class'].columns.values
    
    print(np.shape(X), np.shape(y))
    
    clf = GradientBoostingClassifier(warm_start=True, learning_rate=0.1, n_estimators=70, min_samples_split=200, min_samples_leaf=60, max_depth=12, subsample=0.8, random_state=10)
    clf.fit(X, y)
    
    # feature importances
    feat_imp = pd.Series(clf.feature_importances_, predictors).sort_values(ascending=False)[:10]
    print(feat_imp)
    feat_imp.plot(kind='barh', title='', figsize=(4,5), width=0.8, grid=True)
    plt.tight_layout()
    plt.ylabel('Feature Importance Score')
    plt.savefig('feature_importances.png')

    # 0 is the negative (minor) class
    accuracy = []
    TP, TN, FP, FN = [], [], [], []
    f_score = []
    recall = []
    precision = []
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        
        y_pred = clf.fit(X[train], y[train]).predict(X[test])
        
        accuracy.append(accuracy_score(y[test], y_pred))
        recall.append(recall_score(y[test], y_pred))
        precision.append(precision_score(y[test], y_pred))
        f_score.append(f1_score(y[test], y_pred))
        
        tn, fp, fn, tp = confusion_matrix(y[test], y_pred).ravel()
        TP.append(tp)
        FP.append(fp)
        TN.append(tn)
        FN.append(fn)
        
    print('k&1&2&3&4&5&\\textbf{Avg}\\\\\\hline\\hline')
    print('\\textbf{TP}&' + '&'.join([str(t) for t in TP]) + '&%.3f' % np.mean(TP) + '\\\\\\hline')
    print('\\textbf{TN}&' + '&'.join([str(t) for t in TN]) + '&%.3f' % np.mean(TN) + '\\\\\\hline')
    print('\\textbf{FP}&' + '&'.join([str(t) for t in FP]) + '&%.3f' % np.mean(FP) + '\\\\\\hline')
    print('\\textbf{FN}&' + '&'.join([str(t) for t in FN]) + '&%.3f' % np.mean(FN) + '\\\\\\hline')
    print('\\textbf{Acc}&' + '&'.join(['%.3f' % t for t in accuracy]) + '&%.3f' % np.mean(accuracy) + '\\\\\\hline')
    print('\\textbf{R}&' + '&'.join(['%.3f' % t for t in recall]) + '&%.3f' % np.mean(recall) + '\\\\\\hline')
    print('\\textbf{P}&' + '&'.join(['%.3f' % t for t in precision]) + '&%.3f' % np.mean(precision) + '\\\\\\hline')
    print('$\\mathbf{F_1}$&' + '&'.join(['%.3f' % t for t in f_score]) + '&%.3f' % np.mean(f_score) + '\\\\\\hline')
    
    '''
    # control:
    prec, rec, acc = [], [], []
    for i in range(5):
        acc.append((TP[i]+TN[i])/(TP[i]+TN[i]+FP[i]+FN[i]))
        prec.append((TP[i])/(TP[i]+FP[i]))
        rec.append((TP[i])/(TP[i]+FN[i]))
        
    print('Accuracy ', acc, np.mean(acc))
    print('Recall ', rec, np.mean(rec))
    print('Precision ', prec, np.mean(prec))
    '''
