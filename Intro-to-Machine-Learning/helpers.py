#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
helpers.py

Contains functions to be used in in poi_id.py and for creating the jupyter
notebook that becomes report.html
"""

#import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, \
f1_score
from sklearn.feature_selection import SelectKBest
#from sklearn.pipeline import Pipeline




def construct_dataframe(data_dict, unwanted = {'poi', 'email_address'}):
    """
    Make a dataframe of the data dictionary
    """
    df = pd.DataFrame.from_dict(data_dict, orient = 'index')
    df = df.drop('email_address', axis = 1)
    df = df.replace('NaN', np.nan)
    features_list = df.columns
    features_list = [e for e in features_list if e not in unwanted]
    features_list.insert(0, 'poi')
    df = df[features_list]
    df['poi'] = df['poi'].astype(int)
    return df

def create_new_features(df):
    """
    New features reflect portion of emails that are related to poi
    """
    df['from_this_person_to_poi_ratio'] = (df['from_this_person_to_poi']
                                           /df['from_messages'])
    df['from_poi_to_this_person_ratio'] = (df['from_poi_to_this_person']
                                           /df['to_messages'])
    df['shared_receipt_with_poi_ratio'] = (df['shared_receipt_with_poi']
                                           /df['to_messages'])
    return df

def dump_preperation(df, features_list):
    """
    Creates a data dictionary from the features created in the tuning section
    Creates features_list as a list of strings, each of which is a feature name.
    The first feature must be "poi"
    """
    features, labels = prep_df(df, desired_features = features_list)
    features.insert(0, 'poi', labels)
    features_list.insert(0, 'poi')
    my_dataset = features.to_dict(orient = 'index')
    return my_dataset, features_list

def good_corr_features(features, labels, clf):
    """
    Ranks feature importance by corellation to labels
    Returns a dataframe containing only those features
    """
    results = []
    top_corr = list(abs(features.corrwith(labels)).sort_values(
            ascending = False).index)
    cols = list(features.columns)
    for f in range(1,len(cols)):
        features_test = features[top_corr[0:f]]
        scores = cross_val_score(clf, features_test, labels, scoring = "f1", 
                                 cv =3)
        results.append([scores.mean()])
    
    return features[top_corr[0:np.argmax(results)+1]]

def good_corr_features_scores(features, labels, clf, cv = 3):
    """
    Ranks feature importance by corellation to labels
    Returns a dataframe containing results of cv
    """
    results = []
    top_corr = list(abs(features.corrwith(labels)).sort_values(
            ascending = False).index)
    cols = list(features.columns)
    for f in range(1,len(cols)):
        features_test = features[top_corr[0:f]]
        scores = cross_val_score(clf, features_test, labels, scoring = "f1", 
                                 cv = cv)
        results.append([scores.mean()])
    results_df = (pd.DataFrame(results, columns = ['Mean F1 Score'], index = range(1,len(results)+1)))
    return results_df

def k_best_features(pipe, features):
    """
    Return the list of features and scores found from SelectKBest
    """
    step_number = 0
    for step, method in pipe.steps:
        if type(method) == SelectKBest:
            break
        step_number +=1
    if (step_number > 1) and (step_number == len(pipe.steps)):
        print "Could not Find SelectKBest in Pipeline"
        return
    else:
        k = pipe.steps[step_number][1].get_params()['k']
        features_list = sorted(zip(pipe.steps[step_number][1].scores_, 
                                   features.columns), reverse = True)[0:k]
        return features_list

def load_dataset():
    """
    Load the dictionary containing the dataset
    """
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
    return data_dict

def outlier_table(df):
    """
    Returns a table with samples containing features that are 3 standard 
    deviations from the mean
    Table contains row index, feature name and value
    """
    features = ['Outlier Feature', 'Value', 'poi']
    outlier_table = pd.DataFrame(columns = features)
    for col in df.columns:
        ol = pd.DataFrame(columns = features)
        outliers = df[(np.abs(df[col]-df[col].mean())>(3*df[col].std()))]
        ol['Value'] = outliers[col]
        ol['Outlier Feature'] = col
        ol['poi'] = outliers['poi']
        outlier_table = outlier_table.append(ol)
    return outlier_table

def prep_df(df, desired_features = [], target = 'poi'):
    """
    Replicates all the preperation performed during EDA
    """
    df_prep = df.copy()
    df_prep = remove_outliers(df_prep)
    df_prep = create_new_features(df_prep)
    df_prep = process_nan(df_prep)
    df_prep = remove_all_zeros(df_prep)
    labels = df_prep[target]
    features = df_prep.drop(target, 1)
    if len(desired_features) > 0:
        features = features[desired_features]
    return features, labels

def print_cv_scores(clf, features, labels, cv = 3, print_out = True):
    """
    Calculates and displays the mean of cross validation scores
    Returns a dataFrame row
    """
    scoring = ["accuracy", "precision", "recall", "f1"]
    scores = []
    if print_out:
        print "Mean Cross Validation Scores"
    for s in scoring:
        cv_score = cross_val_score(clf, features, labels, scoring = s, 
                                   cv = cv).mean()
        scores.append(cv_score)
        if print_out:
            print s, cv_score
    scores = pd.DataFrame(scores).T
    scores.columns = scoring    
    return scores

def print_val_scores(clf, features_train, features_test, labels_train, 
                        labels_test, print_out = True):
    """
    Prints out the performance of clf on the test data
    Also returns a dataframe of these scores
    """
    scoring = ["accuracy", "precision", "recall", "f1"]
    scores = []
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    scores = pd.DataFrame([[accuracy_score(labels_test, pred), 
                            precision_score(labels_test, pred),
                            recall_score(labels_test, pred),
                            f1_score(labels_test, pred)]],
             columns = scoring)
    if print_out:
        print "Validation Scores"
        for s in range(len(scoring)):
            print scoring[s], scores.values[0][s]
    return scores

def process_nan(df):
    '''
    Changes NaN to zero
    It was a judgement call to fill NaN with zeros
    Use imputer if other values are desired
    '''
    df = df.fillna(0)
    return df

def remove_all_zeros(df):
    """
    Remove rows where all values are equal zero
    """
    df = df.loc[(df != 0).any(1)]
    return df

def remove_outliers(df):
    """
    For now, the only outlier is 'TOTAL'
    """
    if 'TOTAL' in df.index:
        df = df.drop(['TOTAL'])
    return df

def train_test_split(df, test_size = 0.3, target = "poi", random_state = 42):
    """
    Split the data using Stratified Shufflesplit
    Create Training and Test Sets
    Carry on exploration with Training Set
    """ 
    ss_split = StratifiedShuffleSplit(n_splits = 2, test_size = test_size, 
                                      random_state = random_state)
    for train_index, test_index in ss_split.split(df, df[target]):
        strat_train_set = df.iloc[train_index]
        strat_test_set = df.iloc[test_index]
    return strat_train_set, strat_test_set

def validate_classifier(clf, features_train, features_test, labels_train, 
                        labels_test, index):
    """
    Prints out the performance of clf on the test data
    Also returns a dataframe of these scores
    Index is string for the dataframe row
    """
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    scores = pd.DataFrame([[accuracy_score(labels_test, pred), 
                            precision_score(labels_test, pred),
                            recall_score(labels_test, pred),
                            f1_score(labels_test, pred),
                            roc_auc_score(labels_test, pred)]],
             columns = ["accuracy", "precision", "recall", "f1", "roc_auc"], 
                         index = [index])
    return scores
