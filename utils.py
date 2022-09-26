import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

def load_data_split(dat,split_type, seed):
    n_fold = 5
    idx_test_fold = 0
    idx_val_fold = -1
    idx_test = None
    idx_train = None
    x_pep = dat.epi
    x_tcr = dat.tcr
    
    if split_type == 'random':
        n_total = len(x_pep)
    elif split_type == 'epi':
        unique_peptides = np.unique(x_pep)
        n_total = len(unique_peptides)
    elif split_type == 'tcr':
        unique_tcrs = np.unique(x_tcr)
        n_total = len(unique_tcrs)
        
    np.random.seed(seed)    
    idx_shuffled = np.arange(n_total)
    np.random.shuffle(idx_shuffled)
    
    # Determine data split from folds
    n_test = int(round(n_total / n_fold))
    n_train = n_total - n_test

    # Determine position of current test fold
    test_fold_start_index = idx_test_fold * n_test
    test_fold_end_index = (idx_test_fold + 1) * n_test

    if split_type == 'random':
        # Split data evenly among evenly spaced folds
        # Determine if there is an outer testing fold
        if idx_val_fold < 0:
            idx_test = idx_shuffled[test_fold_start_index:test_fold_end_index]
            idx_train = list(set(idx_shuffled).difference(set(idx_test)))
        else:
            validation_fold_start_index = args.idx_val_fold * n_test
            validation_fold_end_index = (args.idx_val_fold + 1) * n_test
            idx_test_remove = idx_shuffled[test_fold_start_index:test_fold_end_index]
            idx_test = idx_shuffled[validation_fold_start_index:validation_fold_end_index]
            idx_train = list(set(idx_shuffled).difference(set(idx_test)).difference(set(idx_test_remove)))
    elif split_type == 'epi':
        if idx_val_fold < 0:
            idx_test_pep = idx_shuffled[test_fold_start_index:test_fold_end_index]
            test_peptides = unique_peptides[idx_test_pep]
            idx_test = [index for index, pep in enumerate(x_pep) if pep in test_peptides]
            idx_train = list(set(range(len(x_pep))).difference(set(idx_test)))
        else:
            validation_fold_start_index = args.idx_val_fold * n_test
            validation_fold_end_index = (args.idx_val_fold + 1) * n_test
            idx_test_remove_pep = idx_shuffled[test_fold_start_index:test_fold_end_index]
            test_remove_peptides = unique_peptides[idx_test_remove_pep]
            idx_test_pep = idx_shuffled[validation_fold_start_index:validation_fold_end_index]
            test_peptides = unique_peptides[idx_test_pep]
            idx_test = [index for index, pep in enumerate(x_pep) if pep in test_peptides]
            idx_test_remove = [index for index, pep in enumerate(x_pep) if pep in test_remove_peptides]
            idx_train = list(set(range(len(x_pep))).difference(set(idx_test)).difference(set(idx_test_remove)))
    elif split_type == 'tcr':
        if idx_val_fold < 0:
            idx_test_tcr = idx_shuffled[test_fold_start_index:test_fold_end_index]
            test_tcrs = unique_tcrs[idx_test_tcr]
            idx_test = [index for index, tcr in enumerate(x_tcr) if tcr in test_tcrs]
            idx_train = list(set(range(len(x_tcr))).difference(set(idx_test)))
        else:
            validation_fold_start_index = args.idx_val_fold * n_test
            validation_fold_end_index = (args.idx_val_fold + 1) * n_test
            idx_test_remove_tcr = idx_shuffled[test_fold_start_index:test_fold_end_index]
            test_remove_tcrs = unique_tcrs[idx_test_remove_tcr]
            idx_test_tcr = idx_shuffled[validation_fold_start_index:validation_fold_end_index]
            test_tcrs = unique_tcrs[idx_test_tcr]
            idx_test = [index for index, tcr in enumerate(x_tcr) if tcr in test_tcrs]
            idx_test_remove = [index for index, tcr in enumerate(x_tcr) if tcr in test_remove_tcrs]
            idx_train = list(set(range(len(x_tcr))).difference(set(idx_test)).difference(set(idx_test_remove)))

    testData = dat.iloc[idx_test, :].sample(frac=1).reset_index(drop=True)
    trainData = dat.iloc[idx_train, :].sample(frac=1).reset_index(drop=True)
    

    print('================check Overlapping========================')
    print('number of overlapping tcrs: ', str(len(set(trainData.tcr).intersection(set(testData.tcr)))))
    print('number of overlapping epitopes: ', str(len(set(trainData.epi).intersection(set(testData.epi)))))
    
    # tcr_split testing read 
    X1_test_list, X2_test_list, y_test_list = testData.tcr_embeds.to_list(), testData.epi_embeds.to_list(),testData.binding.to_list()
    X1_test, X2_test, y_test = np.array(X1_test_list), np.array(X2_test_list), np.array(y_test_list)
    # tcr_split training read 
    X1_train_list, X2_train_list, y_train_list = trainData.tcr_embeds.to_list(), trainData.epi_embeds.to_list(),trainData.binding.to_list()
    X1_train, X2_train, y_train = np.array(X1_train_list), np.array(X2_train_list), np.array(y_train_list)
    return  X1_train, X2_train, y_train, X1_test, X2_test, y_test, testData, trainData



def print_performance(y, yhat):
    print('================Performance========================')
    print('AUC: ' + str(roc_auc_score(y, yhat)))
    yhat[yhat>=0.5] = 1
    yhat[yhat<0.5] = 0

    accuracy = accuracy_score(y, yhat)
    precision1 = precision_score(
        y, yhat, pos_label=1, zero_division=0)
    precision0 = precision_score(
        y, yhat, pos_label=0, zero_division=0)
    recall1 = recall_score(y, yhat, pos_label=1, zero_division=0)
    recall0 = recall_score(y, yhat, pos_label=0, zero_division=0)
    f1macro = f1_score(y, yhat, average='macro')
    f1micro = f1_score(y, yhat, average='micro')

    print('precision_recall_fscore_macro ' + str(precision_recall_fscore_support(y,yhat, average='macro')))
    print('acc is '  + str(accuracy))
    print('precision1 is '  + str(precision1))
    print('precision0 is '  + str(precision0))
    print('recall1 is '  + str(recall1))
    print('recall0 is '  + str(recall0))
    print('f1macro is '  + str(f1macro))
    print('f1micro is '  + str(f1micro))
    return