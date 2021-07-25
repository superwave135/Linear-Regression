import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt 
import csv
import pprint as pp
from time import perf_counter

name_cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
raw_df = pd.read_csv("housing.data", names = name_cols, header=None, delim_whitespace = True)
raw_df

# check for null values. found none
row_num = raw_df.shape[0]
col_num = raw_df.shape[1]
print(f'{raw_df.isnull().sum()}, ROWS: {row_num}, COLS:{col_num}') # check for null values

# data preparation for latter use 
df = pd.DataFrame(raw_df).to_numpy() # convert from dataframe format to numpy format
# print(f'last row last column: {df[505][-1]}, type: {type(df[1][-1])}')
print(df.shape)
x_features_only = df[:, :-1]  # all features
y_target = df[:, -1]  # only y label

y_response = y_target.reshape((506,1)) # gotta reshape for concatenate purpose
print('y_response shape:', y_response.shape)

x_features_ones_ylabel = np.concatenate([x_features_only, np.ones([np.shape(x_features_only)[0], 1]), y_response], axis=1)
# pp.pprint(x_features_ones_ylabel)  # x_features are NOT normalised yet.
print('df_shape of features + ones + label:', x_features_ones_ylabel.shape)

def dataNorm(X):
    '''
    normalize the feature data for latter use 
    input: X is a matrix of x-features and y-label
    output: x-features only all in proper columns
    '''
    xMerged = np.ones([np.shape(X)[0]])  # create a temp row of all zeros for vstack purpose only
    f_transpose = X.T #feature cols. switch the columns to rows for iteration later
    for i in f_transpose:  
        arr_transpose = (i - np.min(i)) / np.ptp(i)
        xMerged = np.vstack((xMerged, arr_transpose)) # merging output and features row-wise

    final_merged = xMerged[1:] # remove temp row of all zeros
    return final_merged.T # transpose to make features col-wise again

# Part 5
def gradient_func(weights, X, y_target):  # Vectorized gradient function
    '''
    Given `weights` - a current "Guess" of what our weights should be
          `X` - matrix of shape (N,D) of input features
          `y_target` - target y values
    Return gradient of each weight evaluated at the current value
    '''
    N, D = np.shape(X)
    y_pred = np.dot(X, weights)  # alternative, use np.matmul()
    error = np.subtract(y_pred, y_target)
    return y_pred, error  # return the gradient of the cost function

def predict(x_test, y_test, w):
    '''
    Compute y prediction, error and rmse 
    Given `X` - matrix of shape (N,D) of input features
          `y_target` - target y values
    Solves for rmse  y prediction and y_test.
    Return y prediction, y_test and rmse
    '''
    y_pred, error = gradient_func(w, x_test, y_test)   # call the gradient function. get y_pred, error output
#     print('y_pred shape:', y_pred.shape)   
    rmse = np.sqrt(np.square(np.subtract(y_test,y_pred)).mean()) 
    print('rmse:', rmse)
    
    return y_pred, y_test, rmse  # return the gradient of the cost function

# function to find the optimal learn rate
def optimal_learn_rate(X, y_target, alpha, print_every=1000, niter=50000):  # gotta varies the alpha to get the most accurate w
    '''
    Given `X` - matrix of shape (N,D) of input features
          `y_target` - target y values
    Solves for linear regression weights.
    Return weights after `niter` iterations.
    '''
    N, D = np.shape(X)                                  # feature matrix has N rows and D cols
    w = np.zeros([D]) 

    # initialize all the weights to zeros based on N cols of feature matrix
    for k in range(niter):   # loop over niter counts
       
        y_pred, error = gradient_func(w, X, y_target)        # call the gradient function. get y_pred, error output
        dw = np.dot(np.transpose(X), error) / float(N)
        # -------------------------------------------------------------------------------
        prev = w                           # assign the previous weight to prev variable
        w = w - alpha * dw                 # update the weight with the learning rate and gradient change 
        new = w                            # update the new weight to new variable
        # ------------------------------------------------------------------------------   
        # when there is no improvement over the previous w, then get the latest optimal value
        if k % print_every == 0 and np.all(new-prev) == False:           # for every 5000 count
            print(f"Learning rate (alpha) is: {str(alpha)}")
            print(f'Weight after {k} iteration:\n {str(w)}')
            print()
            break                 

    return w
### ------ calling main to determine optimal learn rate -------
x_features = x_features_ones_ylabel[:,:-2]
x_features_normalized = dataNorm(x_features)
x_features_normalized_ones = np.concatenate([x_features_normalized, np.ones([np.shape(x_features_normalized)[0], 1])], axis=1)
y_entire = x_features_ones_ylabel[:,-1]

for i in np.arange(1.0, 0.1, -0.1):  # Part 5 main calling block
    print('Running in progress ...')
    weight = optimal_learn_rate(X = x_features_normalized_ones, y_target = y_entire, alpha = round(i,3))

print('Running completed.')
# print('The first learning rate that shows up is the optimal learning rate.\n')
print('Final optimal weights:\n', weight)
print('\nThe optimal learn rate is 0.5')
np.savetxt("optimal_weights.csv", weight, fmt="%10.8f", delimiter=",")

# Check confirm our weights telly with the np.linalg.lstsq weights 
np.linalg.lstsq(x_features_normalized_ones, y_entire, rcond=None)[0] # the last in the output is the y-intercept

# Part 5
def gradient_descent(X, y_target, alpha, print_every=5000, niter=100000):  # gotta varies the alpha to get the most accurate w
    '''
    Given `X` - matrix of shape (N,D) of input features
          `t` - target y values
    Solves for linear regression weights.
    Return weights after `niter` iterations.
    '''
    N, D = np.shape(X)                                  # feature matrix has N rows and D cols
    w = np.zeros([D])                                   # initialize all the weights to zeros based on N cols of feature matrix
    for k in range(niter):   # loop over niter counts
       
        y_pred, error = gradient_func(w, X, y_target)        # call the gradient function. get y_pred, error output
        dw = np.dot(np.transpose(X), error) / float(N)
        # -------------------------------------------------------------------------------
        prev = w                           # assign the previous weight to prev variable
        w = w - alpha * dw                 # update the weight with the learning rate and gradient change 
        new = w                            # update the new weight to new variable
        # ------------------------------------------------------------------------------        
    return w

# Part 6
def splitCV(X_norm, K): # Split a dataset into k folds
    '''
    inputs - X_norm
    K = num of splits
    '''
    dataset_split = []
    np.random.shuffle(X_norm) # shuffles the rows in the X_norm matrix
    fold_size = int(len(X_norm) / K) # compute the num of rows per fold
    row_num = X_norm.shape[0]

    for i in range(K):
        if i == K-1:
            fold = np.array(X_norm)
            dataset_split.append(X_norm)
        else:
            dataset_split.append(X_norm[:fold_size])
            X_norm = X_norm[fold_size:]       
    return dataset_split

#Part 6
def CV_Main(x_features_ones_ylabel, cv_num): # k = number of neighbors
    '''
    cv split of the dataset into test and train
    '''
    cv_list = []
    X_cv = splitCV(x_features_ones_ylabel, cv_num) # split the data set into K folds = number of parts. X_cv is a list of folds
    print('\nCV_computation ongoing ... ')
    for idx, list_array in enumerate(X_cv): # looping the dataset for cross validation 
        duplicate = X_cv.copy()
        test = list_array
        del duplicate[idx]  # delete the test element from duplicate set, remaining become train elements
        train = duplicate   # remaining elements in duplicate become train set
        train = np.vstack((train)) # convert train stack up vertically
        cv_list.append(np.array([test, train])) #append test and train into a list before return
    return cv_list  # cv_list is a list type containing 2 elements - test and train

## PART 6 and 7    
# MAIN CALL BLOCK for CROSS VALIDATION over 5, 10, 15
cv5_ypred = []   # stores 5 elements of y_pred.
cv10_ypred = []  # stores 10 elements of y_pred.
cv15_ypred = []  # stores 15 elements of y_pred.
cv5_yactual = []   # stores 5 elements of y_actual.
cv10_yactual = []  # stores 10 elements of y_actual.
cv15_yactual = []  # stores 15 elements of y_actual.
cv5_rmse = []    # stores 5 rmse values
cv10_rmse = []   # stores 10 rmse values
cv15_rmse = []   # stores 15 rmse values

for cv in [5, 10, 15]:  # Looping over the cv numbers

    t1_start = perf_counter() # Start the stopwatch / counter 
    cv_list = CV_Main(x_features_ones_ylabel, cv)    
    print(f"-------- CV {cv} --------")
    
    for num in cv_list:  # for each fold in a list of k folds

        test = num[0]            # grab the test set from the fold
        x_test_features = test[:, :-2]    # grab the features from the test set
        test_ones = test[:, -2]
        x_test_ones = test_ones.reshape((test_ones.shape[0], 1))
        x_test_features_norm = dataNorm(x_test_features)     
        y_test = test[:, -1]     # grab the label from the test set
        # after test features are normalized, add the col of ones to become x_test
        x_test = np.concatenate((x_test_features_norm, x_test_ones), axis=1)

        train = num[1]           # grab the train set from the fold
        x_train_features = train[:, :-2]    # grab the features from the test set
        train_ones = train[:, -2]
        x_train_ones = train_ones.reshape((train_ones.shape[0], 1))
        x_train_features_norm = dataNorm(x_train_features)       
        y_train = train[:, -1]   # grab the label from the train set
        # after train features are normalized, add the col of ones to become x_train
        x_train = np.concatenate((x_train_features_norm, x_train_ones), axis=1)        

        w = gradient_descent(x_train, y_train, alpha=0.5)  # get the fitted weights from x, y train sets 
        y_pred, y_actual, rmse = predict(x_train, y_train, w)  # apply the w onto the x, y test sets to yield y_pred 
        print()
        if cv == 5:
            cv5_ypred.append(y_pred)
            cv5_yactual.append(y_actual)
            cv5_rmse.append(rmse)
            cv5_train = train
            cv5_test = test
            
        elif cv == 10:
            cv10_ypred.append(y_pred)
            cv10_yactual.append(y_actual)
            cv10_rmse.append(rmse)
            cv10_train = train
            cv10_test = test

        elif cv == 15:
            cv15_ypred.append(y_pred)
            cv15_yactual.append(y_actual)
            cv15_rmse.append(rmse)
            cv15_train = train
            cv15_test = test

    t1_stop = perf_counter() # Stop the stopwatch / counter 
    print(f'\nElapsed time {t1_stop-t1_start} secs\n') 
print()
print('---- Run completed ----')    
# -------------------------------------------------
with open('cv5_ypred.csv', 'w') as f: 
    write = csv.writer(f) 
    write.writerows(val for val in cv5_ypred) 
    
with open('cv5_yactual.csv', 'w') as f: 
    write = csv.writer(f)       
    write.writerows(val for val in cv5_yactual) 
    
with open('cv5_rmse.csv', 'w', newline='') as f: 
    write = csv.writer(f) 
    write.writerow(val for val in cv5_rmse)
    
with open('cv5_train.csv', 'w') as f: 
    write = csv.writer(f) 
    write.writerows(list(val) for val in cv5_train)
    
with open('cv5_test.csv', 'w') as f: 
    write = csv.writer(f) 
    write.writerows(list(val) for val in cv5_test) 
#------------------------------------------------    
with open('cv10_ypred.csv', 'w') as f: 
    write = csv.writer(f) 
    write.writerows(val for val in cv10_ypred) 
    
with open('cv10_yactual.csv', 'w') as f: 
    write = csv.writer(f)       
    write.writerows(val for val in cv10_yactual) 
    
with open('cv10_rmse.csv', 'w', newline='') as f: 
    write = csv.writer(f) 
    write.writerow(val for val in cv10_rmse)
    
with open('cv10_train.csv', 'w') as f: 
    write = csv.writer(f) 
    write.writerows(list(val) for val in cv10_train)
    
with open('cv10_test.csv', 'w') as f: 
    write = csv.writer(f) 
    write.writerows(list(val) for val in cv10_test)
#------------------------------------------------
with open('cv15_ypred.csv', 'w') as f: 
    write = csv.writer(f) 
    write.writerows(val for val in cv15_ypred) 
    
with open('cv15_yactual.csv', 'w') as f: 
    write = csv.writer(f)       
    write.writerows(val for val in cv15_yactual) 
    
with open('cv15_rmse.csv', 'w', newline='') as f: 
    write = csv.writer(f) 
    write.writerow(val for val in cv15_rmse)
    
with open('cv15_train.csv', 'w') as f: 
    write = csv.writer(f) 
    write.writerows(list(val) for val in cv15_train)
    
with open('cv15_test.csv', 'w') as f: 
    write = csv.writer(f) 
    write.writerows(list(val) for val in cv15_test) 

# generate the RMSE tables for CV 5, 10 and 15
i, j = 0, 0
for rmse in [cv5_rmse, cv10_rmse, cv15_rmse]: 
    df = pd.DataFrame(rmse, columns=[f'---CV-{j+5}--RMSE---'])
    print(df)
    print(f'Average RMSE for CV-{j+5} is {np.mean(rmse)}\n')
    i+=1
    j+=5
