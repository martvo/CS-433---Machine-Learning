
# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from data_cleaning import *
import implementations as imp
import matplotlib.pyplot as plt
import seaborn as sns



# # Import data

y_train, x_train, ids_train = load_csv_data("../data/train.csv")
y_test, x_test, ids_test = load_csv_data("../data/test.csv")


# # Clean test data and add features

NUM_JETS = 4

PRI_jet_num_train = np.array([x_train[:, COLUMN_TO_DROP]]).astype(int)
del_x_train = np.delete(x_train, COLUMN_TO_DROP, axis=1)

replaced_x_train = replace_undefined_with_mean(del_x_train, UNDEFINED_VALUE)

norm_x_train, train_data_mean, train_data_std = mean_std_normalization(replaced_x_train)


# # Do the same for the test data

PRI_jet_num_test = np.array([x_test[:, COLUMN_TO_DROP]]).astype(int)
del_x_test = np.delete(x_test, COLUMN_TO_DROP, axis=1)

replaced_x_test = replace_undefined_with_mean(del_x_test, UNDEFINED_VALUE)

norm_x_test, test_data_mean, test_data_std = mean_std_normalization(replaced_x_test, train_data_mean, train_data_std)


# # Train model

#hyperparameters
init_w = np.ones(30)
max_iters = 100
lambda_list = np.logspace(-9,-1,50)
degree_list = range(8,15)
gamma_list = np.linspace(0.05,10)
b_size = 1
ratio = 0.8
seed = 7

#scoring placeholders
best_lambda = np.zeros(4)
best_degree = np.zeros(4,int)
best_score = np.zeros(4)
all_scores = np.zeros([len(lambda_list), len(degree_list)])

#indexing variables
d = 0
g = 0

#grid search loop
for degree in degree_list:
    print("running with degree = {}".format(degree)) #progress visualization
    for i in range(NUM_JETS):
        #selecting current part of the dataset
        curr_x = norm_x_train[PRI_jet_num_train[0,:]==i]
        curr_y = y_train[PRI_jet_num_train[0,:]==i]
        
        #splitting training data into a training and validataion part
        (t1_x, t1_y, t2_x, t2_y) = split_data(curr_x, curr_y, ratio, seed)
        
        px_t1 = build_poly(t1_x,degree)
        px_t2 = build_poly(t2_x,degree)
        g = 0
        for lambda_ in lambda_list:

            _, w1 = imp.ridge_regression(t1_y,px_t1, lambda_)
            _, w2 = imp.ridge_regression(t2_y,px_t2, lambda_)

            #check how well the hyperparameters did using 2-fold cross validation
            y_validation_1 = predict_labels(w2, px_t1)
            y_validation_2 = predict_labels(w1, px_t2)

            score = (sum(y_validation_2 == t2_y)/len(t2_y) + sum(y_validation_1 == t1_y)/len(t1_y))/2
            
            #save best parameters
            if score > best_score[i]:
                best_lambda[i] = lambda_
                best_degree[i] = degree
                best_score[i] = score
                
            #save all scores for plotting later
            all_scores[g,d] = all_scores[g,d] + score*sum(PRI_jet_num_train[0,:]==i)/len(norm_x_train)
            g = g+1
    d = d+1


# # Test accuracy

actual_score = 0
for i in range(NUM_JETS):
    actual_score = actual_score + best_score[i]*sum(PRI_jet_num_train[0,:]==i)/len(norm_x_train)

print(actual_score)


# # Create submission

for i in range(NUM_JETS):
    
    #selecting current part of the dataset based on PRI_jet_num
    curr_x = norm_x_train[PRI_jet_num_train[0,:]==i]
    curr_y = y_train[PRI_jet_num_train[0,:]==i]
    
    #generate weights from optimal hyperparameter values
    px_tr = build_poly(curr_x, some_degree[i])
    _, w = imp.ridge_regression(curr_y, px_tr, best_lambda[i])
    
    #generate output array
    curr_x_test = norm_x_test[PRI_jet_num_test[0,:]==i]
    px_test = build_poly(curr_x_test,some_degree[i])
    y_test[PRI_jet_num_test[0,:]==i] = predict_labels(w, px_test)

create_csv_submission(ids_test, y_test, "y_pred.csv")


# # Visualize data


heatmap_fig = sns.heatmap(all_scores[:,:], xticklabels=degree_list, yticklabels=lambda_list[:])
heatmap_fig.get_figure().savefig('ridge_regression.png', bbox_inches='tight')


# In[49]:

print(best_lambda)
print(best_degree)
print(best_score)

