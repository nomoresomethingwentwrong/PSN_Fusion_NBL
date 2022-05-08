from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
from scipy import inner, outer
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from sklearn.utils import shuffle
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from random import seed
import time
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#session = tf.Session(config=config)
#K.set_session(session)

print(tf.__version__)

parser = argparse.ArgumentParser(description='cv classification model')
parser.add_argument('-i', '--inpath', type=str, default='../TARGET_Fusion/', help='input file')
parser.add_argument('-o', '--outpath', type=str, default='../final_result/TARGET/methy+counts/', help='output file')
parser.add_argument('-d', '--dataset', type=str, default='TARGET', help='SEQC or TARGET')
parser.add_argument('-fus', '--fusion', type=str, default='net', help='fusion technique: none, net or feature')
# parser.add_argument('--rfe', type=str, default='none', help='RFE choice: none, svm, dt, rf, lr')
parser.add_argument('-fea', '--feature_type', type = str, default='both', help='both, cen or mod')
parser.add_argument('-m', '--mycn', action='store_true', default=False, help='add this tag to include mych feature')

args = parser.parse_args()
inpath = args.inpath
outpath = args.outpath
dataset = args.dataset
fusion = args.fusion
feature_type = args.feature_type
mycn = args.mycn

os.makedirs(outpath, exist_ok=True)

if mycn:
    preflix = 'mycn_'
else:
    preflix = ''

if fusion == 'net' or fusion == 'none':
    filename = 'new_scaled_result.csv'
elif fusion == 'feature':
    filename = 'new_scaled_result_FeaFusion.csv'
else:
    print('fusion tech error')

file1 = inpath + preflix + filename

expression = np.loadtxt(file1, dtype=float, delimiter = ",")
label_vec = np.array(expression[:,-1],dtype=int)

if feature_type == 'both':
    expression = np.array(expression[:, :-1])	# include both centrality and modularity features
elif feature_type == 'cen':
    if mycn: 
        expression = np.append(expression[:, :13], np.expand_dims(expression[:, -2], axis=1), axis=1)
    else:
        expression = np.array(expression[:, :13])	# include only centrality features from GSE49710/62564
elif feature_type == 'mod':
    expression = np.array(expression[:, 13:-1])	# include only modularity features from GSE49710/62564
else:
    print('feature type error')

labels = []
for l in label_vec:
    if l==1:
        labels.append([0,1])    # 105 samples
    else:
        labels.append([1,0])    # 393 samples
labels = np.array(labels, dtype=int)
# labels = np.array(label_vec, dtype=int)
print('input features shape:', expression.shape)
print('labels shape:', labels.shape)

outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=40)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=40)

L2 = True
max_pooling = False
droph1 = False
display_step = 1

def dfn(x, layers, weights, biases, keep_prob):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    dfn_layer = []
    dfn_layer.append(layer_1)
    if droph1:
        layer_1 = tf.nn.dropout(layer_1, keep_prob=keep_prob)
    if layers[0]:
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        layer_2 = tf.nn.dropout(layer_2, keep_prob = keep_prob)
        dfn_layer.append(layer_2)
    if layers[1]:
        layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        layer_3 = tf.nn.relu(layer_3)
        layer_3 = tf.nn.dropout(layer_3, keep_prob = keep_prob)
        dfn_layer.append(layer_3)
    if layers[2]:
        layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
        layer_4 = tf.nn.relu(layer_4)
        layer_4 = tf.nn.dropout(layer_4, keep_prob=keep_prob)
        dfn_layer.append(layer_4)
    if layers[3]:
        layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
        layer_5 = tf.nn.relu(layer_5)
        layer_5 = tf.nn.dropout(layer_5, keep_prob=keep_prob)
        dfn_layer.append(layer_5)

    out_layer = tf.matmul(dfn_layer[number_of_layers], weights['out']) + biases['out']

    return out_layer

# file = open("final_result/SEQC/Feature_level_fusion/feafusion_result_log", "w")

# all_layers = [[True, False, False, False],[True, False, False, False],
#               [True, True, False, False],[True, True, False, False],
#               [True, True, True, False],[True, True, True, False],[True, True, True, False],[True, True, True, False],[True, True, True, False],[True, True, True, False],
#               [True, True, True, True],[True, True, True, True],[True, True, True, True],[True, True, True, True],[True, True, True, True],[True, True, True, True],[True, True, True, True],[True, True, True, True]
# ]
# all_layers_size = [[4], [8],
#                    [8,8], [4,4],
#                    [8,64,16], [8,16,4], [8,4,2], [4,4,2], [4,2,2], [2,2,2],
#                    [8,64,16,8], [8,16,4,8], [8,8,4,4], [8,4,4,2], [8,2,2,2], [4,4,2,2], [4,2,2,2], [2,2,2,2]
#                    ]
all_layers = [[True, False, False, False],[True, False, False, False],
              [True, True, False, False],[True, True, False, False],
              [True, True, True, False],[True, True, True, False],[True, True, True, False],[True, True, True, False],
              [True, True, True, True],[True, True, True, True],[True, True, True, True],[True, True, True, True]
]
all_layers_size = [[4], [8],
                   [8,8], [4,4],
                   [8,16,4], [8,4,2], [4,4,2], [4,2,2],
                   [8,16,4,8], [8,8,4,4], [8,4,4,2], [4,4,2,2]
                   ]
all_learning_rates = [ 1e-2, 1e-3, 1e-4]
# all_learning_rates = [1e-3, 1e-4]
all_batch_sizes = [8,32]
# all_training_epochs = [100,1000]
# all_training_epochs = [1000]
training_epochs = 1000
count = 0
all_exp_count = (len(all_layers_size))*len(all_learning_rates)*len(all_batch_sizes)

log_outer=[]
k_outer = 0

# metrics for outer_cv
acc_outer, auc_outer, f1_outer = [], [], []
test_acc_outer, test_auc_outer, test_f1_outer = [], [], []

# outer_cv is for model selection (tune the parameters of the model)
for train_index, test_index in outer_cv.split(expression, label_vec):
    k_outer += 1
    # training set for outer_cv is the whole dataset for inner_cv
    inner_data = expression[train_index, :]
    inner_label = labels[train_index, :]
    
    # initialize the best hyperparameters
    log_inner=[]
    best_layer_size = all_layers_size[0]
    best_lr = all_learning_rates[0]
    best_bs = all_batch_sizes[0]
    
    for i in range(len(all_layers_size)):
        for learning_rate in all_learning_rates:
            # for training_epochs in all_training_epochs:
            for batch_size in all_batch_sizes:
                layers = all_layers[i]
                layers_size = all_layers_size[i]

                number_of_layers = 0
                for j in range(len(layers)):
                    if layers[j] is True:
                        number_of_layers = j+1

                n_hidden_1 = np.shape(expression)[1]
                n_classes = 2
                n_features = np.shape(expression)[1]

                

                x = tf.placeholder(tf.float32, [None, n_features])
                y = tf.placeholder(tf.float32, [None, n_classes])

                keep_prob = tf.placeholder(tf.float32)
                lr = tf.placeholder(tf.float32)

                weights = {
                    'h1': tf.Variable(tf.truncated_normal(shape=[n_features, n_hidden_1], stddev=1/np.sqrt(n_features)))
                }

                biases = {
                    'b1': tf.Variable(tf.zeros([n_hidden_1])),
                    'out': tf.Variable(tf.zeros([n_classes]))
                }
                if layers[0]:
                    weights['h2'] = tf.Variable(tf.truncated_normal(shape=[n_hidden_1, layers_size[0]], stddev=1/np.sqrt(n_hidden_1)))
                    biases['b2'] = tf.Variable(tf.zeros([layers_size[0]]))
                if layers[1]:
                    weights['h3'] = tf.Variable(tf.truncated_normal(shape=[layers_size[0], layers_size[1]], stddev=1/np.sqrt(layers_size[0])))
                    biases['b3'] = tf.Variable(tf.zeros([layers_size[1]]))
                if layers[2]:
                    weights['h4'] = tf.Variable(tf.truncated_normal(shape=[layers_size[1], layers_size[2]], stddev=1 / np.sqrt(layers_size[1])))
                    biases['b4'] = tf.Variable(tf.zeros([layers_size[2]]))
                if layers[3]:
                    weights['h5'] = tf.Variable(tf.truncated_normal(shape=[layers_size[2], layers_size[3]], stddev=1 / np.sqrt(layers_size[2])))
                    biases['b5'] = tf.Variable(tf.zeros([layers_size[3]]))


                weights['out'] = tf.Variable(tf.truncated_normal(shape=[layers_size[number_of_layers-1], n_classes], stddev=1/np.sqrt(layers_size[number_of_layers-1])))

                pred_inner = dfn(x, layers, weights, biases, keep_prob)

                # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
                if dataset == 'SEQC':
                    class_weights = tf.constant([1, 3.74])
                elif dataset == 'TARGET':
                    class_weights = tf.constant([1, 1.04])
                else:
                    print('wrong dataset type!')
                    exit()
                cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=pred_inner, targets=y, pos_weight=class_weights))
                if L2:
                    reg = tf.nn.l2_loss(weights['h1'])
                    if layers[0]:
                        reg+=tf.nn.l2_loss(weights['h2'])
                    if layers[1]:
                        reg+=tf.nn.l2_loss(weights['h3'])
                    if layers[2]:
                        reg+=tf.nn.l2_loss(weights['h4'])
                    if layers[3]:
                        reg+=tf.nn.l2_loss(weights['h5'])
                    reg+=tf.nn.l2_loss(weights['out'])
                    cost = tf.reduce_mean(cost+0.001*reg)
                optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(cost)

                correct_prediction = tf.equal(tf.argmax(pred_inner,1), tf.argmax(y,1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                y_score = tf.nn.softmax(logits=pred_inner)

                loss_rec = np.zeros([training_epochs, 1])
                training_eval = np.zeros([training_epochs, 2])


                # metrics for inner_cv
                acc_inner, auc_inner, f1_inner = [], [], []
                test_acc_inner, test_auc_inner, test_f1_inner = [], [], []
                
                # inner cv is to tune the hyperparameters
                for train_index_inner, test_index_inner in inner_cv.split(inner_data, label_vec[train_index]):
                    x_train_inner, x_test_inner = inner_data[train_index_inner, :], inner_data[test_index_inner, :]
                    y_train_inner, y_test_inner = inner_label[train_index_inner, :], inner_label[test_index_inner, :]
                    with tf.Session(config=config) as sess:
                        sess.run(tf.global_variables_initializer())
                        total_batch = int(np.shape(x_train_inner)[0] / batch_size)


                        #earlystopping
                        best_cost = 1e9
                        best_acc = 0
                        best_f1 = 0
                        stop=False
                        last_improvement=0
                        required_improvement=50
                        costs=[]

                        acc, auc, f1 = [], [], []
                        # validate_acc, validate_auc = [], []
                        test_acc, test_auc, test_f1 = [], [], []

                        for epoch in range(training_epochs):
                            avg_cost = 0
                            # x_tmp, y_tmp = shuffle(x_train, y_train)
                            for k in range(total_batch - 1):
                                batch_x, batch_y = x_train_inner[k*batch_size:k*batch_size+batch_size], y_train_inner[k*batch_size:k*batch_size+batch_size]
                                _, c = sess.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y, keep_prob:0.8, lr:learning_rate})
                                avg_cost += c/total_batch
                            # del x_tmp
                            # del y_tmp


                            #early stopping
                            # if avg_cost<best_cost:
                            # 	save_session = sess
                            # 	best_cost = avg_cost
                            # 	last_improvement=0
                            # else:
                            # 	last_improvement+=1
                            # if last_improvement>required_improvement:
                            # 	print("no improvement, early stopping applies")
                            # 	stop = True
                            # 	sess = save_session

                            acc_, y_s = sess.run([accuracy, y_score], feed_dict={x:x_train_inner, y:y_train_inner, keep_prob:1})
                            auc_ = metrics.roc_auc_score(y_train_inner, y_s)
                            y_train_label = np.argmax(y_train_inner, axis=1)
                            y_s_label = np.argmax(y_s, axis=1)
                            f1_ = metrics.f1_score(y_train_label, y_s_label)

                            acc.append(acc_)
                            auc.append(auc_)
                            f1.append(f1_)

                            # acc_inner.append(acc_)
                            # auc_inner.append(auc_)
                            # f1_inner.append(f1_)

                            # acc_, y_s = sess.run([accuracy, y_score], feed_dict={x: x_validate, y: y_validate, keep_prob: 1})
                            # auc_ = metrics.roc_auc_score(y_validate, y_s)


                            if acc_>best_acc:
                                save_session = sess
                                best_acc = acc_
                                last_improvement = 0
                            # if f1_ > best_f1:
                            #     save_session = sess
                            #     best_f1 = f1_
                            #     last_improvement = 0    
                            else:
                                last_improvement +=1
                            if last_improvement>required_improvement:
                                print("stop at epoch", epoch)
                                print('final training acc, auc, f1:', acc_, auc_, f1_)
                                print("==============early stopping===================")
                                stop = True
                                sess = save_session


                            # validate_acc.append(acc_)
                            # validate_auc.append(auc_)


                            acc_, y_s = sess.run([accuracy, y_score], feed_dict={x:x_test_inner, y:y_test_inner, keep_prob:1})
                            auc_ = metrics.roc_auc_score(y_test_inner, y_s)
                            y_test_label = np.argmax(y_test_inner, axis=1)
                            y_s_label = np.argmax(y_s, axis=1)
                            f1_ = metrics.f1_score(y_test_label, y_s_label)
                            
                            

                            
                            test_acc.append(acc_)
                            test_auc.append(auc_)
                            test_f1.append(f1_)

                            # test_acc_inner.append(acc_)
                            # test_auc_inner.append(auc_)
                            # test_f1_inner.append(f1_)

                            if stop is True:
                                break

                        print('final testing acc, auc, f1:', test_acc[-1], test_auc[-1], test_f1[-1])
                        print("==============next inner fold===================")
                        acc_inner.append(acc[-1])
                        auc_inner.append(auc[-1])
                        f1_inner.append(f1[-1])
                        test_acc_inner.append(test_acc[-1])
                        test_auc_inner.append(test_auc[-1])
                        test_f1_inner.append(test_f1[-1])
                
                avg_acc = np.array(acc_inner).mean()
                avg_auc = np.array(auc_inner).mean()
                avg_f1 = np.array(f1_inner).mean()
                avg_test_acc = np.array(test_acc_inner).mean()
                avg_test_auc = np.array(test_auc_inner).mean()
                avg_test_f1 = np.array(test_f1_inner).mean()

                std_acc = np.array(acc_inner).std()
                std_auc = np.array(auc_inner).std()
                std_f1 = np.array(f1_inner).std()
                std_test_acc = np.array(test_acc_inner).std()
                std_test_auc = np.array(test_auc_inner).std()
                std_test_f1 = np.array(test_f1_inner).std()
    
                arr = np.array([layers, layers_size, learning_rate, training_epochs, batch_size, avg_acc, std_acc, avg_test_acc, std_test_acc, avg_auc, std_auc, avg_test_auc, std_test_auc, avg_f1, std_f1, avg_test_f1, std_test_f1])
                log_inner.append(arr)
                
                print("<<<<<<<<<<<<<<<<<<<")
                print(str(layers_size))
                print("current inner cv process: "+str(count/all_exp_count))
                count+=1
                print("<<<<<<<<<<<<<<<<<<<")
    
    df_log_inner = pd.DataFrame(log_inner, columns = ['layers', 'layer_size', 'learning_rate', 'training_epochs', 'batch_size', 'train_acc', 'std', 'test_acc', 'std', 'train_auc', 'std', 'test_auc', 'std', 'train_f1', 'std', 'test_f1', 'std'])
    if fusion == 'none':
        filename2 = '_'
    elif fusion == 'net':
        filename2 = '_netfusion'
    elif fusion == 'feature':
        filename2 = '_feafusion'

    if feature_type == 'both':
        postfix = '_log.csv'
    elif feature_type == 'cen':
        postfix = '_cen_log.csv'
    elif feature_type == 'mod':
        postfix = '_mod_log.csv'
    else:
        print('output file naming error!')

    file2 = outpath + preflix + 'inner' + str(k_outer) + filename2 + postfix
    df_log_inner.to_csv(file2)
    
    best_idx = df_log_inner['test_f1'].idxmax()
    
    layers = df_log_inner['layers'].loc[best_idx]
    layers_size = df_log_inner['layer_size'].loc[best_idx]
    learning_rate = df_log_inner['learning_rate'].loc[best_idx]
    batch_size = df_log_inner['batch_size'].loc[best_idx]
    
    number_of_layers = 0
    for j in range(len(layers)):
        if layers[j] is True:
            number_of_layers = j+1

    n_hidden_1 = np.shape(expression)[1]
    n_classes = 2
    n_features = np.shape(expression)[1]

    

    x = tf.placeholder(tf.float32, [None, n_features])
    y = tf.placeholder(tf.float32, [None, n_classes])

    keep_prob = tf.placeholder(tf.float32)
    lr = tf.placeholder(tf.float32)

    weights = {
        'h1': tf.Variable(tf.truncated_normal(shape=[n_features, n_hidden_1], stddev=1/np.sqrt(n_features)))
    }

    biases = {
        'b1': tf.Variable(tf.zeros([n_hidden_1])),
        'out': tf.Variable(tf.zeros([n_classes]))
    }
    if layers[0]:
        weights['h2'] = tf.Variable(tf.truncated_normal(shape=[n_hidden_1, layers_size[0]], stddev=1/np.sqrt(n_hidden_1)))
        biases['b2'] = tf.Variable(tf.zeros([layers_size[0]]))
    if layers[1]:
        weights['h3'] = tf.Variable(tf.truncated_normal(shape=[layers_size[0], layers_size[1]], stddev=1/np.sqrt(layers_size[0])))
        biases['b3'] = tf.Variable(tf.zeros([layers_size[1]]))
    if layers[2]:
        weights['h4'] = tf.Variable(tf.truncated_normal(shape=[layers_size[1], layers_size[2]], stddev=1 / np.sqrt(layers_size[1])))
        biases['b4'] = tf.Variable(tf.zeros([layers_size[2]]))
    if layers[3]:
        weights['h5'] = tf.Variable(tf.truncated_normal(shape=[layers_size[2], layers_size[3]], stddev=1 / np.sqrt(layers_size[2])))
        biases['b5'] = tf.Variable(tf.zeros([layers_size[3]]))


    weights['out'] = tf.Variable(tf.truncated_normal(shape=[layers_size[number_of_layers-1], n_classes], stddev=1/np.sqrt(layers_size[number_of_layers-1])))

    pred_outer = dfn(x, layers, weights, biases, keep_prob)

    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
    if dataset == 'SEQC':
        class_weights = tf.constant([1, 3.74])  # 
    elif dataset == 'TARGET':
        class_weights = tf.constant([1, 1.04])  # 80/77
    else:
        print('wrong dataset type!')
        exit()
    cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=pred_outer, targets=y, pos_weight=class_weights))
    if L2:
        reg = tf.nn.l2_loss(weights['h1'])
        if layers[0]:
            reg+=tf.nn.l2_loss(weights['h2'])
        if layers[1]:
            reg+=tf.nn.l2_loss(weights['h3'])
        if layers[2]:
            reg+=tf.nn.l2_loss(weights['h4'])
        if layers[3]:
            reg+=tf.nn.l2_loss(weights['h5'])
        reg+=tf.nn.l2_loss(weights['out'])
        cost = tf.reduce_mean(cost+0.001*reg)
    optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(pred_outer,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    y_score = tf.nn.softmax(logits=pred_outer)

    loss_rec = np.zeros([training_epochs, 1])
    training_eval = np.zeros([training_epochs, 2])

    x_train, x_test = expression[train_index, :], expression[test_index, :]
    y_train, y_test = labels[train_index, :], labels[test_index, :]
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        total_batch = int(np.shape(x_train)[0] / batch_size)


        #earlystopping
        best_cost = 1e9
        best_acc = 0
        stop=False
        last_improvement=0
        required_improvement=50
        costs=[]

        acc, auc, f1 = [], [], []
        # validate_acc, validate_auc = [], []
        test_acc, test_auc, test_f1 = [], [], []

        for epoch in range(training_epochs):
            avg_cost = 0
            # x_tmp, y_tmp = shuffle(x_train, y_train)
            for k in range(total_batch - 1):
                batch_x, batch_y = x_train[k*batch_size:k*batch_size+batch_size], y_train[k*batch_size:k*batch_size+batch_size]
                _, c = sess.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y, keep_prob:0.8, lr:learning_rate})
                avg_cost += c/total_batch
            # del x_tmp
            # del y_tmp


            #early stopping
            # if avg_cost<best_cost:
            # 	save_session = sess
            # 	best_cost = avg_cost
            # 	last_improvement=0
            # else:
            # 	last_improvement+=1
            # if last_improvement>required_improvement:
            # 	print("no improvement, early stopping applies")
            # 	stop = True
            # 	sess = save_session

            acc_, y_s = sess.run([accuracy, y_score], feed_dict={x:x_train, y:y_train, keep_prob:1})
            auc_ = metrics.roc_auc_score(y_train, y_s)
            y_train_label = np.argmax(y_train, axis=1)
            y_s_label = np.argmax(y_s, axis=1)
            f1_ = metrics.f1_score(y_train_label, y_s_label)

            acc.append(acc_)
            auc.append(auc_)
            f1.append(f1_)

            # acc_cv.append(acc_)
            # auc_cv.append(auc_)
            # f1_cv.append(f1_)

            # acc_, y_s = sess.run([accuracy, y_score], feed_dict={x: x_validate, y: y_validate, keep_prob: 1})
            # auc_ = metrics.roc_auc_score(y_validate, y_s)


            if acc_>best_acc:
                save_session = sess
                best_acc = acc_
                last_improvement = 0
            else:
                last_improvement +=1
            if last_improvement>required_improvement:
                print("==============early stopping===================")
                stop = True
                sess = save_session


            # validate_acc.append(acc_)
            # validate_auc.append(auc_)


            acc_, y_s = sess.run([accuracy, y_score], feed_dict={x:x_test, y:y_test, keep_prob:1})
            auc_ = metrics.roc_auc_score(y_test, y_s)
            y_test_label = np.argmax(y_test, axis=1)
            y_s_label = np.argmax(y_s, axis=1)
            f1_ = metrics.f1_score(y_test_label, y_s_label)

            test_acc.append(acc_)
            test_auc.append(auc_)
            test_f1.append(f1_)

            # test_acc_outer.append(acc_)
            # test_auc_outer.append(auc_)
            # test_f1_outer.append(f1_)

            if stop is True:
                break

        acc_outer.append(acc[-1])
        auc_outer.append(auc[-1])
        f1_outer.append(f1[-1])
        test_acc_outer.append(test_acc[-1])
        test_auc_outer.append(test_auc[-1])
        test_f1_outer.append(test_f1[-1])
        
    arr = np.array([k_outer, layers_size, learning_rate, training_epochs, batch_size, acc_outer[-1], auc_outer[-1], f1_outer[-1]])
    log_outer.append(arr)
    print('<<<<<<<<<<<<<<<<< FINISH OUTER FOLD', k_outer, '<<<<<<<<<<<<<<<<<<<')
    
avg_acc = np.array(acc_outer).mean()
avg_auc = np.array(auc_outer).mean()
avg_f1 = np.array(f1_outer).mean()
avg_test_acc = np.array(test_acc_outer).mean()
avg_test_auc = np.array(test_auc_outer).mean()
avg_test_f1 = np.array(test_f1_outer).mean()

std_acc = np.array(acc_outer).std()
std_auc = np.array(auc_outer).std()
std_f1 = np.array(f1_outer).std()
std_test_acc = np.array(test_acc_outer).std()
std_test_auc = np.array(test_auc_outer).std()
std_test_f1 = np.array(test_f1_outer).std()

print('final results:', avg_acc, std_acc, avg_test_acc, std_test_acc, avg_auc, std_auc, avg_test_auc, std_test_auc, avg_f1, std_f1, avg_test_f1, std_test_f1)
df_log_outer = pd.DataFrame(log_outer, columns = ['k_outer', 'layer_size', 'lr', 'training_epochs', 'bs', 'acc', 'auc', 'f1'])
file3 = outpath + preflix + 'outer' + filename2 + postfix

df_log_outer.to_csv(file3)