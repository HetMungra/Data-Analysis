#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
missing_values = ["n/a", "na", "--","NaN",]

dataset = pd.read_csv("KDD_train_20.csv", na_values = missing_values, names=['Duration','Protocol_Type','Service','Flag','Src Bytes','Dst_Bytes','Land','Wrong_Fragment','Urgent','Hot','Num Failed Logins','Logged In','Num Compromised','Root Shell','Su Attempted','Num Root','Num File Creations','Num Shells','Num Access Files','Num Outbound Cmds','Is Hot Logins','Is Guest Login','Count','Srv Count','Serror Rate','Srv Serror Rate','Rerror Rate','Srv Rerror Rate','Same Srv Rate','Diff Srv Rate','Srv Diff Host Rate','Dst Host Count','Dst Host Srv Count','Dst Host Same Srv Rate','Dst Host Diff Srv Rate','Dst Host Same Src Port Rate','Dst Host Srv Diff Host Rate','Dst Host Serror Rate','Dst Host Srv Serror Rate','Dst Host Rerror Rate','Dst Host Srv Rerror Rate','Label','Score']
)


#print(dataset.dtypes)
#dataset= pd.read_csv('OSX_Dataset_R.csv')
print(dataset.shape)
#print(dataset.describe())
#dataset.head(100)

#Protocol_Type string to int
dataset["Protocol_Type"] = dataset["Protocol_Type"].astype('category')
dataset["Protocol_Type_code"] = dataset["Protocol_Type"].cat.codes
#Service string to int
dataset["Service"] = dataset["Service"].astype('category')
dataset["Service_code"] = dataset["Service"].cat.codes
#Service flag to int
dataset["Flag"] = dataset["Flag"].astype('category')
dataset["Flag_code"] = dataset["Flag"].cat.codes
#Label string to int

dataset.Label[dataset.Label == 'normal'] = 0
dataset.Label[dataset.Label == 'neptune'] = 1
dataset.Label[dataset.Label == 'warezclient'] = 1
dataset.Label[dataset.Label == 'ipsweep'] = 1
dataset.Label[dataset.Label == 'portsweep'] = 1
dataset.Label[dataset.Label == 'teardrop'] = 1
dataset.Label[dataset.Label == 'nmap'] = 1
dataset.Label[dataset.Label == 'satan'] = 1
dataset.Label[dataset.Label == 'smurf'] = 1
dataset.Label[dataset.Label == 'pod'] = 1
dataset.Label[dataset.Label == 'back'] = 1
dataset.Label[dataset.Label == 'guess_passwd'] = 1
dataset.Label[dataset.Label == 'ftp_write'] = 1
dataset.Label[dataset.Label == 'multihop'] = 1
dataset.Label[dataset.Label == 'rootkit'] = 1
dataset.Label[dataset.Label == 'buffer_overflow'] = 1
dataset.Label[dataset.Label == 'imap'] = 1
dataset.Label[dataset.Label == 'warezmaster'] = 1
dataset.Label[dataset.Label == 'phf'] = 1
dataset.Label[dataset.Label == 'land'] = 1
dataset.Label[dataset.Label == 'loadmodule'] = 1
dataset.Label[dataset.Label == 'spy'] = 1
dataset.Label[dataset.Label == 'perl'] = 1
dataset['Label'] = dataset['Label'].astype('int')

#df.drop_duplicates(['col1','col2'])[['col1','col2']]

#dataset['Label'] = dataset['Label'].astype('int')


feature_cols = ['Duration','Protocol_Type_code','Service_code','Flag_code','Src Bytes','Dst_Bytes','Land','Wrong_Fragment','Urgent','Hot','Num Failed Logins','Logged In','Num Compromised','Root Shell','Su Attempted','Num Root','Num File Creations','Num Shells','Num Access Files','Num Outbound Cmds','Is Hot Logins','Is Guest Login','Count','Srv Count','Serror Rate','Srv Serror Rate','Rerror Rate','Srv Rerror Rate','Same Srv Rate','Diff Srv Rate','Srv Diff Host Rate','Dst Host Count','Dst Host Srv Count','Dst Host Same Srv Rate','Dst Host Diff Srv Rate','Dst Host Same Src Port Rate','Dst Host Srv Diff Host Rate','Dst Host Serror Rate','Dst Host Srv Serror Rate','Dst Host Rerror Rate','Dst Host Srv Rerror Rate','Score']
int_features = ['Flag_code','Srv Serror Rate','Same Srv Rate','Logged In','Dst Host Srv Serror Rate','Dst Host Same Srv Rate','Dst Host Serror Rate','Dst Host Same Srv Rate','Dst Host Serror Rate','Dst Host Same Src Port Rate','Protocol_Type_code','Score','Dst Host Diff Srv Rate','Srv Rerror Rate','Dst Host Srv Count','Service_code','Serror Rate','Label']
int_dataset_featues= dataset[int_features] 
int_dataset = dataset[feature_cols]
X = dataset[feature_cols]
y = dataset['Label'] # Target variable


print(X.dtypes)
#print(dataset) 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
# print (X)
print('whole dataset shape:', X.shape)
#print('train dataset shape: ', X_train.shape)
#print('test dataset shape: ', X_test.shape)





# In[2]:


#print(featureScores.nlargest(13,'Score'))  #print 10 best features

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')
print(feat_importances.nlargest(42))


# In[3]:


import seaborn as sns
int_features = ['Flag_code','Srv Serror Rate','Same Srv Rate','Logged In','Dst Host Srv Serror Rate','Dst Host Same Srv Rate','Dst Host Serror Rate','Dst Host Same Src Port Rate','Protocol_Type_code','Score','Dst Host Diff Srv Rate','Srv Rerror Rate','Dst Host Srv Count','Service_code','Serror Rate','Label']
int_dataset_featues= dataset[int_features] 
plt.figure(figsize=(16, 14))
cmap = sns.diverging_palette(222, 10, as_cmap=True)
_ = sns.heatmap(int_dataset_featues.corr(), annot=True, vmax=.8, square=True, cmap=cmap)


# In[17]:


dataset["Protocol_Type_code"] = dataset["Protocol_Type_code"].astype('category')
dataset["Flag_code"] = dataset["Flag_code"].astype('category')
feature_cols = ['Flag_code','Srv Serror Rate','Same Srv Rate','Logged In','Dst Host Srv Serror Rate','Dst Host Same Srv Rate','Dst Host Serror Rate','Dst Host Same Srv Rate','Dst Host Serror Rate','Dst Host Same Src Port Rate','Protocol_Type_code','Score','Dst Host Diff Srv Rate','Srv Rerror Rate','Dst Host Srv Count','Service_code','Serror Rate']
#int_dataset = dataset[feature_cols]
X = dataset[feature_cols]
y = dataset['Label'] # Target variable


print(X.dtypes)
#print(dataset) 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


# # Normal Approach Result 

# In[5]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from time import time
from sklearn.metrics import precision_score,recall_score,f1_score



dataset["Protocol_Type_code"] = dataset["Protocol_Type_code"].astype('category')
dataset["Flag_code"] = dataset["Flag_code"].astype('category')
#feature_cols = ['Duration','Protocol_Type_code','Service_code','Flag_code','Src Bytes','Dst_Bytes','Land','Wrong_Fragment','Urgent','Hot','Num Failed Logins','Logged In','Num Compromised','Root Shell','Su Attempted','Num Root','Num File Creations','Num Shells','Num Access Files','Num Outbound Cmds','Is Hot Logins','Is Guest Login','Count','Srv Count','Serror Rate','Srv Serror Rate','Rerror Rate','Srv Rerror Rate','Same Srv Rate','Diff Srv Rate','Srv Diff Host Rate','Dst Host Count','Dst Host Srv Count','Dst Host Same Srv Rate','Dst Host Diff Srv Rate','Dst Host Same Src Port Rate','Dst Host Srv Diff Host Rate','Dst Host Serror Rate','Dst Host Srv Serror Rate','Dst Host Rerror Rate','Dst Host Srv Rerror Rate','Score']
#int_dataset = dataset[feature_cols]
feature_cols = ['Flag_code','Srv Serror Rate','Same Srv Rate','Logged In','Dst Host Srv Serror Rate','Dst Host Same Srv Rate','Dst Host Serror Rate','Dst Host Same Srv Rate','Dst Host Serror Rate','Dst Host Same Src Port Rate','Protocol_Type_code','Score','Dst Host Diff Srv Rate','Srv Rerror Rate','Dst Host Srv Count','Service_code','Serror Rate']

X = dataset[feature_cols]
y = dataset['Label'] # Target variable


#print(X.dtypes)
#print(dataset) 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

scaler= MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

cv_report_normal = []
random_state=40


start = time()
clf_lr= LogisticRegression(random_state=random_state).fit(X_train,y_train)
'''
start = time()
scores = cross_val_score(clf_lr, X_train, y_train, cv=5, scoring='roc_auc')
end =time()
duration = end- start
cv_report_normal.append(['LogisticRegression', scores.mean(), duration])
'''
#print('LogisticRegression: Accuracy in training: {:.4f}'.format(clf_lr.score(X_train,y_train)))
print('LogisticRegression:Accuracy in testing: {:.4f}'.format(clf_lr.score(X_test,y_test)))
end =time()
duration = end- start
print(duration)

start = time()
clf_svc= SVC(random_state=random_state,probability= True).fit(X_train,y_train)
'''
start = time()
scores = cross_val_score(clf_svc, X_train, y_train, cv=5, scoring='roc_auc')
end =time()
duration = end- start
cv_report_normal.append(['SVC', scores.mean(), duration])
'''
#print('SVC:Accuracy in training: {:.4f}'.format(clf_svc.score(X_train,y_train)))
print('SVC:Accuracy in testing: {:.4f}'.format(clf_svc.score(X_test,y_test)))
end =time()
duration = end- start
print(duration)

start = time()
clf_dt= DecisionTreeClassifier(random_state=random_state).fit(X_train,y_train)
'''
start = time()
scores = cross_val_score(clf_dt, X_train, y_train, cv=5, scoring='roc_auc')
end =time()
duration = end- start
cv_report_normal.append([' DecisionTree', scores.mean(), duration])
'''
#print('DecisionTreeClassifier:Accuracy in training: {:.4f}'.format(clf_dt.score(X_train,y_train)))
print('DecisionTreeClassifier:Accuracy in testing: {:.4f}'.format(clf_dt.score(X_test,y_test)))
end =time()
duration = end- start
print(duration)

start = time()
clf_rf= RandomForestClassifier(random_state=random_state).fit(X_train,y_train)
'''
start = time()
scores = cross_val_score(clf_rf, X_train, y_train, cv=5, scoring='roc_auc')
end =time()
duration = end- start
cv_report_normal.append(['RandomForest', scores.mean(), duration])
'''
#print('RandomForestClassifier:Accuracy in training: {:.4f}'.format(clf_rf.score(X_train,y_train)))
print('RandomForestClassifier:Accuracy in testing: {:.4f}'.format(clf_rf.score(X_test,y_test)))
end =time()
duration = end- start
print(duration)


start = time()
clf_nb= GaussianNB().fit(X_train,y_train)
'''
start = time()
scores = cross_val_score(clf_nb, X_train, y_train, cv=5, scoring='roc_auc')
end =time()
duration = end- start
cv_report_normal.append(['GaussianNB', scores.mean(), duration])
#print('Naivebias:Accuracy in training: {:.4f}'.format(clf_nb.score(X_train,y_train)))
'''
print('Naivebias:Accuracy in testing: {:.4f}'.format(clf_nb.score(X_test,y_test)))
end =time()
duration = end- start
print(duration)

'''
cv_report_normal = pd.DataFrame(cv_report_normal, columns=['classifier', 'mean_score', 'time'])
cv_report_normal = cv_report_normal.ix[:,0:3]
print(cv_report_normal.sort_values('mean_score', ascending=False))
'''

gamma = 'auto'
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def Accuracy():
    tp, fp, fn, tn = confusion_matrix(y_test,pred).reshape(-1)
    
#     Precision
    precision_score = tp / (tp + fp)
    print('Precision Accuracy: %.4f' % precision_score)
#     Recall

    recall_score = tp / (tp + fn)
    print('Recall Accuracy: %.4f' % recall_score)
    
#     F1 Score
    f1_score = 2*tp / (2*tp + fp + fn)
    print('F1 Score Accuracy: %.4f' % f1_score)
    
#     FPR
    FPR = fp / (fp + tn)
    print('False Positive Rate: %.6f' % FPR)
    
#     FNR
    FNR = fn / (fn + tp)
    print('False Negative  Rate: %.6f' % FNR)
    
    
print('Logistic regression:')
pred= clf_lr.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred,  digits=4))
Accuracy()

print('Support Vector Machines:')
pred= clf_svc.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred,  digits=4))
Accuracy()

print('Decision tree:')
pred= clf_dt.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred,  digits=4))
Accuracy()

print('Random Forest:')
pred= clf_rf.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred,  digits=4))
Accuracy()

print('Naive Bayes:')
pred= clf_nb.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred,  digits=4))
Accuracy()


# In[6]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
pred_lr= clf_lr.predict_proba(X_test)
pred_svc= clf_svc.predict_proba(X_test)
pred_rf= clf_rf.predict_proba(X_test)
pred_nb= clf_nb.predict_proba(X_test)
#lr_probsdt = clfdt.predict_proba(X_testdt)
#lr_probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
ns_probs = [0 for _ in range(len(y_test))]
pred_lr = pred_lr[:, 1]
pred_svc = pred_svc[:, 1]
pred_rf = pred_rf[:, 1]
pred_nb = pred_nb[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, pred_lr)
svc_auc = roc_auc_score(y_test, pred_svc)
rf_auc = roc_auc_score(y_test, pred_rf)
nb_auc = roc_auc_score(y_test, pred_nb)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic Regression: ROC AUC=%.3f' % (lr_auc))
print('Support Vector: ROC AUC=%.3f' % (svc_auc))
print('Random Forest: ROC AUC=%.3f' % (rf_auc))
print('Naive Bayes: ROC AUC=%.3f' % (nb_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, pred_lr)
svc_fpr, svc_tpr, _ = roc_curve(y_test, pred_svc)
rf_fpr, rf_tpr, _ = roc_curve(y_test, pred_rf)
nb_fpr, nb_tpr, _ = roc_curve(y_test, pred_nb)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic Regression')
pyplot.plot(svc_fpr, svc_tpr, marker='.', label='Support Vector')
pyplot.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest')
pyplot.plot(nb_fpr, nb_tpr, marker='.', label='Naive Bayes')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


# In[7]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

pca_data= int_dataset
#print(pca_data.head())
x_std= StandardScaler().fit_transform(pca_data)
pca = PCA(n_components=15)
pca_transform = pca.fit_transform(x_std)
#pca.explained_variance_
np.cumsum(pca.explained_variance_ratio_)
#pca_transform


# In[8]:


from sklearn.decomposition import PCA
pca = PCA(n_components=4)
pca = pca.fit(pca_data)

reduced_data = pca.transform(pca_data)
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2', 'Dimension 3', 'Dimension 4'])


# In[9]:


from sklearn import mixture
from sklearn.metrics import silhouette_score

#score_list = []
#score_columns = []
preds = {}
centers = {}
sample_preds = {}

for n in range(4,1,-1):
    #print "Calculating clusters with {} dimensions.".format(n)
    clusterer = mixture.GaussianMixture(n_components=n)
    y_kmeans=clusterer.fit(pca_transform)

    preds[n] = clusterer.predict(pca_transform)
    centers[n] = clusterer.means_
    '''
    score = silhouette_score(reduced_data, preds[n], metric='euclidean')
    score_list.append(score)
    score_columns.append(str(n) + " components")

score_list = pd.DataFrame(data=[score_list],columns=score_columns, index=['Silhouette Score'])
score_list
'''


# In[10]:


import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9., 5.)
    plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


# In[11]:


plot_results(pca_transform, y_kmeans.predict(pca_transform), y_kmeans.means_, y_kmeans.covariances_, 0,
             'Gaussian Mixture')


# In[12]:


no_clusters = 4
predictions = pd.DataFrame(preds[no_clusters], columns = ['Cluster'])
dataset['cluster_allocated'] = predictions
dataset.head()


# In[13]:


dataset["cluster_allocated"] = dataset["cluster_allocated"].astype('category')
dataset["Protocol_Type_code"] = dataset["Protocol_Type_code"].astype('category')
dataset["Flag_code"] = dataset["Flag_code"].astype('category')
feature_cols = ['Flag_code','Srv Serror Rate','Same Srv Rate','Logged In','Dst Host Srv Serror Rate','Dst Host Same Srv Rate','Dst Host Serror Rate','Dst Host Same Srv Rate','Dst Host Serror Rate','Dst Host Same Src Port Rate','Protocol_Type_code','Score','Dst Host Diff Srv Rate','Srv Rerror Rate','Dst Host Srv Count','Service_code','Serror Rate','cluster_allocated']
#feature_cols = ['Duration','Protocol_Type_code','Service_code','Flag_code','Src Bytes','Dst_Bytes','Land','Wrong_Fragment','Urgent','Hot','Num Failed Logins','Logged In','Num Compromised','Root Shell','Su Attempted','Num Root','Num File Creations','Num Shells','Num Access Files','Num Outbound Cmds','Is Hot Logins','Is Guest Login','Count','Srv Count','Serror Rate','Srv Serror Rate','Rerror Rate','Srv Rerror Rate','Same Srv Rate','Diff Srv Rate','Srv Diff Host Rate','Dst Host Count','Dst Host Srv Count','Dst Host Same Srv Rate','Dst Host Diff Srv Rate','Dst Host Same Src Port Rate','Dst Host Srv Diff Host Rate','Dst Host Serror Rate','Dst Host Srv Serror Rate','Dst Host Rerror Rate','Dst Host Srv Rerror Rate','Score','cluster_allocated']
#int_dataset = dataset[feature_cols]
X = dataset[feature_cols]
y = dataset['Label'] # Target variable


print(X.dtypes)
#print(dataset) 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


# In[14]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')
print(feat_importances.nlargest(42))


# # Hybrid Approach Result

# In[15]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from time import time

scaler= MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

cv_report_clus = []
random_state=1


start = time()
clf_lr= LogisticRegression(random_state=random_state).fit(X_train,y_train)
'''
start = time()
scores = cross_val_score(clf_lr, X_train, y_train, cv=5, scoring='roc_auc')
end =time()
duration = end- start
cv_report_normal.append(['LogisticRegression', scores.mean(), duration])
'''
#print('LogisticRegression: Accuracy in training: {:.4f}'.format(clf_lr.score(X_train,y_train)))
print('LogisticRegression:Accuracy in testing: {:.4f}'.format(clf_lr.score(X_test,y_test)))
end =time()
duration = end- start
print(duration)

start = time()
clf_svc= SVC(random_state=random_state,probability= True).fit(X_train,y_train)
'''
start = time()
scores = cross_val_score(clf_svc, X_train, y_train, cv=5, scoring='roc_auc')
end =time()
duration = end- start
cv_report_normal.append(['SVC', scores.mean(), duration])
'''
#print('SVC:Accuracy in training: {:.4f}'.format(clf_svc.score(X_train,y_train)))
print('SVC:Accuracy in testing: {:.4f}'.format(clf_svc.score(X_test,y_test)))
end =time()
duration = end- start
print(duration)

start = time()
clf_dt= DecisionTreeClassifier(random_state=random_state).fit(X_train,y_train)
'''
start = time()
scores = cross_val_score(clf_dt, X_train, y_train, cv=5, scoring='roc_auc')
end =time()
duration = end- start
cv_report_normal.append([' DecisionTree', scores.mean(), duration])
'''
#print('DecisionTreeClassifier:Accuracy in training: {:.4f}'.format(clf_dt.score(X_train,y_train)))
print('DecisionTreeClassifier:Accuracy in testing: {:.4f}'.format(clf_dt.score(X_test,y_test)))
end =time()
duration = end- start
print(duration)

start = time()
clf_rf= RandomForestClassifier(random_state=random_state).fit(X_train,y_train)
'''
start = time()
scores = cross_val_score(clf_rf, X_train, y_train, cv=5, scoring='roc_auc')
end =time()
duration = end- start
cv_report_normal.append(['RandomForest', scores.mean(), duration])
'''
#print('RandomForestClassifier:Accuracy in training: {:.4f}'.format(clf_rf.score(X_train,y_train)))
print('RandomForestClassifier:Accuracy in testing: {:.4f}'.format(clf_rf.score(X_test,y_test)))
end =time()
duration = end- start
print(duration)


start = time()
clf_nb= GaussianNB().fit(X_train,y_train)
'''
start = time()
scores = cross_val_score(clf_nb, X_train, y_train, cv=5, scoring='roc_auc')
end =time()
duration = end- start
cv_report_normal.append(['GaussianNB', scores.mean(), duration])
#print('Naivebias:Accuracy in training: {:.4f}'.format(clf_nb.score(X_train,y_train)))
'''
print('Naivebias:Accuracy in testing: {:.4f}'.format(clf_nb.score(X_test,y_test)))
end =time()
duration = end- start
print(duration)

'''
cv_report_normal = pd.DataFrame(cv_report_normal, columns=['classifier', 'mean_score', 'time'])
cv_report_normal = cv_report_normal.ix[:,0:3]
print(cv_report_normal.sort_values('mean_score', ascending=False))
'''

gamma = 'auto'
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def AccuracyHybrid():
    tp, fp, fn, tn = confusion_matrix(y_test,pred).reshape(-1)
    
#     Precision
    precision_score = tp / (tp + fp)
    print('Precision Accuracy: %.4f' % precision_score)
    
#     Recall
    recall_score = tp / (tp + fn)
    print('Recall Accuracy: %.4f' % recall_score)
    
#     F1 Score
    f1_score = 2*tp / (2*tp + fp + fn)
    print('F1 Score Accuracy: %.4f' % f1_score)
    
#     FPR
    FPR = fp / (fp + tn)
    print('False Positive Rate: %.6f' % FPR)
    
#     FNR
    FNR = fn / (fn + tp)
    print('False Negative  Rate: %.6f' % FNR)

print('Logistic Regression:')
pred= clf_lr.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred, digits=4))
AccuracyHybrid()

print('Support Vector Machines:')
pred= clf_svc.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred,  digits=4))
AccuracyHybrid()

print('Decision Tree:')
pred= clf_dt.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred,  digits=4))
AccuracyHybrid()

print('Random Forest:')
pred= clf_rf.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred,  digits=4))
AccuracyHybrid()

print('Naive Bayes:')
pred= clf_nb.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred,  digits=4))
AccuracyHybrid()



# In[16]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
pred_lr= clf_lr.predict_proba(X_test)
pred_svc= clf_svc.predict_proba(X_test)
pred_rf= clf_rf.predict_proba(X_test)
pred_nb= clf_nb.predict_proba(X_test)
#lr_probsdt = clfdt.predict_proba(X_testdt)
#lr_probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
ns_probs = [0 for _ in range(len(y_test))]
pred_lr = pred_lr[:, 1]
pred_svc = pred_svc[:, 1]
pred_rf = pred_rf[:, 1]
pred_nb = pred_nb[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, pred_lr)
svc_auc = roc_auc_score(y_test, pred_svc)
rf_auc = roc_auc_score(y_test, pred_rf)
nb_auc = roc_auc_score(y_test, pred_nb)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic Regression: ROC AUC=%.3f' % (lr_auc))
print('Support Vector: ROC AUC=%.3f' % (svc_auc))
print('Random Forest: ROC AUC=%.3f' % (rf_auc))
print('Naive Bayes: ROC AUC=%.3f' % (nb_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, pred_lr)
svc_fpr, svc_tpr, _ = roc_curve(y_test, pred_svc)
rf_fpr, rf_tpr, _ = roc_curve(y_test, pred_rf)
nb_fpr, nb_tpr, _ = roc_curve(y_test, pred_nb)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic Regression')
pyplot.plot(svc_fpr, svc_tpr, marker='.', label='Support Vector')
pyplot.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest')
pyplot.plot(nb_fpr, nb_tpr, marker='.', label='Naive Bayes')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

