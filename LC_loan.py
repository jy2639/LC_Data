#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 16:11:48 2018

@author: christinayang
"""



import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
import matplotlib.dates as mdates
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold

lending = pd.read_csv('/Users/christinayang/Downloads/loan.csv')
############################## data exploration
# check number of columns 
lending.info()
lending['term']=lending['term'].astype('|S')
lending['issue_d']=pd.to_datetime(lending['issue_d'])
lending['last_pymnt_d']=pd.to_datetime(lending['last_pymnt_d'])
lending_lite.funded_amnt.describe()
# Funded amount, interest rate, term are the three key metrics. check if there are missing values
print len(lending[lending['funded_amnt'].isnull()])
print len(lending[lending['funded_amnt']<0])
print lending.funded_amnt.describe()
print len(lending[lending['int_rate'].isnull()])
print len(lending[lending['int_rate']<=0])
print len(lending[lending['term']==' 36 months'])
print len(lending[lending['term']==' 60 months'])
# check to see if there are duplicates 
dup_check = lending.groupby(['id']).size().to_frame()
print 'there are '+ str(len(dup_check[dup_check[0]!=1]))+' dupliactes'
# % of loans that have funded amount less than loan amount 
ans = len(lending[lending['funded_amnt']!=lending['loan_amnt']])*1.0/len(lending)
print str(ans) + ' of loans have funded amount smaller than loan amount'
####test=lending[lending['funded_amnt']!=lending['loan_amnt']]
# types of loan status 
status=lending['loan_status'].unique().tolist()
# distribution of funded amount 
print 'min funded amount is '+ str(min(lending['funded_amnt']))
print 'max funded amount is '+str(max(lending['funded_amnt'])
print 'avg funede amount is '+ str(lending['funded_amnt'].mean())
lending.boxplot(column='funded_amnt')
lending['funded_amnt'].hist(bins=20)
# trend of average funded amount over time 
amnt_ts = lending.groupby(['issue_d']).agg({'funded_amnt':np.mean})
amnt_ts.index = amnt_ts.index.to_datetime()
amnt_ts=amnt_ts.sort()
amnt_ts.plot()
# trend of loan count over time 
cnt_ts = lending.groupby(['issue_d','purpose']).size().reset_index()
cnt_ts['issue_d'] = pd.to_datetime(cnt_ts['issue_d'])
cnt_mth = cnt_ts.groupby(['issue_d']).agg({0:np.sum})
cnt_mth.plot()
cnt_mth_yy=cnt_mth.pct_change(12)
cnt_mth_yy.plot()
# top purpose 
purpose = cnt_ts.groupby(['purpose']).agg({0:np.sum})
purpose.sort()
cnt_ts=cnt_ts.rename(columns={0:'value'})
## seeing huge volatility in count 
sns.tsplot(cnt_ts[cnt_ts['purpose'].isin(['credit_card','debt_consolidation','home_improvement','major_purpose','other'])], time='issue_d', condition='purpose',unit='purpose', value='value')
# distribution of interest rate 
print 'min interest rate is '+ str(min(lending['int_rate']))
print 'max interest rate is '+ str(max(lending['int_rate']))
print 'avg interest rate is '+str(lending['int_rate'].mean())
lending['int_rate'].hist(bins=20)
## average interest rate by loan grade, distribution of loan grade over time
avg_rate_gd = lending.groupby(['grade','issue_d']).agg({'id':'count','int_rate':np.mean}).reset_index()
avg_rate_gd['issue_d']=pd.to_datetime(avg_rate_gd['issue_d'])
# average interest rate over time 
fig, ax = plt.subplots()
sns.tsplot(avg_rate_gd[['issue_d','grade','int_rate']], time='issue_d', condition='grade',unit='grade', value='int_rate',ax=ax)
plt.show()
## check distribution of loan grade over time 
grad_mth = avg_rate_gd[['grade','issue_d','id']]
grad_mth = grad_mth.sort_values(by='issue_d')
grad_mth_t = grad_mth.groupby(['issue_d']).agg({'id':np.sum})
grad_mth = pd.merge(grad_mth,grad_mth_t,right_index=True,left_on = 'issue_d')
grad_mth['pct']=grad_mth['id_x']/grad_mth['id_y']
grade_plot = grad_mth[['issue_d','grade','pct']]
grade_plot['year'] = grade_plot['issue_d'].astype('str').str[0:4]
annual_grade_pct = grade_plot.groupby(['year','grade']).agg({'pct':np.mean}).reset_index()
sns.tsplot(annual_grade_pct,time='year',condition='grade',unit='grade',value='pct')
table = pd.pivot_table(grade_plot, values='pct', index=['issue_d'],columns=['grade'], aggfunc=np.sum)
fig, ax = plt.subplots()
ax.stackplot(range(0,len(table.index)),  table["A"],  table["B"],  table["C"], table['D'],table['E'],table['F'],table['G'],labels=['A','B','C','D','E','F','G'])
ax.legend(loc='upper left')
plt.show()
############
# remove some columns that are not used 
lending_lite=lending.dropna(thresh=0.9*len(lending), axis=1)
# get 36 months 
lending_36 = lending_lite[lending_lite['term']==' 36 months']
lending_36_f = lending_36[lending_36['loan_status'].isin(['Fully Paid','Charged Off','Default'])]
lending_36_f['issue_d']=pd.to_datetime(lending_36_f['issue_d'])
lending_36_f_early = lending_36_f[lending_36_f['issue_d']<'2013-01-01']
lending_36_f_early['return']=lending_36_f_early['total_pymnt']/lending_36_f_early['funded_amnt']-1
print 'average rate of return for loans originated before 2013-01 is '+str(lending_36_f_early['return'].mean())
# charge off rate before 2013
print 'charge of rate for loans issued before 2013 is '+ str(len(lending_36_f_early[lending_36_f_early['loan_status'].isin(['Charged Off','Default'])])*1.0/len(lending_36_f_early))
lending_36_f_early['length']=lending_36_f_early['last_pymnt_d']-lending_36_f_early['issue_d']
lending_36_f_early[lending_36_f_early['loan_status']=='Fully Paid']['length'].mean()
##### Cohort 
# before 2013 return 
lending_36_f_early['issue_d_str']=lending_36_f_early['issue_d'].astype('str')
lending_36_return = lending_36_f_early.groupby([lending_36_f_early.issue_d_str.str[0:4],'grade']).agg({'return':np.mean}).reset_index()
ret_table = pd.pivot_table(lending_36_return,values=0,index='issue_d_str',columns='grade',aggfunc=np.sum)
lending_36_return = lending_36_f_early.groupby([lending_36_f_early.issue_d_str.str[0:4],'grade']).agg({'funded_amnt':np.sum,'total_pymnt':np.sum}).reset_index()
lending_36_return['wret']=lending_36_return['total_pymnt']/lending_36_return['funded_amnt']-1
ret_table = pd.pivot_table(lending_36_return,values='wret',index='issue_d_str',columns='grade',aggfunc=np.sum)
sns.heatmap(ret_table)
####################### Model 
good_loan =  len(lending_36_f_early[(lending_36_f_early.loan_status == 'Fully Paid')])
print 'Good Loan Ratio is ' + str(good_loan*1.0/len(lending_36_f_early))
# create binary variable 
df = lending_36_f_early.copy()
df['good_loan']= np.where((df.loan_status == 'Fully Paid'), 1, 0)
# create binary for open_acc
df['open_acc_b'] = np.where((df.open_acc>=10),1,0)
# create binary for delinq_2yrs 
df['delinq_2yrs_b']=np.where((df.delinq_2yrs>0),1,0)
# selecting features 
data = df[['good_loan','annual_inc','delinq_2yrs_b','emp_length',
           'home_ownership','open_acc_b','grade','revol_bal','funded_amnt','id','return','total_pymnt']]
data = data[~data['home_ownership'].isin(['NONE','OTHER'])]
data = data[data['emp_length']!='n/a']
# create dummy variables 
mapping = {'< 1 year':'0-3 year','1 year':'0-3 year','2 years':'0-3 year','3 years':'0-3 year',
           '4 years':'4-6 year','5 years':'4-6 year','6 years':'4-6 year','7 years':'7-9 year',
           '8 years':'7-9 year','9 years':'7-9 year','10+ years': '10+ years'}
data['emp_l_grp']=data.emp_length.map(mapping)
cat_vars = ['emp_l_grp','home_ownership','grade']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var,drop_first=True)
    data1=data.join(cat_list)
    data=data1
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
data_final=data[to_keep]
data_final=data_final.drop(['emp_length'],axis=1)
data_final=data_final.dropna(axis=0)

#### try to build some ratios as features 
data_final['revol_pct']=data_final['revol_bal']/data_final['annual_inc']
data_final['fund_pct']=data_final.funded_amnt/data_final['annual_inc']

# scale continuous variable 
scaler = StandardScaler()
#scaled_x=scaler.fit_transform(data_final[['annual_inc','delinq_2yrs','revol_bal','funded_amnt']])
scaled_x=scaler.fit_transform(data_final[['annual_inc','revol_bal','funded_amnt','revol_pct','fund_pct']])
cat_var=[col for col in data_final.columns if col not in ['annual_inc','delinq_2yrs','revol_bal',
                                                          'funded_amnt','id','return','revol_pct','fund_pct','total_pymnt']]
cat_x = data_final[cat_var]
id_ret = data_final[['id','return','total_pymnt','funded_amnt']]
x = np.concatenate([cat_x.values, scaled_x,id_ret], axis=1)
n_data = pd.DataFrame(data=x,columns=cat_var+['annual_inc','delinq_2yrs','revol_bal','funded_amnt_n','revol_pct','fund_pct','id','return','total_pymnt','funded_amnt'])
#RFE 
y=['good_loan','id','return','revol_bal','funded_amnt','total_pymnt','funded_amnt_n','delinq_2yrs']
X = [i for i in n_data.columns if i not in y]
y=['good_loan']
#logreg = LogisticRegression()
#rfe = RFE(logreg, 20)
#rfe = rfe.fit(x, data_final[y].values.ravel())
#b_lst= rfe.support_
# implementing model 
x = sm.add_constant(n_data[X], prepend=False)
logit_model=sm.Logit(n_data[y],x)
result=logit_model.fit()
print(result.summary2())
########################## train/test 
y=n_data[['good_loan','return','funded_amnt','total_pymnt']]
#x=n_data.ix[:,n_data.columns!='good_loan']
#x=n_data.ix[:,n_data.columns.isin(['grade_B','grade_C','grade_D','grade_E','grade_F',
 #                                  'grade_G','funded_amnt','annual_inc','open_acc_b'])]
x=n_data.ix[:,n_data.columns.isin(['grade_B','grade_C','grade_D','grade_E','grade_F',
                                   'grade_G','fund_pct','revol_pct','open_acc_b','annual_inc'])]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=44)
logreg=LogisticRegression(class_weight='balanced')
clf=logreg.fit(x_train,y_train['good_loan'])
predictions = clf.predict(x_test)
prob = clf.predict_proba(x_test)
ret_map = np.column_stack([prob[:,1],y_test['funded_amnt'].values,y_test['total_pymnt'].values])
#ret_df = pd.DataFrame(data=ret_map,columns=['prob','return'])
ret_df = pd.DataFrame(data=ret_map,columns=['prob','funded_amnt','total_pymnt'])
print np.mean(ret_df['return'])
tmp = ret_df[ret_df['prob']>0.8]
print np.mean(tmp['return'])
#evaluate model 
y_train.mean()
score = clf.score(x_test,y_test['good_loan'])
print score 
#confusion matrix 
cf=confusion_matrix(y_test['good_loan'],predictions)
print cf 
print(classification_report(y_test['good_loan'],predictions))
# roc, auc curve 
logit_roc_auc = roc_auc_score(y_train['good_loan'], clf.predict(x_train))
fpr, tpr, thresholds = roc_curve(y_train['good_loan'], clf.predict_proba(x_train)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
################### k fold cross validation 
#n_data_f = n_data[['good_loan','open_acc_b','grade_B','grade_C','grade_D','grade_E','grade_F',
#                 'grade_G','revol_pct','fund_pct','annual_inc','return']]
n_data_f = n_data[['good_loan','open_acc_b','grade_B','grade_C','grade_D','grade_E','grade_F',
                 'grade_G','revol_pct','funded_amnt_n','annual_inc','return']]
kf = KFold(n_splits=10,shuffle=True)
auc_l = []
ret_all = []
for train, test in kf.split(n_data_f):
    train_data = np.array(n_data_f)[train]
    test_data = np.array(n_data_f)[test]
    logreg=LogisticRegression(class_weight='balanced')
    clf=logreg.fit(train_data[:,1:11],train_data[:,0])
    logit_roc_auc = roc_auc_score(test_data[:,0], clf.predict(test_data[:,1:11]))
    auc_l.append(logit_roc_auc)
    prob = clf.predict_proba(test_data[:,1:11])
    ret_map = np.column_stack([prob[:,1],test_data[:,-1]])
    ret_df = pd.DataFrame(data=ret_map,columns=['prob','return'])
    raw_ret = np.mean(ret_df['return'])
    #ret_map = np.column_stack([prob[:,1],test_data[:,11:13]])
    #ret_df = pd.DataFrame(data=ret_map,columns=['prob','funded_amnt','total_pymnt'])
    #raw_ret = ret_df['total_pymnt'].sum()/ret_df['funded_amnt'].sum()-1
    ret_dic = {}
    for threshold in [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]:
        tmp = ret_df[ret_df['prob']>threshold]
        #ret_diff = tmp['total_pymnt'].sum()/tmp['funded_amnt'].sum()-1-raw_ret
        ret_diff = np.mean(tmp['return'])-raw_ret
        ret_dic[threshold]=ret_diff
    ret_all.append(ret_dic)
        
    #pred = clf.predict(test_data[:,1:11])
    #cf=confusion_matrix(test_data[:,0],pred)
    #print(classification_report(test_data[:,0],pred))
    #print logit_roc_auc
print 'average auc is ' + str(np.mean(auc_l))

avg_ret = {}
for num in [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]:
    res = 0
    for l in ret_all: 
        res += l[num]
    avg_ret[num]=res/len(ret_all)
        
        


