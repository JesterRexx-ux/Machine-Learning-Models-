#!/usr/bin/env python
# coding: utf-8

# In[56]:


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import stats
from scipy.stats import zscore
from scipy.stats.mstats import winsorize
import warnings


# In[2]:


df=pd.read_csv("edu.csv")
df.head(6)
df.shape
df.info()
df.dtypes
df['Semester'].value_counts()

df['SectionID'].value_counts()


df['gender'].value_counts()

df['gender'].hist(bins=3)
df['ParentschoolSatisfaction'].value_counts()
df['ParentschoolSatisfaction'].hist(bins=4)

fil1=df['Discussion']>70
fil2=df['ParentschoolSatisfaction']=='Good'


# In[49]:


fil=fil1&fil2
df[fil].head()


# In[50]:


df[fil].shape


# In[52]:


df["weighted_rh"]=pd.cut(x=df["raisedhands"], bins=[-1,25,50,75,100], labels=[2,3,4,5])
df["weighted_vr"]=pd.cut(x=df["VisITedResources"], bins=[-1,25,50,75,100], labels=[2,3,4,5])
df["weighted_av"]=pd.cut(x=df["AnnouncementsView"], bins=[-1,25,50,75,100], labels=[2,3,4,5])
df["weighted_dis"]=pd.cut(x=df["Discussion"], bins=[-1,25,50,75,100], labels=[2,3,4,5])

# All weighted variables are category variable type so we should change the type to 'int64'

df["weighted_rh"]=df["weighted_rh"].astype('int64')
df["weighted_vr"]=df["weighted_vr"].astype('int64')
df["weighted_av"]=df["weighted_av"].astype('int64')
df["weighted_dis"]=df["weighted_dis"].astype('int64')

# For each student add a new column called "studentsucces"

df["studentsuccess"]=((df["raisedhands"]*df["weighted_rh"])+(df["VisITedResources"]*df["weighted_vr"])+
                      (df["AnnouncementsView"]*df["weighted_av"])+(df["Discussion"]*df["weighted_dis"]))/(df["weighted_rh"]+df["weighted_vr"]+df["weighted_av"]+df["weighted_dis"])


df.head()


# # Data Visualization

# In[53]:



plt.figure(figsize=(17,10), dpi=100)
plt.subplot(2,3,1)
plt.title("Student Success for Topic")
sns.barplot(df["Topic"],y=df["studentsuccess"], data=df, palette="Greens")
plt.xticks(rotation=45)
plt.ylim(0,85)

plt.subplot(2,3,2)
plt.title("Student Success for NationalITy")
sns.barplot(df["NationalITy"],y=df["studentsuccess"], data=df, palette="Greens")
plt.xticks(rotation=45)
plt.ylim(0,85)

plt.subplot(2,3,3)
plt.title("Student Success for Student Absence Day")
sns.barplot(df["StudentAbsenceDays"],y=df["studentsuccess"], data=df, palette="Greens")
plt.ylabel("Student  Success")
    
plt.subplot(2,3,4)
sns.barplot(x=df["ParentschoolSatisfaction"], y=df["studentsuccess"], data=df, palette="Greens")
plt.xticks(rotation=55)

plt.subplot(2,3,5)
sns.barplot(x=df["Relation"], y=df["studentsuccess"], data=df, palette="Greens")
plt.xticks(rotation=45)

plt.subplot(2,3,6)
sns.barplot(x=df["StageID"], y=df["studentsuccess"], data=df, palette="Greens")
plt.xticks(rotation=45)

plt.show()


# In[57]:


df.groupby("Topic")["studentsuccess"].mean().sort_values().reset_index()

topic_natio=df.groupby(by=["NationalITy","Topic"])["studentsuccess"].mean().reset_index()
topic_natio.head()

fig=px.bar(topic_natio, x="NationalITy", y="studentsuccess", color="Topic")
fig.show()


# In[58]:


plt.figure(figsize=(16,4), dpi=100)

column=["raisedhands","VisITedResources","AnnouncementsView","Discussion","studentsuccess"]
for i in range(len(column)):
    plt.subplot(1,5,i+1)
    plt.title("{}".format(column[i]))
    plt.boxplot(df[column[i]], whis=1.5 )
plt.show() 


# In[60]:


zscores=zscore(df["raisedhands"])
zscores=pd.DataFrame(zscores)

for threshold in range(1,5):
    print("Threshold value is {}: ".format(threshold))
    print("--"*10)
    print("Count of outliers: ",len(np.where(zscores>threshold)[0]),"\n")


# In[61]:


# PERCENTILE METHOD

column=["raisedhands","VisITedResources","AnnouncementsView","Discussion","studentsuccess"]
IQR=[]

for i in range(len(column)):        
    q75,q25=np.percentile(df[column[i]], [75,25])
    IQR.append(q75-q25)

df_interqu=pd.DataFrame(columns=column)
df_interqu=df_interqu.append({"raisedhands":IQR[0],
                              "VisITedResources":IQR[1],
                              "AnnouncementsView":IQR[2],
                              "Discussion":IQR[3],
                              "studentsuccess":IQR[4]}, ignore_index=True)
display(df_interqu)


# # Relationship between Numeric Variables

# In[62]:


plt.figure(figsize=(18,5),dpi=100)

column=["raisedhands","VisITedResources","AnnouncementsView","Discussion"]

plt.figure(figsize=(18,5), dpi=100)
for i in range(len(column)):
    plt.subplot(1,4,i+1)
    sns.scatterplot(y=df[column[i]], x=df["studentsuccess"], data=df)

plt.show()


# In[63]:


correlation=df[["raisedhands","VisITedResources","AnnouncementsView","Discussion","studentsuccess"]].corr()
correlation


# In[64]:


corr_topic=df.groupby("Topic")[["raisedhands","VisITedResources","AnnouncementsView","Discussion","studentsuccess"]].corr()
corr_topic.head(20)


# In[65]:


corr_english=df[df["Topic"]=="English"][["raisedhands","VisITedResources","AnnouncementsView","Discussion","studentsuccess"]].corr()
sns.heatmap(corr_english, annot=True, linewidth=0.5, fmt='.4g')

plt.show()


# In[66]:



pd.options.display.float_format= '{:.15f}'.format

df_satisfied=df["ParentschoolSatisfaction"].unique()

for var in ["raisedhands","VisITedResources","AnnouncementsView","Discussion","studentsuccess"]:
    comparison=pd.DataFrame(columns=["Satisfied","Not_Satisfied","statistic","p_value"])
    print("Comparison for {}".format(var), end='')
    for i in range(0, len(df_satisfied)):
        for j in range(i+1, len(df_satisfied)):
            ttest=stats.ttest_ind(df[df["ParentschoolSatisfaction"]==df_satisfied[i]][var],
                                  df[df["ParentschoolSatisfaction"]==df_satisfied[j]][var])
           
            Satisfied=df_satisfied[i]
            Not_Satisfied=df_satisfied[j]
            statistic=ttest[0]
            p_value=ttest[1]
            
            comparison=comparison.append({"Satisfied":Satisfied,
                                         "Not_Satisfied":Not_Satisfied,
                                         "statistic":statistic,
                                          "p_value":p_value}, ignore_index=True)
    display(comparison)         


# In[67]:


# to test whether there is a difference between the two categorical variables, we can make chiquare in the scipy.stats libraries.
# If p value is less than 0.05, we can accept HA, on the other hand we can say "there is an important difference between the two categorical variables

pd.options.display.float_format= '{:.15f}'.format
natio_topic=pd.crosstab(df["NationalITy"],df["Topic"])

print(stats.chisquare(natio_topic, axis=None))


# In[68]:


pd.options.display.float_format='{:.15f}'.format
natio_stfcn=pd.crosstab(df["ParentschoolSatisfaction"],df["NationalITy"])

print(stats.chisquare(natio_stfcn, axis=None))


# In[69]:


print("Raisedhands & success:","\n",stats.ttest_ind(df["raisedhands"], df["studentsuccess"], equal_var=False),"\n")
print("Visited resources & success:","\n",stats.ttest_ind(df["VisITedResources"], df["studentsuccess"], equal_var=False),"\n")
print("Announcement view & success:","\n",stats.ttest_ind(df["AnnouncementsView"], df["studentsuccess"], equal_var=False),"\n")
print("Discussion & success:","\n",stats.ttest_ind(df["Discussion"], df["studentsuccess"], equal_var=False))


# In[70]:


# ıs there any differences between relation and studentsuccess?

pd.options.display.float_format= '{:.15f}'.format
relation=df["Relation"].unique()


for var in ["raisedhands","VisITedResources","AnnouncementsView","Discussion","studentsuccess"]: #önce sürekli değişkenler
    relation_df=pd.DataFrame(columns=["group_1","group_2","statistic","p_value"])
    print("{}:".format(var),end='')
    for i in range(0,len(relation)): # kategorik değişken
        for j in range(i+1,len(relation)):
            ttest=stats.ttest_ind(df[df["Relation"]==relation[i]][var],
                                  df[df["Relation"]==relation[j]][var])
            
            group_1=relation[i]
            group_2=relation[j]
            statistic=ttest[0]
            p_value=ttest[1]
            
            relation_df=relation_df.append({"group_1":group_1,
                                            "group_2":group_2,
                                            "statistic":statistic,
                                            "p_value":p_value}, ignore_index=True)
    display(relation_df)


# In[71]:


# Is there any differences between StudentAbsenceDays and studentsuccess?

df_absence=df["StudentAbsenceDays"].unique()

for var in ["raisedhands","VisITedResources","AnnouncementsView","Discussion","studentsuccess"]:
    print("{}: ". format(var),end=' ')
    absence_comparison=pd.DataFrame(columns=["group_1","group_2","statistic","p_value"])
    for i in range(0,len(df_absence)):
        for j in range(i+1,len(df_absence)):
            ttest=stats.ttest_ind(df[df["StudentAbsenceDays"]==df_absence[i]][var],
                                 df[df["StudentAbsenceDays"]==df_absence[j]][var])
            
            absence_comparison=absence_comparison.append({"group_1":df_absence[i],
                                                         "group_2":df_absence[j],
                                                         "statistic":ttest[0],
                                                         "p_value":ttest[1]}, ignore_index=True)
    display(absence_comparison)


# # Normality Test

# In[72]:


# Lets use histogram in seaborn libraries to test whether there is a normal distribution.

plt.figure(figsize=(19,4), dpi=100)

column=["raisedhands","VisITedResources","AnnouncementsView","Discussion","studentsuccess"]

for i in range(len(column)):
    plt.subplot(1,5,i+1)
    sns.distplot(df[column[i]])
    plt.ylim(0,0.0175)
plt.show()    
    


# In[73]:



from scipy.stats import jarque_bera

pd.options.display.float_format='{:.8f}'.format

variables=["raisedhands","VisITedResources","AnnouncementsView","Discussion","studentsuccess"]
distributions=pd.DataFrame(columns=["variable","jarque_bera_stats","jarque_bera_pvalue"])

for i in variables:
    jb=jarque_bera(df[i])
    distributions=distributions.append({"variable":i,
                                       "jarque_bera_stats":jb[0],
                                       "jarque_bera_pvalue":jb[1]}, ignore_index=True)

distributions

# H0 --> Data has normal distribution
# H1 --> Data does not have normal distribution

# We accept H1 Hypothesis because of p value is less than 0.05


# In[74]:


# NORMAL TEST:

from scipy.stats import normaltest
pd.options.display.float_format='{:.6f}'.format

variables=["raisedhands","VisITedResources","AnnouncementsView","Discussion","studentsuccess"]
distributions=pd.DataFrame(columns=["variable","normal_test_stats","normal_test_pvalue"])

for i in variables:
    norm=normaltest(df[i])
    distributions=distributions.append({"variable":i,
                                       "normal_test_stats":norm[0],
                                       "normal_test_pvalue":norm[1]}, ignore_index=True)

distributions

# H0 --> Data has normal distribution
# H1 --> Data does not have normal distribution

# We accept H1 Hypothesis because of p value is less than 0.05


# In[75]:


from sklearn.preprocessing import normalize

df["normalized_studentsuccess"]=normalize(np.array(df["studentsuccess"]).reshape(1,-1).reshape(-1,1))

normal_features=["studentsuccess","normalized_studentsuccess"]

print("Minimum Value\n-------------")
print(df[normal_features].min(),"\n")
print("Maximum Value\n-------------")
print(df[normal_features].max())


# In[76]:


sns.distplot(np.log(df["studentsuccess"]));


# # PCA

# In[78]:



df["absence_days"]=pd.get_dummies(df["StudentAbsenceDays"], drop_first=True)
df["mother_father"]=pd.get_dummies(df["Relation"], drop_first=True)
df["satisfaction"]=pd.get_dummies(df["ParentschoolSatisfaction"], drop_first=True)
df["male_female"]=pd.get_dummies(df["gender"], drop_first=True)
df["semester"]=pd.get_dummies(df["Semester"], drop_first=True)


# In[79]:


df.head()


# In[80]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df_new=df[["absence_days","mother_father","satisfaction","male_female","semester","studentsuccess"]]

X=df_new.values


X=StandardScaler().fit_transform(df_new)

pca=PCA(n_components=6)
principalComponents=pca.fit_transform(X)


exp_var=pca.explained_variance_ratio_
cumsum_var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print(exp_var,"\n")
print(cumsum_var)

plt.plot(exp_var)

plt.plot(cumsum_var)

