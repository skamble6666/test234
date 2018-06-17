
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import seaborn as sns


# In[2]:


train = pd.read_csv("D:/Kaggle/Titanic/datasets/train.csv")
test = pd.read_csv('D:/Kaggle/Titanic/datasets/test.csv',dtype={"Age": np.float64})
PassengerId=test['PassengerId']
all_data = pd.concat([train, test], ignore_index = True)


# In[3]:


train.head()


# In[4]:


sns.barplot(x="Sex", y="Survived", data=train)


# In[5]:


print("Percentage of females who survived:%.2f" % (train["Survived"][train["Sex"] == 'female'].mean()))
print("Percentage of males who survived:%.2f" % (train["Survived"][train["Sex"] == 'male'].mean()))
#先将数组限制到Survived这一列，再进行布尔判断，再进行求平均值
#结论：女性幸存率远高于男性


# In[6]:


sns.barplot(x="Pclass", y="Survived", data=train, palette='Set3')


# In[9]:


print("Percentenge of Pclass = 1 who survived:%.2f" % (train["Survived"][train["Pclass"] == 1]).mean())
print("Percentenge of Pclass = 2 who survived:%.2f" % (train["Survived"][train["Pclass"] == 2]).mean())
print("Percentenge of Pclass = 3 who survived:%.2f" % (train["Survived"][train["Pclass"] == 3]).mean())


# In[10]:


#结论：仓等级越高，幸存率越高


# In[11]:


sns.barplot(x="SibSp", y="Survived", data=train)


# In[12]:


sns.barplot(x="Parch", y="Survived", data=train)


# In[13]:


facet = sns.FacetGrid(train, hue="Survived",aspect=2)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()


# In[14]:


#未成年幸存率高于成年


# In[15]:


facet = sns.FacetGrid(train, hue="Survived", aspect=2)
#hue可将数据再进行细分，进行显示
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, 200))
facet.add_legend()


# In[16]:


#船费越高幸存率越高


# In[23]:


all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
#提取文本关键字，来构建新的特征


# In[22]:


all_data['Title']


# In[24]:


sns.barplot(x="Title", y="Survived", data=all_data)


# In[25]:


Title_Dict = {}
Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))
all_data['Title'] = all_data['Title'].map(Title_Dict)
#将文本进行归类映射


# In[26]:


sns.barplot(x="Title", y="Survived", data=all_data, palette='Set3')


# In[27]:


#不同称呼乘客幸存率不同


# In[28]:


all_data['FamilySize']=all_data['SibSp']+all_data['Parch']+1
sns.barplot(x="FamilySize", y="Survived", data=all_data, palette='Set3')
all_data_copy = all_data.copy()
#新增家庭成员数量特征，2-4幸存率高


# In[29]:


def Fam_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif (s > 7):
        return 0


# In[30]:


all_data['FamilyLabel']=all_data['FamilySize'].apply(Fam_label)
#通过构建分类函数和使用apply函数将家庭成员数量分为3类
sns.barplot(x="FamilyLabel", y="Survived", data=all_data)


# In[31]:


all_data['Cabin'] = all_data['Cabin'].fillna('Unknown')
#填补Cabin属性的缺失值


# In[32]:


all_data['Deck'] = all_data['Cabin'].str.get(0)
#获得Cbain中的首字母构建新属性


# In[33]:


sns.barplot(x='Deck', y='Survived', data=all_data)


# In[34]:


Ticket_Count = dict(all_data['Ticket'].value_counts())
all_data['Ticket_Group'] = all_data['Ticket'].apply(lambda x:Ticket_Count[x])
#利用字典进行映射
#先找出同一个票号的对应的数量，然后利用字典映射将票号转换成相同票号的人数
sns.barplot(x='Ticket_Group', y="Survived", data=all_data)


# In[35]:


#2-4人共票号的幸存率较高


# In[36]:


def Ticket_label(s):
    if ((s>=2) & (s<=4)):
        return 2
    elif (((s>4) & (s<=8)) | (s == 1)):
        return 1
    elif (s > 8):
        return 0


# In[37]:


all_data['Ticket_Group'] = all_data["Ticket_Group"].apply(Ticket_label)
sns.barplot(x="Ticket_Group", y= 'Survived', data=all_data)


# In[38]:


corr_matrix = all_data.corr()


# In[39]:


'''
    #数据清洗
'''


# In[40]:


#1.缺失值补充


# In[41]:


#用Sex，Title，Pclass三个特征构建随机森林模型，填补年龄缺失值


# In[42]:


age_df1 = all_data[["Age", "Pclass", 'Sex', 'Title']]
age_df = pd.get_dummies(age_df1)#将分类变量转换为虚拟指标标量


# In[43]:


age_df


# In[44]:


age_df1


# In[45]:


#通过已知数据来估计未知数据


# In[46]:


known_age = age_df[age_df.Age.notnull()].as_matrix()
unknown_age = age_df[age_df.Age.isnull()].as_matrix()
#将DataFrame或Series转换为numpy数组(不是Numpy矩阵)


# In[47]:


known_age


# In[48]:


unknown_age


# In[49]:


age_df[age_df.Age.notnull()]


# In[50]:


y = known_age[:, 0]
X= known_age[:, 1:]


# In[51]:


from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(random_state = 0,n_estimators=100, n_jobs = -1)
rfr.fit(X, y)


# In[52]:


predictedAges = rfr.predict(unknown_age[:, 1::])
all_data.loc[ (all_data.Age.isnull()), 'Age' ] = predictedAges
#利用loc的两个参数，先用布尔类型进行选择，再精准确定Age属性


# In[53]:


all_data.info()


# In[54]:


all_data[all_data.Embarked.isnull()]
#与all_data[all_data['Embarked'].isnull()]不同


# In[55]:


all_data['Embarked'] = all_data['Embarked'].fillna('C')


# In[56]:


sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=all_data)


# In[57]:


all_data[all_data['Fare'].isnull()]


# In[58]:


fare=all_data[(all_data['Embarked'] == "S") & (all_data['Pclass'] == 3)].Fare.median()
all_data['Fare']=all_data['Fare'].fillna(fare)


# In[59]:


all_data.info()


# In[60]:


all_data[all_data.Title.isnull()]


# In[61]:


all_data.head()


# In[62]:


train.iloc[[759]]


# In[64]:


all_data['Title'] = all_data['Title'].fillna('Mrs')


# In[65]:


all_data.info()


# In[66]:


all_data['Surname'] = all_data['Name'].apply(lambda x:x.split(',')[0].strip())


# In[67]:


#提取出姓氏


# In[68]:


Surname_Count = dict(all_data['Surname'].value_counts())
all_data['FamilyGroup'] = all_data['Surname'].apply(lambda x:Surname_Count[x])
#构建姓氏相同的人数作为属性，假设有三个人相同姓氏，则这三个人此属性均为3


# In[69]:


Female_Child_Group=all_data.loc[(all_data['FamilyGroup']>=2) & ((all_data['Age']<=12) | (all_data['Sex']=='female'))]


# In[70]:


Male_Adult_Group=all_data.loc[(all_data['FamilyGroup']>=2) & (all_data['Age']>12) & (all_data['Sex']=='male')]


# In[71]:


#从人数大于1的组中分别提取出每组的妇女儿童和成年男性


# In[72]:


Female_Child_Group


# In[73]:


Female_Child = DataFrame(Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts())


# In[74]:


Female_Child.columns = ['GroupCount']


# In[75]:


Female_Child


# In[76]:


#同姓氏的女性和儿童要么全部幸存，要么全部遇难


# In[77]:


sns.barplot(x=Female_Child.index, y=Female_Child["GroupCount"]).set_xlabel('AverageSurvived')


# In[78]:


Male_Adult = DataFrame(Male_Adult_Group.groupby('Surname')['Survived'].mean().value_counts())


# In[79]:


Male_Adult.columns = ['GroupCount']


# In[80]:


Male_Adult


# In[81]:


sns.barplot(x=Male_Adult.index, y=Male_Adult['GroupCount']).set_xlabel('AverageSurvived')


# In[82]:


#普遍规律是女性和儿童幸存率高，成年男性幸存率低，所以把不符合普遍规律的反常组选出来单独处理


# In[83]:


#把女性儿童组中幸存率为0的组设为遇难组，把成年男性中存活率为1的组设置为幸存族


# In[84]:


Female_Child_Group = Female_Child_Group.groupby('Surname')['Survived'].mean()


# In[85]:


Dead_List = set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)
print(Dead_List)


# In[86]:


Male_Adult_List = Male_Adult_Group.groupby('Surname')['Survived'].mean()
Survived_List=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)
print(Survived_List)


# In[87]:


#为了使处于这两种反常组中的样本能够被正确分类，对测试集中处于反常组中的样本的Age，Title，Sex进行惩罚


# In[88]:


train=all_data.loc[all_data['Survived'].notnull()]


# In[89]:


test=all_data.loc[all_data['Survived'].isnull()]


# In[90]:


import warnings
warnings.filterwarnings('ignore')
#要引入warnings来忽略警告


# In[91]:


test.loc[(test['Surname'].apply(lambda x:x in Dead_List)), 'Sex'] = 'male'
#先通过前一个apply布尔索引定位反常组，再利用'Sex'进行精准定位columns进行修改
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)), 'Age'] = 60
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)), 'Title'] = 'Mr'

test.loc[(test['Surname'].apply(lambda x:x in Survived_List)), 'Sex'] = 'female'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)), 'Age'] = 5
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)), 'Title'] = 'Miss'


# In[92]:


#特征转换：选取特征，转换为数值变量，划分训练集和测试集


# In[93]:


all_data=pd.concat([train, test])
all_data = all_data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilyLabel', 'Deck', 'Ticket_Group']]
all_data=pd.get_dummies(all_data)
train = all_data[all_data['Survived'].notnull()]
test = all_data[all_data['Survived'].isnull()]
X = train.as_matrix()[:,1:]
y = train.as_matrix()[:,0]


# In[94]:


#建模和优化


# In[95]:


#1）利用网格搜索进行自动化选取最佳参数


# In[97]:


from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
pipe=Pipeline([('select',SelectKBest(k=20)), 
               ('classify', RandomForestClassifier(random_state = 10, max_features = 'sqrt'))])

param_test = {'classify__n_estimators':list(range(20,50,2)), 
              'classify__max_depth':list(range(3,60,3))}

gsearch = GridSearchCV(estimator=pipe, param_grid=param_test, scoring='roc_auc', cv=10)
gsearch.fit(X,y)
print(gsearch.best_params_, gsearch.best_score_)


# In[98]:


#训练模型


# In[99]:


select = SelectKBest(k=20)
clf = RandomForestClassifier(random_state=10, warm_start=True,
                            n_estimators=26,
                             max_depth=6,
                             max_features='sqrt'
                            )
pipeline = make_pipeline(select, clf)
pipeline.fit(X,y)
#'sqrt'使得max_feature是auto
#根据K个最高分数选择参数


# In[100]:


#交叉验证


# In[101]:


from sklearn import cross_validation, metrics

cv_score = cross_validation.cross_val_score(pipeline, X, y ,cv=10)
print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))


# In[104]:


test=test.drop('Survived', axis=1)


# In[105]:


predictions = pipeline.predict(test)
submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})
submission.to_csv("D:/Kaggle/Titanic/datasets/submission1.csv", index=False)

