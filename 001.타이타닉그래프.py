
# coding: utf-8

# In[1]:



get_ipython().run_line_magic('matplotlib', 'nbagg')
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# In[2]:


train = pd.read_csv('../Data/train.csv', header = 0)


# In[3]:


# 위의 5개만 보기
train.head()


# In[60]:


# 아래 5개만 보기
names.tail()


# In[59]:


# 위의 10개만 보기
names.head(10)


# In[58]:


# 전체 갯수
names.count() # 예를들어 gender에 널값이 있다면 널값을 뺀 갯수를 보여준다.


# In[56]:


# 년도별 남아, 여아 출생건수
total_births = names.pivot_table('Survived', index='Age', columns='Sex', aggfunc=sum)


# In[57]:


total_births.tail(10)


# In[21]:


# 그래프로 그리기
total_births.plot(title='Survive')


# In[5]:


# 정리된 data로 csv file 만들기
names2 = pd.DataFrame(names, columns=['id','name','year','gender','births']) # 컬럼 순서 바꾸기


# In[6]:


names2.head()


# In[ ]:


# csv 저장
names2.to_csv('./Data/births_names.csv', index = False, header = True) 
# index = False를 꼭 써주어야 한다. False를 안해주면 저장될때 인덱스 값도 저장
# 되기 때문에 엑셀에 인덱스값도 컬럼하나로 잡혀서 들어가게 된다.
# header = True를 하지 않으면 첫번째 행값이 헤더로 들어가기 때문에 문제가 생긴다.


# In[ ]:


names2.head()


# In[111]:


HeadList = names.columns.values.tolist()
HeadList


# In[149]:


x1


# In[138]:


x1=names['Pclass']
x2=names['Age']
x=np.concatenate((x1,x2),axis=1)
x


# In[ ]:



df3=pd.DataFrame(x,columns=['x1','x2'])
df3.head()


# In[ ]:


plt.scatter(df3['x1'],df3['x2'])


# In[ ]:


#-----------


# In[150]:


HeadList


# In[ ]:


names['Survived']&names['Survived']


# In[162]:


names.loc[:,['Survived','Age']]


# In[19]:


child=(train['Age']<10).sum()
one=((train['Age']>=10) & (train['Age']<20)).sum()
two=((train['Age']>=20) & (train['Age']<30)).sum()
three=((train['Age']>=30) & (train['Age']<40)).sum()
four=((train['Age']>=40) & (train['Age']<50)).sum()
five=((train['Age']>=50) & (train['Age']<60)).sum()
six=((train['Age']>=60) & (train['Age']<70)).sum()
seven=((train['Age']>=70) & (train['Age']<80)).sum()
eight=(train['Age']>=80).sum()
ages=[child,one,two,three,four,five,six,seven,eight]


# In[20]:


childsur=((train['Age']<10) & (train['Survived']==1)).sum()
onesur=((train['Age']>=10) & (train['Age']<20)& (train['Survived']==1)).sum()
twosur=((train['Age']>=20) & (train['Age']<30)& (train['Survived']==1)).sum()
threesur=((train['Age']>=30) & (train['Age']<40)& (train['Survived']==1)).sum()
foursur=((train['Age']>=40) & (train['Age']<50)& (train['Survived']==1)).sum()
fivesur=((train['Age']>=50) & (train['Age']<60)& (train['Survived']==1)).sum()
sixsur=((train['Age']>=60) & (train['Age']<70)& (train['Survived']==1)).sum()
sevensur=((train['Age']>=70) & (train['Age']<80)& (train['Survived']==1)).sum()
eightsur=((train['Age']>=80) & (train['Survived']==1)).sum()


# In[22]:


ti0="%0.1f%%" %((childsur/child)*100)
ti1="%0.1f%%" %((onesur/one)*100)
ti2="%0.1f%%" %((twosur/two)*100)
ti3="%0.1f%%" %((threesur/three)*100)
ti4="%0.1f%%" %((foursur/four)*100)
ti5="%0.1f%%" %((fivesur/five)*100)
ti6="%0.1f%%" %((sixsur/six)*100)
ti7="%0.1f%%" %((sevensur/seven)*100)
ti8="%0.1f%%" %((eightsur/eight)*100)


# In[26]:


aList=np.arange(0,90,10)
aList


# In[27]:


df=pd.DataFrame(
    [childsur,onesur,twosur,threesur,foursur,fivesur,sixsur,sevensur,eightsur],
    columns=['count'],
    index=aList
)
df


# In[28]:


df.plot(kind='bar')


# In[ ]:


df2=pd.DataFrame(np.random.rand(6,4),
                index=['one','two','three','four','five','six'],
                columns=pd.Index(['A','B','C','D'],name='BoxRing')
                 
                )
df2


# In[183]:


((train['Pclass']==1) & (train['Embarked']=='S')).sum()


# In[179]:


train['Fare'].mean()


# In[36]:


Class1Smean = train.loc[(train.Pclass==1)&(train.Embarked=='S'),'Fare'].mean()
Class1Qmean = train.loc[(train.Pclass==1)&(train.Embarked=='Q'),'Fare'].mean()
Class1Cmean = train.loc[(train.Pclass==1)&(train.Embarked=='C'),'Fare'].mean()
Class2Smean = train.loc[(train.Pclass==2)&(train.Embarked=='S'),'Fare'].mean()
Class2Qmean = train.loc[(train.Pclass==2)&(train.Embarked=='Q'),'Fare'].mean()
Class2Cmean = train.loc[(train.Pclass==2)&(train.Embarked=='C'),'Fare'].mean()
Class3Smean = train.loc[(train.Pclass==3)&(train.Embarked=='S'),'Fare'].mean()
Class3Qmean = train.loc[(train.Pclass==3)&(train.Embarked=='Q'),'Fare'].mean()
Class3Cmean = train.loc[(train.Pclass==3)&(train.Embarked=='C'),'Fare'].mean()


# In[47]:


EmFare=pd.DataFrame(
    [[Class1Smean,Class2Smean,Class3Smean],
    [Class1Qmean,Class2Qmean,Class3Qmean],
    [Class1Cmean,Class2Cmean,Class3Cmean]]
)
EmFare.index=['S','Q','C']
EmFare.columns=['FirstClass','SecondClass','ThirdClass']
EmFare=EmFare.T
EmFare.plot(kind='bar')


# In[49]:


df=pd.DataFrame(

[(train['Survived']==1).sum(),(train['Survived']==0).sum()],
        index=['survival','death'],
        columns=['count']
)
df.T


# In[53]:


df.plot(kind='bar')


# In[7]:


import seaborn as sns


# In[8]:


sns.set()


# In[9]:


def bar_chart(feature):
    survived=train[train['Survived']==1][feature].value_counts()
    dead=train[train['Survived']==0][feature].value_counts()
    df=pd.DataFrame([survived, dead])
    df.index=['Survived','Dead']
    df.plot(kind='bar',stacked=True,figsize(10,5))


# In[10]:


bar_chart('Sex')

