#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>  
# 
# <div class="toc"><ul class="toc-item">
# <li><span><a href="#安装依赖包" data-toc-modified-id="安装依赖包-1">安装依赖包</a></span></li>
# <li><span><a href="#数据清洗" data-toc-modified-id="数据清洗-2">数据清洗</a></span></li>
# <li><span><a href="#数据可视化" data-toc-modified-id="数据可视化-3">数据可视化</a></span></li>
# <li><span><a href="#构建猪肉价格预测模型" data-toc-modified-id="构建猪肉价格预测模型-4">构建猪肉价格预测模型</a></span>
#     <ul class="toc-item">
#     <li><span><a href="#多特征模型训练（多元线性回归）" data-toc-modified-id="多特征模型训练（多元线性回归）-4.1">多特征模型训练（多元线性回归）</a></span></li>
#     <li><span><a href="#假设验证法选出最佳特征组合" data-toc-modified-id="假设验证法选出最佳特征组合-4.2">假设验证法选出最佳特征组合</a></span></li>
#     </ul>
# </li>
# <li><span><a href="#猪肉价格的预测" data-toc-modified-id="猪肉价格的预测-5">猪肉价格的预测</a></span></li>
# </ul></div>

# # 安装依赖包

# In[144]:


get_ipython().system('pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple/')


# # 数据清洗

# In[142]:


import numpy as np
import pandas as pd


# In[109]:


data = pd.read_csv('/Users/xiaozi/Desktop/猪肉数据.csv') 
data.head()


# In[110]:


#列名只保留英文
data.columns = [
    'year',
    'pork_price',
    'sow_stock',
    'corn_price',
    'bean_meal_price',
    'hog_out_price',
    'pork_import'
]

print(data.columns)
print(data.head())


# In[111]:


# 删除index列（用del更方便）
data.drop('year',axis=1,inplace=True) 
data.head()


# In[112]:


#过滤异常值（非洲猪瘟 outliers）
df = data[data['sow_stock'] >= 4000] 


# # 数据可视化

# In[113]:


data['sow_stock']


# In[114]:


import matplotlib.pyplot as plt
sow = data['sow_stock']
price = data['pork_price']
plt.scatter(sow,price)
plt.show() # 有离群点数据，对线性分析不利，需要过滤


# In[115]:


df = data[data['sow_stock'] >=4000] # 正常住宅面积小于等于300平米
sow = df['sow_stock']
price = df['pork_price']
#print(sow.count()) #过滤后的数据量
plt.scatter(sow,price)
plt.xlabel("sow")
plt.ylabel("price")
plt.show()


# # 构建猪肉价格预测模型

# In[116]:


# 先根据可育母猪存栏和平均单价训练模型（一元线性回归）
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
sow = np.array(sow).reshape(-1,1) 
price = np.array(price).reshape(-1,1)
# 训练模型
model = linear.fit(sow,price)
# 打印截距和回归系数
print(model.intercept_, model.coef_)


# In[117]:


# 线性回归可视化(数据拟合)
linear_p = model.predict(sow)
plt.figure(figsize=(12,6))
plt.scatter(sow,price)
plt.plot(sow,linear_p,'red')
plt.xlabel("sow")
plt.ylabel("price")
plt.show()


# ## 多特征模型训练（多元线性回归）

# In[118]:


X = df[['sow_stock', 
        'corn_price', 
        'bean_meal_price', 
        'hog_out_price', 
        'pork_import']]


# In[119]:


y = df['pork_price']


# In[121]:


print(type(X))
print(type(y))
# 使用train_test_split进行交叉验证
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=12)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# In[122]:


# 模型训练
linear = LinearRegression()
model = linear.fit(x_train,y_train)
print(model.intercept_, model.coef_)


# In[123]:


# 模型性能评分
price_end = model.predict(x_test)
score = model.score(x_test,y_test) 
print("模型得分：",score)# 一般模型在0.6以上就表现的不错


# ## 假设验证法选出最佳特征组合

# In[124]:


# 使用假设验证法，选出最佳特征组合
cols = ['sow_stock', 
        'corn_price', 
        'bean_meal_price', 
        'hog_out_price', 
        'pork_import']
import statsmodels.api as sm
Y = df['pork_price']
X = df[cols]
X_ = sm.add_constant(X) #增加一列值为1的const列，保证偏置项的正常
#print(X_)
# 使用最小平方法
result = sm.OLS(Y,X_)
# 使用fit方法进行计算
summary = result.fit()
# 调用summary2方法打印出假设验证信息（性能指标）
summary.summary2() # R-squared:模型评分 AIC：组合完越小越好


# #### 名词解释
# - coef 回归系数
# - Std.Err 标准差
# - t 虚无假设成立时的t值
# - P>|t| 虚无假设成立时的概率值
# - [0.025,0.975] 97.5%置信估计区间
# - 要做假设性验证，首先要设置显著性标准。
# - a. 假设显著性标准是0.01
# - b. 推翻虚无假设的标准是p<0.01
# - c. 上面的SqFt的t=9.2416，P(>5) = 0.0000 < 0.01,因此虚无假设被推翻（这里的虚无假设是SqFt对price的回归系数为0，即SqFt与price不相关）
# 
# ---
# 
# #### F统计
# - 回归平方和 Regression Square Sum[RSS]: 依变量的变化归咎于回归模型 A=sum((y-y_)^2)
# - 误差平方和 Error Square Sum[ESS]: 依变量的变化归咎于线性模型 B=sum((y-y_)^2)
# - 总的平方和 Total Square Sum[TSS]: 依变量整体变化 C=A+B
# - 回归平方平均 Model Mean Square:=RSS/Regression d.f(k) k=自变量的数量
# - 误差平方平均 Error Mean square:=ESS/Error d.f(n-k-1) n=观测值得数量
# - F统计 F=Model Mean Square /Error Mean Square
# - F值越大越好，Prob(F-statistic)越小越好
# 
# ---
# 
# #### R Square
# - 回归可以解释变量比例，可以作为自变量预测因变量准确度的指标
# - SSE(残差平方和) = sum((y-y_)^2)
# - SST(整体平方和) = sum((yi-yavg)^2)
# - R^2 = 1-SEE/SST 一般要大于0.6,0.7才算好
# 
# ---
# 
# #### Adjust R Square
# - R^2 = 1-SSE/SST SSE最小，推导出R^2不会递减
# - yi = b1x1 + b2x2 +...+bkxk+...增加任何一个变量都会增加R^2
# - Adj R^2 = 1-(1-R^2) * ((n-1)/(n-p-1))
# - n为总体大小，p为回归因子个数
# 
# ---
# 
# #### AIC/BIC
# - AIC(The Akaike Information Criterion)= 2K + nln(SSE/n) K是参数数量，n是观察数，SSE是残差平方和。
# - AIC鼓励数据拟合的优良性，但是应该尽量避免过拟合，所以优先考虑的模型应该是AIC最小的那一个。
# - 赤池信息量的准则是寻找可以最好的解释数据但是包含最少自由参数的模型。

# In[68]:


import itertools

list1 = [1, 2,3, 4, 5,6,7,8,9,10,11,12,13,14,15,16] #特征超过16个将发生异常
list2 = []
for i in range(1, len(list1)+1):
    iter1 = itertools.combinations(list1, i)
    list2.append(list(iter1))
#print(list2)


# In[125]:


import itertools
# 使用itertools，找出AIC最小值的特征组合作为模型训练的特征
# 寻找最小AIC值的特征组合
fileds = ['sow_stock', 
        'corn_price', 
        'bean_meal_price', 
        'hog_out_price', 
        'pork_import']
acis = {}
for i in range(1,len(fileds)+1):
    for virables in itertools.combinations(fileds,i): #从fileds中随机选择i个特征机型组合，返回的virables为元组类型
        x1 = sm.add_constant(df[list(virables)])
        x2 = sm.OLS(Y,x1)
        res = x2.fit()
        acis[virables] = res.aic # AIC评分越小越好


# In[126]:


from collections import Counter
# 对字典进行统计
counter = Counter(acis)
# 降序选出AIC最小的5个数，也就是最佳特征组合
counter.most_common()[-5:] 


# In[128]:


# 接下来使用AIC值最小的特征组合进行预测
col2 = ['sow_stock','corn_price', 'hog_out_price', 'pork_import']
X = df[col2]
y = df['pork_price']

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=13)
linear = LinearRegression()
model = linear.fit(x_train,y_train)
model.score(x_test,y_test) 


# # 猪肉价格的预测

# #### 现在我们可以根据给定的最佳特征组合进行预测猪肉

# In[141]:


# 输入4个特征预测
p_price = [3950, 2550, 16, 78]
predict_data = np.array(p_price).reshape(1, -1)
pred = model.predict(predict_data)
# 输出结果
print("预测猪肉价格 =", pred[0])

