#MODEL DEVELOPMENT
#Developing prediction models using Python
#In order to help us predict future observations from the data we have
#A model can help us understand the exact relationship b/n different variables
#these relationships between variables are used to predict the result

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#We'll be using the same collected Automobile data set used in the other Walkthrough
#path of data
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
df = pd.read_csv(path)
df.head()

#SIMPLE LINEAR REGRESSION
#relationship between the independant variable and the dependant variable. (X & Y)

#First, load the modules for linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm #Creates the linear regression object
#In this example, we're going to look at how highway-mpg can help us predict car price.
X = df[['highway-mpg']]
Y = df['price']
lm.fit(X,Y)  #This will fit the linear model using highway-mpg.
#We can output a prediction
Yhat=lm.predict(X)
Yhat[0:5]
lm.intercept_  #gives us the value of intercept (a)
lm.coef_       #gives us the value of slope(b)
#Now we will create a Linear Regression Object with this data
lm1 = LinearRegression()
lm1
#We can train the model using 'engine-size' as the independent variable and 'price' as the dependent variable
lm1.fit(df[['engine-size']], df[['price']])
lm1
#Find the value of the slope and the intercept of the model
lm1.coef_
lm1.intercept_
#So, Yhat = -7963.34 + 166.86*X , where Yhat is Price and X is 'engine-size'
Yhat=-7963.34 + 166.86*X
print(Yhat)

#MULTIPLE LINEAR REGRESSION
#In this example, we'll use MLR to predict car price using more than one variable
#This will explain the relationship b/n one continues DVariable and two or more IVariables(Predictors)
#In MLR, Yhat=(intercept) + ((coefficients of Variable 1,b1)*(Predictor Variable 1,X1)) + (b2X2) + (b3X3)...
#In this exercise, our Predictor Variables will be 'horsepower', 'curb-weight', 'engine-size', 'highway-mpg'

#First, develop a model using these variables as the predictor variables:
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z, df['price']) #Fit the linear model using the four variables
lm.intercept_ #The value of intercept(a)
lm.coef_      #The value of the coefficients (b1, b2, b3, b4)
#So we can input these numbers into the formula, where we can create and train a MLR model, "lm2"
#The response variable of "lm2" (it's Y) will be price, The predictor variable is 'normalized-losses' and 'highway-mpg'
lm2 = LinearRegression()
lm2.fit(df[['normalized-losses' , 'highway-mpg']],df['price'])
lm2.coef_ #This will give us the coefficient of the model
#NOW we will Evaluate the Model through Visualization of our data!
# import the visualization package: seaborn
import seaborn as sns
%matplotlib inline

#Regression Plot, visualizing highway-mpg as the potential predictor variable of price
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
#Now, we can build a similar regression plot using 'peak-rpm' as the predictor variable, and compare their coorelations with "price"
plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
#We can use .corr() to verify which IV or Predictor Variable is more strongly coorelated with "price"
df[["peak-rpm","highway-mpg","price"]].corr()
#This data will show that "highway-mpg"'s correlation with "price" is closer to -1 than that of "peak-rpm"...
    #...showing more of a coorelation or impact to "price"

#RESIDUAL PLOT
#This plot can visualize the variance of the data
#The Residual(e) shows the difference b/n the observed value(y) and the predicted value(Yhat)
#The Residual(e) is the distance from the data point to the fitted regression line

#If the points in a residual plot are randomly spread out around the x-axis, then a linear model is appropriate for the data...
    #...Randomly spread out residuals means that the variance is constant, and thus the linear model is a good fit for this data.
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['highway-mpg'], df['price'])
plt.show()
#We can see from this residual plot that the residuals are not randomly spread around the x-axis,
    #...maybe a non-linear model is more appropriate for this data.
    #...This means we can try visualizing the MLR model with the distribution plot.
#First, we make a prediction(Yhat)
Y_hat = lm.predict(Z)
plt.figure(figsize=(width, height))

ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)

plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()
#The fitted values are close to the actual values, but there can be a little more accuracy using another plot...
#Fitting a Polynomila model to the data instead:

#This will plot the data:
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()
#Now, we get the variables:
x = df['highway-mpg']
y = df['price']
#Fitting the polynomial using the function polyfit
# Here we use a polynomial of the 3rd order (cubic)
f = np.polyfit(x, y, 3)
p = np.poly1d(f) #poly1d to display the polynomial function
print(p)
#Now, we plot the function
PlotPolly(p, x, y, 'highway-mpg')
np.polyfit(x, y, 3)
#This model performs better than the linear model. The generated function hits more of the data points

#Here we will Create a polynomial of the 11th order (cubic)
f1 = np.polyfit(x, y, 11)
p1 = np.poly1d(f1)
print(p1)
PlotPolly(p1,x,y, 'Highway MPG')

#We can perform a Polynomial Transform on multiple features
#Import the Module:
from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=2) #Creating a PolynomialFeatures object of degree 2
pr
Z_pr=pr.fit_transform(Z)
Z.shape #This data will have 201 samples and 4 features
Z_pr.shape #Adding the transormation, there will now be 201 samples and 15 features

#DATA PIPELINES
#Simplifying the steps of processing the data.
from sklearn.pipeline import Pipeline #creates the pipeline
from sklearn.preprocessing import StandardScaler #a step in our pipeline
#Create a pipeline here by creating a list of tuples including the name of the model or estimator and its corresponding constructor.
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
pipe
#Now, we can normalize the data, perform a transform, and fit the model simultaneously..
pipe.fit(Z,y)
#We can also normalize the data, perform a transform, and produce a prediction simultaneously..
ypipe=pipe.predict(Z)
ypipe[0:4]

#Create a pipeline that Standardizes the data, then perform prediction using a linear regression model using features Z and targets y
Input=[('scale',StandardScaler()),('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(Z,y)

ypipe=pipe.predict(Z)
ypipe[0:10]

#MEASURES FOR IN-SAMPLE EVALUATION
#R^2 - coefficient of determination, indicates how close data is to the fitted regression line.
#(MSE)Mean Squared Error - measures the difference b/n actual value (y) and the estimated value

#In Simple Linear Regression
#Calulating R^2
#highway_mpg_fit
lm.fit(X, Y)
# Find the R^2
print('The R-square is: ', lm.score(X, Y))

#Calculate MSE
Yhat=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)

#In Multiple Linear Regression:
#Calculating R^2:
# fit the model
lm.fit(Z, df['price'])
# Find the R^2
print('The R-square is: ', lm.score(Z, df['price']))

#Calculating MSE:
Y_predict_multifit = lm.predict(Z) #Producing a prediction
#Compare predicted results with actual results:
print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))

#In Polynomial Fit
#Calculating r^2
from sklearn.metrics import r2_score #different function is used for this
r_squared = r2_score(y, p(x)) #Applying the function to get the value of R^2
print('The R-square value is: ', r_squared)

#Calculating MSE:
mean_squared_error(df['price'], p(x)) #Also done from this function

#Now we'll pyplot and use numpy to visualize and produce a prediction:
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline

new_input=np.arange(1, 100, 1).reshape(-1, 1) #Creates a new input
lm.fit(X, Y) #Fit the model
lm
yhat=lm.predict(new_input) #Produce a prediction
yhat[0:5]
#Now, Plot the data:
plt.plot(new_input, yhat)
plt.show()
#Higher R^2 value = better fit for the data
#Smallest MSE value = better fit for the data
#with the above models we have created, the MLR model is the best model to predict price from our dataset!
#the dataset has 27 variables in total, we know more than one of these variables are potential predictors of the final car price.
