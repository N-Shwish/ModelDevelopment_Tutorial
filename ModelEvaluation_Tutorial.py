#MODEL EVALUATION AND REFINEMENT
#Evaluating and adjusting prediction models to better fit
import pandas as pd
import numpy as np

# Import clean data
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv'
df = pd.read_csv(path)
df.to_csv('module_5_auto.csv')

df=df._get_numeric_data() #Only collects the numeric data
df.head()
#Should display the first 5 rows x all 21 columns
%%capture
! pip install ipywidgets #These are Libraries for plotting
from ipywidgets import interact, interactive, fixed, interact_manual
#Now, we build the functions for plotting
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()

def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    #training data
    #testing data
    # lr:  linear regression object
    #poly_transform:  polynomial transformation object
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)

    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()
#FIRST, we must split our data into training and testing data.
    #We will place the target data price in a seperate dataframe y:
y_data = df['price']
x_data=df.drop('price',axis=1) #Drop price data in x data
#Randomly split our data using train_test_split
from sklearn.model_selection import train_test_split
#Below, we are setting 40% of our data into "testing data"
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.4, random_state=0)
print("number of test samples :", x_test1.shape[0])
print("number of training samples:",x_train1.shape[0])

#Now, we will import LinearRegression from the module linear_model
from sklearn.linear_model import LinearRegression
lre=LinearRegression() #Creates a Linear Regression object.
lre.fit(x_train1[['horsepower']], y_train1) #Fits the model using the feature 'horsepower'
#We will now calculate the R^2 on the test data:
lre.score(x_test1[['horsepower']], y_test1)
lre.score(x_train1[['horsepower']], y_train1)

#Sometimes you do not have sufficient testing data... in which you may want to try Cross-validation

#CROSS VALIDATION SCORE
from sklearn.model_selection import cross_val_score
#We input the object, in this case, feature = 'horsepower', the target data = (y_data)
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4) #parameter 'cv' determines the number of folds; in this case 4.
Rcross #default scoring is R^2. each element in the array has the avg R^2 value in the fold.
#Calculate the avg and std dev of our estimate:
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())
#We can use negative squared error as a score by setting the parameter 'scoring' metric to 'neg_mean_squared_error'.
-1 * cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error')

#Here, we will calculate the avg R^2 using two folds, and find the avg R^2 for the 2nd fold using 'horsepower' as the feature.
Rc=cross_val_score(lre,x_data[['horsepower']], y_data,cv=2)
Rc.mean()

#You can also use the function 'cross_val_predict' to predict the output.
from sklearn.model_selection import cross_val_predict
yhat = cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4) #Number of folds = 4.
yhat[0:5]

#OVERFITTING, UNDERFITTING, AND MODEL SELECTION:
#We will create multiple linear regression objects.
#We will then train the model using 'horsepower', 'curb-weight', 'engine-size', and 'highway-mpg' as features.
lr = LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)
#Prediction using the training data:
yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_train[0:5]
#Prediction using test data:
yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_test[0:5]

#Now some Model evaluation: Using our training and testing data seperately:
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)

Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)
#This will show that the Distribution of the test data in Figure 1 is better fitted.
#Figure 2 shows a difference where the ranges are from 5000 to 15,000.
#We will now check if polynomial regression also exhibits prediction inaccuracy when analysing the test data.
from sklearn.preprocessing import PolynomialFeatures

#Overfitting occurs when the model fits the noise, but not the underlying process.
#This will cause the model to not perform well when using the test-set, since it will be modeling noise,
    #Not the the underlying process that generated the relationship.
#We will create a fifth degree polynomial model:
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0) #Using 45% of data for testing
#We will perform the 5th degree polynomial transformation on the feauture 'horsepower'.
pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])
pr
#Now, create a linear regression model and train it.
poly = LinearRegression()
poly.fit(x_train_pr, y_train)
#see the output of our model using predict.
yhat = poly.predict(x_test_pr)
yhat[0:5] #Assign the values to "Yhat"
#Take the first five predicted values and compare to the actual targets
print("Predicted values:", yhat[0:4])
print("True values:", y_test[0:4].values)
#Use "PollyPlot" function to display the training data, testining data, and the predicted model.
PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly,pr)
#At around 200 horsepower, the prediction model begins to diverge from the points
#R^2 of training data:
poly.score(x_train_pr, y_train) #This is low... Lower score = Worse model
#R^2 of test data:
poly.score(x_test_pr, y_test) #This is negative... Negative R^2 is a sign of Overfitting
#See how the R^2 changes on the test data for different order polynomials, and plot the results:
Rsqu_test = []
order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)

    x_train_pr = pr.fit_transform(x_train[['horsepower']])

    x_test_pr = pr.fit_transform(x_test[['horsepower']])

    lr.fit(x_train_pr, y_train)

    Rsqu_test.append(lr.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')
#This plot shows R^2 increases until an order three polynomial is used. Then R^2 dramatically decreases at four.

#This function will allow you to experiment with different polynomial orders and different amts of data.
def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train,y_test, poly, pr)
interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05))

#Creating a Polynomial Transformation with more than one feature...
pr1=PolynomialFeatures(degree=2)
pr1
#Transforming the training and testing samples for the four features:
x_train_pr1=pr1.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
x_test_pr1=pr1.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

x_train_pr1
x_test_pr1
#Check how many dimensions the new feature has (using ".shape")
x_train_pr1.shape #Shows there are now 15 features

#Create a linear regression model "poly1" & train the object using "fit" using the polynomial features...
poly1=LinearRegression().fit(x_train_pr1,y_train)
poly1
#Use "predict" to predict the output on the polynomial features... then display the distribution of the predicted vs test data.
yhat_test1=poly1.predict(x_test_pr1)

Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test, yhat_test1, "Actual Values (Test)", "Predicted Values (Test)", Title)
#The predicted value is higher than actual value for cars where the price $10,000 range,
    #conversely the predicted price is lower than the price cost in the $30,000 to $40,000 range.
    #As such the model is not as accurate in these ranges.

#RIDGE REGRESSION
#perform a degree two polynomial transformation on our data:
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
from sklearn.linear_model import Ridge

RidgeModel=Ridge(alpha=0.1) #Creates a Ridge Regression Object, setting regularization parameter to 0.1
#Fit the model
RidgeModel.fit(x_train_pr, y_train)
yhat = RidgeModel.predict(x_test_pr) #Obtain a prediction
#Now, Compare the first five predicted samples to our test set:
print('predicted:', yhat[0:4])
print('test set :', y_test[0:4].values)
#We can use a for loop to select the value of Alpha that minimizes the test error:
Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = 10 * np.array(range(0,1000))
for alpha in Alpha:
    RidgeModel = Ridge(alpha=alpha)
    RidgeModel.fit(x_train_pr, y_train)
    Rsqu_test.append(RidgeModel.score(x_test_pr, y_test))
    Rsqu_train.append(RidgeModel.score(x_train_pr, y_train))
#We can plot out the R^2 value for diff Alphas:
width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(Alpha,Rsqu_test, label='validation data  ')
plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()
#We will now Perform Ridge Regression and calculate R^2 using the polynomial features
#We will train the model using the training data
#We will test the model using the testing data
RidgeModel = Ridge(alpha=10)
RidgeModel.fit(x_train_pr, y_train)
RidgeModel.score(x_test_pr, y_test)

#GRID SEARCH
#Alpha is a hyperparameter
#sklearn has the class GridSearchCV to make finding the best hyperparameter simpler
from sklearn.model_selection import GridSearchCV
parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]}] #Creating a dictionary of parameter values
parameters1

RR=Ridge() #Creates a ridge regions object
RR
Grid1 = GridSearchCV(RR, parameters1,cv=4) #Creates a ridge grid search object
#Now, fit the model
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
#Obtain the estimator with the best parameters and assign it to the variable BestRR below:
BestRR=Grid1.best_estimator_
BestRR
#Now, test our model on the test data:
BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)

#Perform a Grid Search for the Alpha parameter and the NORMALIZATION parameter, finding the best values of the parameters.
parameters2= [{'alpha': [0.001,0.1,1, 10, 100, 1000,10000,100000,100000],'normalize':[True,False]} ]
Grid2 = GridSearchCV(Ridge(), parameters2,cv=4)
Grid2.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_data)
Grid2.best_estimator_
