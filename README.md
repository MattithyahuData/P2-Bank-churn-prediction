# 🏦 Bank Churn Prediction: Project Overview  
* End to end project researching the effects customer attributes have on the churn of a bank customer and predicting those customers that may churn.
* Optimised Logistic Regression, KNN, SVC, Decision Tree, and Random Forest Classifiers using GridsearchCV to reach the best model. 
* Built a stakeholder facing visual deployment of model to predict churn of new customers 
* Deployed Model in Power BI for Business Intelligence analysis 

[View Deployed Model in Power BI](https://app.powerbi.com/view?r=eyJrIjoiNDExYjQ0OTUtNWI5MC00OTQ5LWFlYmUtYjNkMzE1YzE2NmE0IiwidCI6IjYyZWE3MDM0LWI2ZGUtNDllZS1iZTE1LWNhZThlOWFiYzdjNiJ9&pageName=ReportSection)

## Table of Contents 
*   [Resources](#resources)<br>
*   [Data Collection](#DataCollection)<br>
*   [Data Pre-processing](#DataPre-processing)<br>
*   [Data Warehousing](#DataWarehousing)<br>
*   [Exploratory data analysis](#EDA)<br>
*   [Data Visualisation & Analytics](#Dataviz)<br>
*   [Business Intelligence](#Busintelli)<br>
*   [Feature Engineering](#FeatEng)<br>
*   [ML/DL Model Building](#ModelBuild)<br>
*   [Model performance](#ModelPerf)<br>
*   [Model Optimisation](#ModelOpt)<br>
*   [Model Evaluation](#ModelEval)<br>
*   [Model Productionising](#ModelProd)<br>
*   [Deployment](#ModelDeploy)<br>
*   [Project Management (Agile/Scrum/Kanban)](#Prjmanage)<br>
*   [Project Evaluation](#PrjEval)<br>
*   [Looking Ahead](#Lookahead)<br>
*   [Questions & Contact me](#Questions)<br>


<a name="resources"></a>   
         
## Resources Used
**Python 3, PostgreSQL, Power BI** 

[**Anaconda Packages:**](requirements.txt) **pandas numpy pandas_profiling ipywidgets sklearn matplotlib seaborn sqlalchemy kaggle psycopg2 ipykernel** <br><br>
Powershell command for installing anaconda packages used for this project  
```powershell
pip install pandas numpy pandas_profiling ipywidgets sklearn matplotlib seaborn sqlalchemy kaggle psycopg2 ipykernel
```
<a name="DataCollection"></a>  

## [Data Collection](Code/P2_Code.ipynb)
Powershell command for data import using kaggle API <br>
```
!kaggle datasets download -d kmalit/bank-customer-churn-prediction -p ..\Data --unzip 
```
[Data source link](https://www.kaggle.com/kmalit/bank-customer-churn-prediction)
[Data](Data/Churn_Modelling.csv)
*  Rows: 10000  / Columns: 14
    *   RowNumber
    *   CustomerId  
    *   Surname 
    *   CreditScore
    *   Geography
    *   Gender
    *   Age
    *   Tenure
    *   Balance
    *   NumOfProducts
    *   HasCrCard
    *   IsActiveMember
    *   EstimatedSalary
    *   Exited

<a name="DataPre-processing"></a>  

## [Data Pre-processing](Code/P2_Code.ipynb)
After I had all the data I needed, I needed to check it was ready for exploration and later modelling. I made the following changes and created the following variables:   
*   General NULL and data validity checks  
*   Formatted column headers to SQL compatibility. 
The data contained no null values and all datatypes lined up with their field description. <br>

```python
# Viewing the data types of the columns
data.dtypes

# Viewing dataset shape
data.shape

# 1st check for null values and datatype check 
data.info()
```

<br>

*   Some programming languages can be case sensitive like python and C++ for example, so using lower case letters for variable names allows for straightforward use of data in different programming languages.<br>

```python
# (SQL standard) Formatting column headers by removing potential capital letters and spaces in column headers 
data.columns = data.columns.str.lower()
data.columns = data.columns.str.replace(' ','_')
```

<a name="DataWarehousing"></a>

## [Data Warehousing](Code/P2_Code.ipynb)
I warehouse all data in a Postgre database for later use and reference.

*   ETL in python to PostgreSQL Database.
*   Formatted table names to SQL compatibility. 

```python 
# Function to warehouse data in a Postgre database 
def store_data(data,tablename):
    """
    :param data: variable, enter name of dataset you'd like to warehouse
    :param tablename: str, enter name of table for data 
    """

    # SQL table header format
    tablename = tablename.lower()
    tablename = tablename.replace(' ','_')

    # Saving cleaned data as csv
    data.to_csv(f'../Data/{tablename}_clean.csv', index=False)

    # Engine to access postgre
    engine = create_engine('postgresql+psycopg2://postgres:password@localhost:5432/projectsdb')

    # Loads dataframe into PostgreSQL and replaces table if it exists
    data.to_sql(f'{tablename}', engine, if_exists='replace',index=False)

    # Confirmation of ETL 
    return("ETL successful, {num} rows loaded into table: {tb}.".format(num=len(data.iloc[:,0]), tb=tablename))
 
# Calling store_data function to warehouse cleaned data
store_data(data,"P2 Bank Churn Prediction")
```

<a name="EDA"></a>  

## [Exploratory data analysis](Code/P2_Code.ipynb) 
I looked at the distributions of the data and the value counts for the various categorical variables that would be fed into the model. Below are a few highlights from the analysis.
*   20.37% of customers have churned - Distribution of features and their effects on churning - Some features have outliers, visualising this allows for greater clarity on the extent. 
* As this is a binary classification use case, I set my colour palette to take red and green. Red for the 'bad' outcome and green for the better outcome. Of course, this can be subjective based on the person setting the colours. 

```python
# Setting my custom color palette
colors = ["#32CD32","#FF0000"]
sns.set_palette(sns.color_palette(colors))
```
<img src="images/Churn_barchart_distrib.png" />
<img src="images/independentfeatures_distrib.png" />

*   The box plots show those features with outliers. Those features are 'creditscore', 'age', 'tenure', 'balance', 'numofproducts', 'estimatedsalary'. 
*   Because of the outliers the features will need to be scaled before modelling. 
<img src="images/boxplots.png" />

*   The features are very correlated generally. 
<img src="images/churn_correlation.png" />

<a name="Dataviz"></a> 

## [Data Visualisation & Analytics](https://app.powerbi.com/view?r=eyJrIjoiNDExYjQ0OTUtNWI5MC00OTQ5LWFlYmUtYjNkMzE1YzE2NmE0IiwidCI6IjYyZWE3MDM0LWI2ZGUtNDllZS1iZTE1LWNhZThlOWFiYzdjNiJ9&pageName=ReportSection)
[View Interactive Dashboard](https://app.powerbi.com/view?r=eyJrIjoiNDExYjQ0OTUtNWI5MC00OTQ5LWFlYmUtYjNkMzE1YzE2NmE0IiwidCI6IjYyZWE3MDM0LWI2ZGUtNDllZS1iZTE1LWNhZThlOWFiYzdjNiJ9&pageName=ReportSection)
*   I created an interactive dashboard to deploy the machine learning model to benefit the business.
*   I visualised various key features and highlighted their overall correlation to a customers churn. 

<!-- Dashboard  -->

<!-- <iframe title="P2Dashboard" width="600" height="373.5" src="https://app.powerbi.com/view?r=eyJrIjoiNDExYjQ0OTUtNWI5MC00OTQ5LWFlYmUtYjNkMzE1YzE2NmE0IiwidCI6IjYyZWE3MDM0LWI2ZGUtNDllZS1iZTE1LWNhZThlOWFiYzdjNiJ9&pageName=ReportSection" frameborder="0" allowFullScreen="true"></iframe> -->

<a name="Busintelli"></a>  

## Business Intelligence
On Page 2 of the interactive dashboard, I have provided the stake holders with the new customer names and the customers that are likely to churn due to their characteristics.

*   These customers can be offered subsidised deals and incentives to keep them on
*   Greater engagement with customers could keep some customers on board 
*   Providing quality customer service can also provide customers with long term value and appreciation for the business
*   The complaints team should pay particular attention to complaints from customers who are predicted to churn.
- 96% of unhappy customers don’t complain and 91% of those will simply leave and never come back?

<a name="FeatEng"></a>  

## [Feature Engineering](Code/P2_Code.ipynb)   
I transformed the categorical variable(s) 'geography' and 'gender' into dummy variables. I also split the data into train and tests sets with a test size of 20%. Shuffling the data is important for randomness as the data can come in clusters of outcome values which would affect the output. Stratifying is used so y_test can reflect y_train. Resulting in a more realistic simulation of how the model is going to perform on new data. 
*   One Hot encoding to encode values
*   Using RobustScaler to scale  

```python
# One Hot encoding for remaining categorical field 
data = pd.get_dummies(data, drop_first = False)
data.head()

# Defining fields that need to be scaled in a list 
scale_vars = ['creditscore', 'age', 'tenure', 'balance', 'numofproducts', 'estimatedsalary' ]

# Robust scaler to address outliers 
transformer = RobustScaler().fit(data[scale_vars])
data[scale_vars] = transformer.transform(data[scale_vars])

# Using train test split to split train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.20, random_state=23, shuffle=True, stratify=y)

# Viewing shape of train / test data
print(X_train.shape)
print(X_test.shape)

```

<a name="ModelBuild"></a> 

## [ML/DL Model Building](Code/P2_Code.ipynb)

I tried five different models and evaluated them using initially using accuracy_score and then MSE/RMSE. I chose MSE and RMSE because it is sensitive to outliers, punishes larger errors and is relatively easy to interpret.   

I tried five different models:
*   **Logistic Regression**
*   **KN Neighbors Classifier** 
*   **Support Vector Classifier** 
*   **Decision Tree Classifier** 
*   **Random Forest Classifier**

*   I put each classifier/model above in a list to iterate through to output accuracy scores.<br>

```python
# Preparing algorithms in a list
random_state = 23
classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(n_neighbors = 5),
    SVC(gamma='auto'),
    DecisionTreeClassifier(criterion = 'gini', max_depth = 3),
    RandomForestClassifier(max_features = 'sqrt', n_jobs = 1, verbose = 1)]
```
*   Cross validation used to get range of accuracy results when randomising the formation of the training datasets 
*    To cater for imbalanced class distributions in classifications use cases StratifiedKFold is used to ensure the test set reflects the distribution of the train set.
This is like "stratify=y" in the train_test_split call

```python
# Cross validate model with Kfold stratified cross validation
kfold = StratifiedKFold(n_splits=5)

# Modeling step Test differents algorithms 
cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["LogisticRegression","KNeighborsClassifier","SVC","DecisionTreeClassifier","RandomForestClassifier"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
```
* Here we can see the models with the greatest accuracies 
<img src="images/Crossvalidation.png" />

<a name="ModelPerf"></a> 

## [Model performance](Code/P2_Code.ipynb)
The Random Forest Classifier model outperformed the other approaches on the test and validation sets. 
*   **Random Forest Classifier** : Accuracy = 86.8% 

<a name="ModelOpt"></a> 

## [Model Optimisation](Code/P2_Code.ipynb)
In this step, I used GridsearchCV to find the best parameters to optimise the performance of the model.
However, in this instance the performance of the model was reduced, so I stuck with the initial parameters. 

*   **Random Forest Classifier** : Accuracy = 86.8% / MSE = 0.1295 / RMSE = 0.36 (2dp)
*   Hyperparameter tuning is effective in finding the best parameters to produce the highest model accuracy. 
*   I gave a large range of values to test by using a great feature from the numpy library. ("np.linspace)

```python
# Using linspace to trial array of values in GridSearchCV parameters 
maxdepth = np.linspace(80, 110, 2,dtype=int)
maxfeatures = np.linspace(2,4, 2,dtype=int)
minsampleaf = np.linspace(2, 5, 2,dtype=int)
minsampsplit = np.linspace(6,12, 2,dtype=int)
nestimators = np.linspace(100,1000, 3,dtype=int)

# Defining parameters for GridSearchCV
params = {
    'bootstrap': [True],
    'max_depth': maxdepth,
    'max_features': maxfeatures,
    'min_samples_leaf': minsampleaf,
    'min_samples_split': minsampsplit,
    'n_estimators': nestimators
}

# Initialising algorithm 
rf = RandomForestClassifier()

# Outputting best param with GridDearch
rf_best = GridSearchCV(rf,params, cv = 3, n_jobs = 1, verbose = 1)

# Fitting best paramters to X and y training
rf_best.fit(X_train,y_train)
```

<a name="ModelEval"></a> 

## [Model Evaluation](Code/P2_Code.ipynb)
*   A confusion matrix showing the accuracy score of 86.8% achieved by the model. 
**Random Forest Classifier**
<img src="images/Confusionmatrixrf.png" />

*   Plotted a ROC curve to show the trade-off between sensitivity (or TPR) and specificity (1 – FPR). Classifiers that give curves closer to the top-left corner indicate a better performance. Here we see how the Random Forest Classifier outperforms the SVC.
<img src="images/ROC_Curve.png" />

*   Using 'feature_importances_' I found the age of a customer has the biggest weight in how likely they are to churn. 

<a name="ModelProd"></a> 

## [Model Productionising](Code/P2_Vis_prep.ipynb)
*   I used the pickle library to export the model. 
<br>

```python
# Dump model into pickle file
pickle.dump(model1, open('.././svc_diabetes.pkl', 'wb'))
```  

*   I then imported the model into a separate notebook to apply it to an 'unseen' dataset to find the predictions and the probability of those predictions. 
*   String fields created for deployment readability aspect. 

<br>

```python
# Predicting based on orginal data 
pred_rf = model.predict(data.values)
pred_prob_rf = model.predict_proba(data.values)

# Adding prediction columns for PowerBI deployment 
data_deploy = data1.copy()
data_deploy['Predictions - Churn or Not'] = pred_rf
data_deploy['Predictions - Probability to Churn'] = column(pred_prob_rf, 1)
data_deploy['Predictions - Churn or Not WORD'] = 'Empty'
data_deploy['Predictions - Churn or Not WORD'][data_deploy['Predictions - Churn or Not'] == 0] = 'Retention'
data_deploy['Predictions - Churn or Not WORD'][data_deploy['Predictions - Churn or Not'] == 1] = 'Churn'
data_deploy.head()
```

<a name="ModelDeploy"></a> 

## [Deployment](https://app.powerbi.com/view?r=eyJrIjoiNDExYjQ0OTUtNWI5MC00OTQ5LWFlYmUtYjNkMzE1YzE2NmE0IiwidCI6IjYyZWE3MDM0LWI2ZGUtNDllZS1iZTE1LWNhZThlOWFiYzdjNiJ9&pageName=ReportSection)
I deployed this model in Microsoft Power BI for business intelligence use. [View visualisation code prep](Code/P2_Vis_prep.ipynb)
*   I exported the model as a .pkl file and applied it to the unseen data set to get churn predictions and probability predictions.
*   I visualised this in Power BI and using conditional formatting to highlight those new customer that are more likely to churn based on the models prediction. 

<a name="Prjmanage"></a> 

## [Project Management (Agile/Scrum/Kanban)](https://www.atlassian.com/software/jira)
* Resources used
    * Jira
    * Confluence
    * Trello 

<a name="PrjEval"></a> 

## [Project Evaluation](Presentation/P11Presentation.pptx) 
*   WWW
    *   The end-to-end process
    *   Deployment and sharing of work 
    *   Communicating business intelligence analysis from data science work
*   EBI 
    *   Better project management and planning would have made this project faster
    

<a name="Lookahead"></a> 

## Looking Ahead
*   What next
*   How can a better accuracy score be achieved? 
*   Explore other data sources 

<a name="Questions"></a> 

## Questions & Contact me 
For questions, feedback, and contribution requests contact me
* ### [Click here to email me](mailto:contactmattithyahu@gmail.com) 
* ### [See more projects here](https://mattithyahudata.github.io/)




