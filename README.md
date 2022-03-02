# 🏦 Bank Churn Analysis: Project Overview  
* End to end project reasearching the effects customer attributes have on the churn of a bank customer and predicting those customers that may churn.
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
*   [Model Productionisation](#ModelProd)<br>
*   [Deployment](#ModelDeploy)<br>
*   [Project Management (Agile | Scrum)](#Prjmanage)<br>
*   [Project Evaluation](#PrjEval)<br>
*   [Looking Ahead](#Lookahead)<br>
*   [Questions | Contact me ](#Questions)<br>

         
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
*  Rows: 10000 | Columns: 14
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
The data contained no null values and all datatypes lined up with their field description. 
```python
# Viewing the data types of the columns
data.dtypes

# Viewing dataset shape
data.shape

# 1st check for null values and datatype check 
data.info()
```
<br>

*   Some programming lanaguages can be case sensitive like python and C++ for example, so using lower case letters for variable names allows for straightforward use of data in different programming languages.

```python
# (SQL standard) Formatting column headers by removing potential capital letters and spaces in column headers 
data.columns = data.columns.str.lower()
data.columns = data.columns.str.replace(' ','_')
```


<a name="DataWarehousing"></a>

## [Data Warehousing](Code/P2_Code.ipynb)
I warehouse all data in a Postgre database for later use and reference.

*   ETL in python to PostgreSQL Database.
*   Formatted tablename to SQL compatibility. 

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
*   20.37% of customers have churned - Distrubution of features and their effects on churning - Some features have outliers, visualising this allows for greater clarifty on the extent. 
* As this is a binary classification use case, I set my colour pallette to take red and green. Red for the 'bad' outcome and green for the bettwer outcome. Of course this can be subjective based on the person setting the colours. 

```python
# Setting my custom color palette
colors = ["#32CD32","#FF0000"]
sns.set_palette(sns.color_palette(colors))
```
<img src="images/Churn_barchart_distrib.png" />
<img src="images/independentfeatures_distrib.png" />
<img src="images/boxplots.png" />

*   I looked at the correlation the features have
<img src="images/churn_correlation.png" />

<a name="Dataviz"></a> 

## [Data Visualisation & Analytics](https://app.powerbi.com/view?r=eyJrIjoiNDExYjQ0OTUtNWI5MC00OTQ5LWFlYmUtYjNkMzE1YzE2NmE0IiwidCI6IjYyZWE3MDM0LWI2ZGUtNDllZS1iZTE1LWNhZThlOWFiYzdjNiJ9&pageName=ReportSection)
[View Interactive Dashboard](https://app.powerbi.com/view?r=eyJrIjoiNDExYjQ0OTUtNWI5MC00OTQ5LWFlYmUtYjNkMzE1YzE2NmE0IiwidCI6IjYyZWE3MDM0LWI2ZGUtNDllZS1iZTE1LWNhZThlOWFiYzdjNiJ9&pageName=ReportSection)
*   I created an interactive dashboard to deploy the machine learning model to benefit the business.
*   I visualised various key features and hihglighted their overall correlation to a customers churn. 

<a name="Busintelli"></a>  

## Business Intelligence
On Page 2 of the interactive dashboard I have provided the stake holders with the new customer names and the customers that are likely to churn due to their characteristics.

*   These customers can be offered subsidised deals and incentives to keep them on
*   Greater engagement with customers could keep some customers on board 
*   Providing quality customer service can also provide customers with long term value and appreciation for the business
*   The complaints team should pay particular attention to complaints from customers who are predicted to churn.
- 96% of unhappy customers dont complain  and 91% of those will simply leave and never come back?

<a name="FeatEng"></a>  

## [Feature Engineering](Code/P2_Code.ipynb)   
I transformed the categorical variable(s) 'geography' and 'gender' into dummy variables. I also split the data into train and tests sets with a test size of 20%.
*   One Hot encoding to encode values
*   Using RobustScaler to scale  

<a name="ModelBuild"></a> 

## [ML/DL Model Building](Code/P2_Code.ipynb)

I tried five different models and evaluated them using initially using accuracy_score and then MSE/RMSE. I chose MSE and RMSE because it is sensitive to outliers, punishes larger errors and is relatively easy to interpret.   

I tried five different models:
*   **Logistic Regression**
*   **KN Neighbors Classifier** 
*   **Support Vector Classifier** 
*   **Decision Tree Classifier** 
*   **Random Forest Classifier**


<img src="images/Crossvalidation.png" />

<a name="ModelPerf"></a> 

## [Model performance](Code/P2_Code.ipynb)
The Random Forest Classifier model outperformed the other approaches on the test and validation sets. 
*   **Random Forest Classifier** : Accuracy = 86.8% 

<a name="ModelOpt"></a> 

## [Model Optimisation](Code/P2_Code.ipynb)
In this step, I used GridsearchCV to find the best parameters to optimise the performance of the model.
However in this instance the performance of the model was reduced, so I stuck with the intial paramaters. 

*   **Random Forest Classifier** : Accuracy = 86.8% | MSE = 0.1295 | RMSE = 0.36 (2dp)

<a name="ModelEval"></a> 

## [Model Evaluation](Code/P2_Code.ipynb)
*   A confusion matrix showing the accuracy score of 86.8% achieved by the model. 
**Random Forest Classifier**
<img src="images/Confusionmatrixrf.png" />

*   Plotted a ROC curve to show the trade-off between sensitivity (or TPR) and specificity (1 – FPR). Classifiers that give curves closer to the top-left corner indicate a better performance. Here we see how the Random Forest Classifier outperforms the SVC.
<img src="images/ROC_Curve.png" />

*   Using 'feature_importances_' I found the age of a customer has the biggest weight in how likely they are to churn. 

<a name="ModelProd"></a> 

## [Model Productionisation](Code/P7_Code.ipynb)
*   I used the pickle library to export the model. 
```python
# Dump model into pickle file
pickle.dump(model1, open('.././svc_diabetes.pkl', 'wb'))
```  

<a name="ModelDeploy"></a> 

## [Deployment](https://app.powerbi.com/view?r=eyJrIjoiNDExYjQ0OTUtNWI5MC00OTQ5LWFlYmUtYjNkMzE1YzE2NmE0IiwidCI6IjYyZWE3MDM0LWI2ZGUtNDllZS1iZTE1LWNhZThlOWFiYzdjNiJ9&pageName=ReportSection)
I deployed this model in Microsoft Power BI for business intellignece use. [View visualisation code prep](Code/P2_Vis_prep.ipynb)
*   I exported the model as a .pkl file and applied it to the unseen data set to get churn predictions and probability predictions.
*   I visualised this in Power BI and using conditional formatting to highlight those new customer sthat are more likely to curn based on the models prediction. 

<a name="Prjmanage"></a> 

## [Project Management (Agile | Scrum)](https://www.atlassian.com/software/jira)
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

## Questions | Contact me 
For questions, feedback, and contribution requests contact me
* ### [Click here to email me](mailto:contactmattithyahu@gmail.com) 
* ### [See more projects here](https://mattithyahudata.github.io/)








