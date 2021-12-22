# 🏦 Bank Churn Analysis: Project Overview 
* End to end project reasearching the effects certain attributes have on the churn of a bank customer and predicting those customers that may churn.
* Optimized Linear, Lasso, and Random Forest Regressors using GridsearchCV to reach the best model. 
* Built a client facing API using flask 

[View Code](Code/P2 Code.ipynb)

## Resources Used
**Python 3.8, SSIS, SQL Server, Power BI, PowerPoint, AWS** 

[**Anaconda Packages:**](requirements.txt) **pandas, numpy, sklearn, matplotlib, seaborn, selenium, flask, json, pickle, lxml**   


[Kaggle Data source link](https://www.kaggle.com/kmalit/bank-customer-churn-prediction) 

## Data Collection
Source: Kaggle | Webscraping AVG Rupees/GBP conversion data
*	Year	
*   Selling_Price	
*   Present_Price	
*   Kms_Driven	
*   Fuel_Type	
*   Seller_Type	
*   Transmission	
*   Owner
-------
*   Conversion

## Data Warehousing
AAAAAAAAAAAAAAAAAAAAAAAAA

*	Made a new column for company state 
*	Added a column for if the job was at the company’s headquarters 
*	Transformed founded date into age of company 



## Data Cleaning
After I had sraped and downloaded all the data I needed, I needed to clean it up so that it was usable for the model and analysis. I made the following changes and created the following variables:

*	Parsed numeric data out of salary 
*	Made columns for employer provided salary and hourly wages 
*	Removed rows without salary 
*	Parsed rating out of company text 
*	Made a new column for company state 
*	Added a column for if the job was at the company’s headquarters 
*	Transformed founded date into age of company 


## Exploratory data analysis 
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights from the pivot tables. 

![alt text](https://github.com/PlayingNumbers/ds_salary_proj/blob/master/salary_by_job_title.PNG "Salary by Position")
![alt text](https://github.com/PlayingNumbers/ds_salary_proj/blob/master/positions_by_state.png "Job Opportunities by State")
![alt text](https://github.com/PlayingNumbers/ds_salary_proj/blob/master/correlation_visual.png "Correlations")

For any business in a designated period of time, customers can fall into 3 main categories: Newly Acquired Customers, existing Customers, and churned customers.
ecause they can translate in a direct loss of revenue, predicting possible customers who can churn beforehand can help the company save this loss.

It cost more to acquire new customers than it is to retain existing ones.
Because it costs so much to acquire them, it is wise to work hard towards retaining them.

A company avoids customer churn by knowing its customers. One of the best ways to achieve this is through the analysis of historical and new customer data.
One of the metrics to keep track of customer churn is Retention Rate, an indication of to what degree the products satisfy a strong market demand, known as product-market fit.
If a product-market fit is not satisfactory, a company will likely experience customers churning.
A powerful tool to analyze and improve Retention Rate is Churn Prediction; a technique that helps to find out which customer is more likely to churn in the given period of time.


## Data Visualisation
AAAAAAAAAAAAAAAAAAAAAAAAA

*	Made a new column for company state 
*	Added a column for if the job was at the company’s headquarters 
*	Transformed founded date into age of company 

## Data Analytics
AAAAAAAAAAAAAAAAAAAAAAAAA

*	Made a new column for company state 
*	Added a column for if the job was at the company’s headquarters 
*	Transformed founded date into age of company 

## Business Intelligence
AAAAAAAAAAAAAAAAAAAAAAAAA

*	Made a new column for company state 
*	Added a column for if the job was at the company’s headquarters 
*	Transformed founded date into age of company 

## ML/DL Model Building 

First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 20%.   

I tried three different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret and outliers aren’t particularly bad in for this type of model.   

I tried three different models:
*	**Multiple Linear Regression** – Baseline for the model
*	**Lasso Regression** – Because of the sparse data from the many categorical variables, I thought a normalized regression like lasso would be effective.
*	**Random Forest** – Again, with the sparsity associated with the data, I thought that this would be a good fit. 

## Model performance
The Random Forest model far outperformed the other approaches on the test and validation sets. 
*	**Random Forest** : MAE = 11.22
*	**Linear Regression**: MAE = 18.86
*	**Ridge Regression**: MAE = 19.67

## Deployment 
In this step, I built a flask API endpoint that was hosted on a local webserver by following along with the TDS tutorial in the reference section above. The API endpoint takes in a request with a list of values from a job listing and returns an estimated salary. 

## Evaluation 
In this step, I built a flask API endpoint that was hosted on a local webserver by following along with the TDS tutorial in the reference section above. The API endpoint takes in a request with a list of values from a job listing and returns an estimated salary. 


## Project Management (Agile | Scrum)
* Resources used
    * Jira
    * Confluence
    * Trello 

## Questions and See more projects    

* #### [See more projects here](https://mattithyahutech.co.uk/)
* #### [Contact me here](mailto:theanalyticsolutions@gmail.com) 





