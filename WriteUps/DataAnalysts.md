# **Interview Questions for Data Analysts**

**1) What is the difference between Data Mining and Data Analysis?**

| **Data Mining**                                          | **Data Analysis**                                            |
| -------------------------------------------------------- | ------------------------------------------------------------ |
| Data mining usually does not require any hypothesis.     | Data analysis begins with a question or an assumption.       |
| Data Mining depends on clean and well-documented data.   | Data analysis involves data cleaning.                        |
| Results of data mining are not always easy to interpret. | Data analysts interpret the results and convey the to the stakeholders. |
| Data mining algorithms automatically develop equations.  | Data analysts have to develop their own equations based on the hypothesis. |

**2) Explain the typical data analysis process.**

Data analysis deals with collecting, inspecting, cleansing, transforming and modelling data to glean valuable insights and support better decision making in an organization. The various steps involved in the data analysis process include –

 **Data Exploration –**

Having identified the business problem, a data analyst has to go through the data provided by the client to analyse the root cause of the problem.

**Data Preparation**

This is the most crucial step of the data analysis process wherein any data anomalies (like missing values or detecting outliers) with the data have to be modelled in the right direction.

**Data Modelling**

The modelling step begins once the data has been prepared. Modelling is an iterative process wherein the model is run repeatedly for improvements. Data modelling ensures that the best possible result is found for a given business problem.

**Validation**

In this step, the model provided by the client and the model developed by the data analyst are validated against each other to find out if the developed model will meet the business requirements.

**Implementation of the Model and Tracking**

This is the final step of the data analysis process wherein the model is implemented in production and is tested for accuracy and efficiency.

**3) What is the difference between Data Mining and Data Profiling?**

Data Profiling, also referred to as Data Archeology is the process of assessing the data values in a given dataset for uniqueness, consistency and logic. Data profiling cannot identify any incorrect or inaccurate data but can detect only business rules violations or anomalies. The main purpose of data profiling is to find out if the existing data can be used for various other purposes.

Data Mining refers to the analysis of datasets to find relationships that have not been discovered earlier. It focusses on sequenced discoveries or identifying dependencies, bulk analysis, finding various types of attributes, etc.

**4) How often should you retrain a data model?**

A good data analyst is the one who understands how changing business dynamics will affect the efficiency of a predictive model. You must be a valuable consultant who can use analytical skills and business acumen to find the root cause of business problems.

The best way to answer this question would be to say that you would work with the client to define a time period in advance. However, I would refresh or retrain a model when the company enters a new market, consummate an acquisition or is facing emerging competition. As a data analyst, I would retrain the model as quick as possible to adjust with the changing behaviour of customers or change in market conditions.

**5) What is data cleansing? Mention few best practices that you have followed while data cleansing.**

From a given dataset for analysis, it is extremely important to sort the information required for data analysis. Data cleaning is a crucial step in the analysis process wherein data is inspected to find any anomalies, remove repetitive data, eliminate any incorrect information, etc. Data cleansing does not involve deleting any existing information from the database, it just enhances the quality of data so that it can be used for analysis.

Some of the best practices for data cleansing include –

- Developing a data quality plan to identify where maximum data quality errors occur so that you can assess the root cause and design the plan according to that.
- Follow a standard process of verifying the important data before it is entered into the database.
- Identify any duplicates and validate the accuracy of the data as this will save lot of time during analysis.
- Tracking all the cleaning operations performed on the data is very important so that you repeat or remove any operations as necessary.

**6) How will you handle the QA process when developing a predictive model to forecast customer churn?**

Data analysts require inputs from the business owners and a collaborative environment to operationalize analytics. To create and deploy predictive models in production there should be an effective, efficient and repeatable process. Without taking feedback from the business owner, the model will just be a one-and-done model.

The best way to answer this question would be to say that you would first partition the data into 3 different sets Training, Testing and Validation. You would then show the results of the validation set to the business owner by eliminating biases from the first 2 sets. The input from the business owner or the client will give you an idea on whether you model predicts customer churn with accuracy and provides desired results.

**7) Mention some common problems that data analysts encounter during analysis.**

- Having a poor formatted data file. For instance, having CSV data with un-escaped newlines and commas in columns.
- Having inconsistent and incomplete data can be frustrating.
- Common Misspelling and Duplicate entries are a common data quality problem that most of the data analysts face.
- Having different value representations and misclassified data.

**8) What are the important steps in data validation process?**

Data Validation is performed in 2 different steps-

Data Screening – In this step various algorithms are used to screen the entire data to find any erroneous or questionable values. Such values need to be examined and should be handled.

Data Verification- In this step each suspect value is evaluated on case by case basis and a decision is to be made if the values have to be accepted as valid or if the values have to be rejected as invalid or if they have to be replaced with some redundant values.

**9) How will you create a classification to identify key customer trends in unstructured data?**

A model does not hold any value if it cannot produce actionable results, an experienced data analyst will have a varying strategy based on the type of data being analysed. For example, if a customer complain was retweeted then should that data be included or not. Also, any sensitive data of the customer needs to be protected, so it is also advisable to consult with the stakeholder to ensure that you are following all the compliance regulations of the organization and disclosure laws, if any.

You can answer this question by stating that you would first consult with the stakeholder of the business to understand the objective of classifying this data. Then, you would use an iterative process by pulling new data samples and modifying the model accordingly and evaluating it for accuracy. You can mention that you would follow a basic process of mapping the data, creating an algorithm, mining the data, visualizing it and so on. However, you would accomplish this in multiple segments by considering the feedback from stakeholders to ensure that you develop an enriching model that can produce actionable results.

**10) What is the criteria to say whether a developed data model is good or not?**

- The developed model should have predictable performance.
- A good data model can adapt easily to any changes in business requirements.
- Any major data changes in a good data model should be scalable.
- A good data model is one that can be easily consumed for actionable results.

**11) According to you what are the qualities/skills that a data analyst must posses to be successful at this position.**

Problem Solving and Analytical thinking are the two important skills to be successful as a data analyst. One needs to skilled ar formatting data so that the gleaned information is available in a easy-to-read manner. Not to forget technical proficiency is of significant importance. You can also talk about other skills that the interviewer expects in an ideal candidate for the job position based on the given job description.

**12) You are assigned a new data analytics project. How will you begin with and what are the steps you will follow?**

The purpose of asking this question is that the interviewer wants to understand how you approach a given data problem and what is the though process you follow to ensure that you are organized. You can start answering this question by saying that you will start with finding the objective of the given problem and defining it so that there is solid direction on what need to be done. The next step would be to do data exploration and familiarize myself with the entire dataset which is very important when working with a new dataset.The next step would be to prepare the data for modelling which would including finding outliers, handling missing values and validating the data. Having validated the data, I will start data modelling until I discover any meaningful insights. After this the final step would be to implement the model and track the output results.

This is the generic data analysis process that we have explained in this answer, however, the answer to your  question might slightly change based on the kind of data problem and the tools available at hand.

**13) What do you know about  interquartile range as data analyst?**

A measure of the dispersion of data that is shown in a box plot is referred to as the interquartile range. It is the difference between the upper and the lower quartile.



**1)Differentiate between Data Science , Machine Learning and AI.**

| **Criteria** | **Data Science**                                             | **Machine Learning**                                         | **Artificial Intelligence**                                  |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Defintion    | Data Science is not exactly a subset of machine learning but it uses machine learning to analyse and make future predictions. | A subset of AI that focuses on narrow range of activities.   | A wide term that focuses on applications ranging from Robotics to Text Analysis. |
| Role         | It can take on a busines role.                               | It is a purely technical role.                               | It is a combination of both business and technical aspects.  |
| Scope        | Data Science is a broad term for diverse disciplines and is not merely about developing and training models. | Machine learning fits within the data science spectrum.      | AI is a sub-field of computer science.                       |
| AI           | Loosely integrated                                           | Machine learning is a sub field of AI and is tightly integrated. | A sub- field of computer science consisting of various task like planning, moving around in the world, recognizing objects and sounds, speaking, translating, performing social or business transactions, creative work.. |

**2)         Python or R – Which one would you prefer for text analytics?**

The best possible answer for this would be Python because it has Pandas library that provides easy to use data structures and high performance data analysis tools.

**3)         Which technique is used to predict categorical responses?**

Classification technique is used widely in mining for classifying data sets.

 

**4)         What is logistic regression? Or State an example when you have used logistic regression recently.**

Logistic Regression often referred as logit model is a technique to predict the binary outcome from a linear combination of predictor variables. For example, if you want to predict whether a particular political leader will win the election or not. In this case, the outcome of prediction is binary i.e. 0 or 1 (Win/Lose). The predictor variables here would be the amount of money spent for election campaigning of a particular candidate, the amount of time spent in campaigning, etc.

**5)         What are Recommender Systems?**

A subclass of information filtering systems that are meant to predict the preferences or ratings that a user would give to a product. Recommender systems are widely used in movies, news, research articles, products, social tags, music, etc.

**6)         Why data cleaning plays a vital role in analysis?**

 Cleaning data from multiple sources to transform it into a format that data analysts or data scientists can work with is a cumbersome process because - as the number of data sources increases, the time take to clean the data increases exponentially due to the number of sources and the volume of data generated in these sources. It might take up to 80% of the time for just cleaning data making it a critical part of analysis task.

**7)         Differentiate between univariate, bivariate and multivariate analysis.**

These are descriptive statistical analysis techniques which can be differentiated based on the number of variables involved at a given point of time. For example, the pie charts of sales based on territory involve only one variable and can be referred to as univariate analysis.

If the analysis attempts to understand the difference between 2 variables at time as in a scatterplot, then it is referred to as bivariate analysis. For example, analysing the volume of sale and a spending can be considered as an example of bivariate analysis.

Analysis that deals with the study of more than two variables to understand the effect of variables on the responses is referred to as multivariate analysis.

**8)         What do you understand by the term Normal Distribution?**

Data is usually distributed in different ways with a bias to the left or to the right or it can all be jumbled up. However, there are chances that data is distributed around a central value without any bias to the left or right and reaches normal distribution in the form of a bell shaped curve. The random variables are distributed in the form of an symmetrical bell shaped curve.

![Bell Curve for Normal Distribution](https://s3.amazonaws.com/files.dezyre.com/images/blog/100+Data+Science+Interview+Questions+and+Answers+(General)/Bell+Shaped+Curve+for+Normal+Distribution.jpg)

**9)         What is Linear Regression?**

Linear regression is a statistical technique where the score of a variable Y is predicted from the score of a second variable X. X is referred to as the predictor variable and Y as the criterion variable.

**10)       What is Interpolation and Extrapolation?**

Estimating a value from 2 known values from a list of values is Interpolation. Extrapolation is approximating a value by extending a known set of values or facts.

**11)       What is power analysis?**

An experimental design technique for determining the effect of a given sample size.

**12)      What is K-means? How can you select K for K-means?**

**13)       What is Collaborative filtering?**

The process of filtering used by most of the recommender systems to find patterns or information by collaborating viewpoints, various data sources and multiple agents.

**14)       What is the difference between Cluster and Systematic Sampling?**

Cluster sampling is a technique used when it becomes difficult to study the target population spread across a wide area and simple random sampling cannot be applied. Cluster Sample is a probability sample where each sampling unit is a collection, or cluster of elements. Systematic sampling is a statistical technique where elements are selected from an ordered sampling frame. In systematic sampling, the list is progressed in a circular manner so once you reach the end of the list,it is progressed from the top again. The best example for systematic sampling is equal probability method.

**15)       Are expected value and mean value different?**

They are not different but the terms are used in different contexts. Mean is generally referred when talking about a probability distribution or sample population whereas expected value is generally referred in a random variable context.

**For Sampling Data**

Mean value is the only value that comes from the sampling data.

Expected Value is the mean of all the means i.e. the value that is built from multiple samples. Expected value is the population mean.

**For Distributions**

Mean value and Expected value are same irrespective of the distribution, under the condition that the distribution is in the same population.

**16)       What does P-value signify about the statistical data?**

P-value is used to determine the significance of results after a hypothesis test in statistics. P-value helps the readers to draw conclusions and is always between 0 and 1.

•           P- Value > 0.05 denotes weak evidence against the null hypothesis which means the null hypothesis cannot be rejected.

•           P-value <= 0.05 denotes strong evidence against the null hypothesis which means the null hypothesis can be rejected.

•           P-value=0.05is the marginal value indicating it is possible to go either way.

**17)  Do gradient descent methods always converge to same point?**

No, they do not because in some cases it reaches a local minima or a local optima point. You don’t reach the global optima point. It depends on the data and starting conditions

**18)  What are categorical variables?**

**19)       A test has a true positive rate of 100% and false positive rate of 5%. There is a population with a 1/1000 rate of having the condition the test identifies. Considering a positive test, what is the probability of having that condition?**

Let’s suppose you are being tested for a disease, if you have the illness the test will end up saying you have the illness. However, if you don’t have the illness- 5% of the times the test will end up saying you have the illness and 95% of the times the test will give accurate result that you don’t have the illness. Thus there is a 5% error in case you do not have the illness.

Out of 1000 people, 1 person who has the disease will get true positive result.

Out of the remaining 999 people, 5% will also get true positive result.

Close to 50 people will get a true positive result for the disease.

This means that out of 1000 people, 51 people will be tested positive for the disease even though only one person has the illness. There is only a 2% probability of you having the disease even if your reports say that you have the disease.

**20)       How you can make data normal using Box-Cox transformation?**

**21)       What is the difference between Supervised Learning an Unsupervised Learning?**

If an algorithm learns something from the training data so that the knowledge can be applied to the test data, then it is referred to as Supervised Learning. Classification is an example for Supervised Learning. If the algorithm does not learn anything beforehand because there is no response variable or any training data, then it is referred to as unsupervised learning. Clustering is an example for unsupervised learning.

**22) Explain the use of Combinatorics in data science.**

**23) Why is vectorization considered a powerful method for optimizing numerical code?**

**24) What is the goal of A/B Testing?**

It is a statistical hypothesis testing for randomized experiment with two variables A and B. The goal of A/B Testing is to identify any changes to the web page to maximize or increase the outcome of an interest. An example for this could be identifying the click through rate for a banner ad.

**25)       What is an Eigenvalue and Eigenvector?**

Eigenvectors are used for understanding linear transformations. In data analysis, we usually calculate the eigenvectors for a correlation or covariance matrix. Eigenvectors are the directions along which a particular linear transformation acts by flipping, compressing or stretching. Eigenvalue can be referred to as the strength of the transformation in the direction of eigenvector or the factor by which the compression occurs.

**26)       What is Gradient Descent?**

**27)       How can outlier values be treated?**

Outlier values can be identified by using univariate or any other graphical analysis method. If the number of outlier values is few then they can be assessed individually but for large number of outliers the values can be substituted with either the 99th or the 1st percentile values. All extreme values are not outlier values.The most common ways to treat outlier values –

1) To change the value and bring in within a range

2) To just remove the value.

**28)       How can you assess a good logistic model?**

There are various methods to assess the results of a logistic regression analysis-

•           Using Classification Matrix to look at the true negatives and false positives.

•           Concordance that helps identify the ability of the logistic model to differentiate between the event happening and not happening.

•           Lift helps assess the logistic model by comparing it with random selection.

**29)       What are various steps involved in an analytics project?**

•           Understand the business problem

•           Explore the data and become familiar with it.

•           Prepare the data for modelling by detecting outliers, treating missing values, transforming variables, etc.

•           After data preparation, start running the model, analyse the result and tweak the approach. This is an iterative step till the best possible outcome is achieved.

•           Validate the model using a new data set.

•           Start implementing the model and track the result to analyse the performance of the model over the period of time.

**30)** **How can you iterate over a list and also retrieve element indices at the same time?**

This can be done using the enumerate function which takes every element in a sequence just like in a list and adds its location just before it.

**31)       During analysis, how do you treat missing values?**

The extent of the missing values is identified after identifying the variables with missing values. If any patterns are identified the analyst has to concentrate on them as it could lead to interesting and meaningful business insights. If there are no patterns identified, then the missing values can be substituted with mean or median values (imputation) or they can simply be ignored.There are various factors to be considered when answering this question-

- Understand the problem statement, understand the data and then give the answer.Assigning a default value which can be mean, minimum or maximum value. Getting into the data is important.
- If it is a categorical variable, the default value is assigned. The missing value is assigned a default value.
- If you have a distribution of data coming, for normal distribution give the mean value.
- Should we even treat missing values is another important point to consider? If 80% of the values for a variable are missing then you can answer that you would be dropping the variable instead of treating the missing values.

**32)       Explain about the box cox transformation in regression models.**

For some reason or the other, the response variable for a regression analysis might not satisfy one or more assumptions of an ordinary least squares regression. The residuals could either curve as the prediction increases or  follow skewed distribution. In such scenarios, it is necessary to transform the response variable so that the data  meets the required assumptions. A Box cox transformation is a statistical technique to transform non-mornla dependent variables into a normal shape. If the given data is not normal then most of the statistical techniques assume normality. Applying a box cox transformation means that you can run a broader number of tests.

**33)       Can you use machine learning for time series analysis?**

Yes, it can be used but it depends on the applications.

**34)       Write a function that takes in two sorted lists and outputs a sorted list that is their union.** 

First solution which will come to your mind is to merge two lists and short them afterwards

```python
def return_union(list_a, list_b):

    return sorted(list_a + list_b)
```

**35)       What is the difference between Bayesian Estimate and Maximum Likelihood Estimation (MLE)?**

In bayesian estimate we have some knowledge about the data/problem (prior) .There may be several values of the parameters which explain data and hence we can look for multiple parameters like 5 gammas and 5 lambdas that do this. As a result of Bayesian Estimate, we get multiple models for making multiple predictions i.e. one for each pair of parameters but with the same prior. So, if a new example need to be predicted than computing the weighted sum of these predictions serves the purpose.

Maximum likelihood does not take prior into consideration (ignores the prior) so it is like being a Bayesian  while using some kind of a flat prior.

**36)       What is Regularization and what kind of problems does regularization solve?**

**37)       What is multicollinearity and how you can overcome it?**

**38)        What is the curse of dimensionality?**

**39)        How do you decide whether your linear regression model fits the data?**

**40)       What is the difference between squared error and absolute error?**

**41)       What is Machine Learning?**

The simplest way to answer this question is – we give the data and equation to the machine. Ask the machine to look at the data and identify the coefficient values in an equation.

For example for the linear regression y=mx+c, we give the data for the variable x, y and the machine learns about the values of m and c from the data.

**42) How are confidence intervals constructed and how will you interpret them?**

**43) How will you explain logistic regression to an economist, physican scientist and biologist?**

**44) How can you overcome Overfitting?**

**45) Differentiate between wide and tall data formats?**

**46) Is Naïve Bayes bad? If yes, under what aspects.**

**47) How would you develop a model to identify plagiarism?**

**48) How will you define the number of clusters in a clustering algorithm?**

Though the Clustering Algorithm is not specified, this question will mostly be asked in reference to K-Means clustering where “K” defines the number of clusters. The objective of clustering is to group similar entities in a way that the entities within a group are similar to each other but the groups are different from each other.

For example, the following image shows three different groups.

![K Mean Clustering Machine Learning Algorithm](https://s3.amazonaws.com/files.dezyre.com/images/blog/100+Data+Science+Interview+Questions+and+Answers+(General)/Data+Science+Interview+Questions+K-Means+Clustering.jpg)

Within Sum of squares is generally used to explain the homogeneity within a cluster. If you plot WSS for a range of number of clusters, you will get the plot shown below. The Graph is generally known as Elbow Curve.

![Data Science Interview Questions K Mean Clustering](https://s3.amazonaws.com/files.dezyre.com/images/blog/100+Data+Science+Interview+Questions+and+Answers+(General)/Data+Science+Interview+Questions+K-Means.png)

Red circled point in above graph i.e. Number of Cluster =6 is the point after which you don’t see any decrement in WSS. This point is known as bending point and taken as K in K – Means.

This is the widely used approach but few data scientists also use Hierarchical clustering first to create dendograms and identify the distinct groups from there.

**49) Is it better to have too many false negatives or too many false positives?**

**51)  What do you understand by Fuzzy merging ? Which language will you use to handle it?**

**52) What is the difference between skewed and uniform distribution?**

When the observations in a dataset are spread equally across the range of distribution, then it is referred to as uniform distribution. There are no clear perks in an uniform distribution. Distributions that have more observations on one side of the graph than the other  are referred to as skewed distribution.Distributions with fewer observations on the left ( towards lower values) are said to be skewed left and distributions with fewer observation on the right ( towards higher values) are said to be skewed right.

**53) You created a predictive model of a quantitative outcome variable using multiple regressions. What are the steps you would follow to validate the model?**

Since the question asked, is about post model building exercise, we will assume that you have already tested for null hypothesis, multi collinearity and Standard error of coefficients.

Once you have built the model, you should check for following –

·         Global F-test to see the significance of group of independent variables on dependent variable

·         R^2

·         Adjusted R^2

·         RMSE, MAPE

In addition to above mentioned quantitative metrics you should also check for-

·         Residual plot

·         Assumptions of linear regression 

**54) What do you understand by Hypothesis in the content of Machine Learning?**

**55) What do you understand by Recall and Precision?**

Recall  measures "Of all the actual true samples how many did we classify as true?"

Precision measures "Of all the samples we classified as true how many are actually true?"

We will explain this with a simple example for better understanding -

Imagine that your wife gave you surprises every year on your anniversary in last 12 years. One day all of a sudden your wife asks -"Darling, do you remember all anniversary surprises from me?".

This simple question puts your life into danger.To save your life, you need to Recall all 12 anniversary surprises from your memory. Thus, Recall(R) is the ratio of number of events you can correctly recall to the number of all correct events. If you can recall all the 12 surprises correctly then the recall ratio is 1 (100%) but if you can recall only 10 suprises correctly of the 12 then the recall ratio is 0.83 (83.3%).

However , you might be wrong in some cases. For instance, you answer 15 times, 10 times the surprises you guess are correct and 5 wrong. This implies that your recall ratio is 100% but the precision is 66.67%.

Precision is the ratio of number of events you can correctly recall to a number of all events you recall (combination of wrong and correct recalls).

**56) How will you find the right K for K-means?**

**57) Why L1 regularizations causes parameter sparsity whereas L2 regularization does not?**

Regularizations in statistics or in the field of machine learning is used to include some extra information in order to solve a problem in a better way. L1 & L2 regularizations are generally used to add constraints to optimization problems.

 

![L1 L2 Regularizations](https://s3.amazonaws.com/files.dezyre.com/images/blog/100+Data+Science+Interview+Questions+and+Answers+(General)/L1+L2+Regularizations.png)

In the example shown above H0 is a hypothesis. If you observe, in L1 there is a high likelihood to hit the corners as solutions while in L2, it doesn’t. So in L1 variables are penalized more as compared to L2 which results into sparsity.

In other words, errors are squared in L2, so model sees higher error and tries to minimize that squared error.

**58) How can you deal with different types of seasonality in time series modelling?**

Seasonality in time series occurs when time series shows a repeated pattern over time. E.g., stationary sales decreases during holiday season, air conditioner sales increases during the summers etc. are few examples of seasonality in a time series.

Seasonality makes your time series non-stationary because average value of the variables at different time periods. Differentiating a time series is generally known as the best method of removing seasonality from a time series. Seasonal differencing can be defined as a numerical difference between a particular value and a value with a periodic lag (i.e. 12, if monthly seasonality is present)

**59) In experimental design, is it necessary to do randomization? If yes, why?**

**60) What do you understand by conjugate-prior with respect to Naïve Bayes?**

**61) Can you cite some examples where a false positive is important than a false negative?**

Before we start, let us understand what are false positives and what are false negatives.

False Positives are the cases where you wrongly classified a non-event as an event a.k.a Type I error.

 And, False Negatives are the cases where you wrongly classify events as non-events, a.k.a Type II error.

**62) Can you cite some examples where a false negative important than a false positive?**

Assume there is an airport ‘A’ which has received high security threats and based on certain characteristics they identify whether a particular passenger can be a threat or not. Due to shortage of staff they decided to scan passenger being predicted as risk positives by their predictive model.

What will happen if a true threat customer is being flagged as non-threat by airport model?

 Another example can be judicial system. What if Jury or judge decide to make a criminal go free?

 What if you rejected to marry a very good person based on your predictive model and you happen to meet him/her after few years and realize that you had a false negative?

**64) Can you explain the difference between a Test Set and a Validation Set?**

Validation set can be considered as a part of the training set as it is used for parameter selection and to avoid Overfitting of the model being built. On the other hand, test set is used for testing or evaluating the performance of a trained machine leaning model.

In simple terms ,the differences can be summarized as-

- Training Set is to fit the parameters i.e. weights.
- Test Set is to assess the performance of the model i.e. evaluating the predictive power and generalization.
- Validation set is to tune the parameters.

**65) What makes a dataset gold standard?**

**66) What do you understand by statistical power of sensitivity and how do you calculate it?**

Sensitivity is commonly used to validate the accuracy of a classifier (Logistic, SVM, RF etc.). Sensitivity is nothing but “Predicted TRUE events/ Total events”. True events here are the events which were true and model also predicted them as true.

Calculation of senstivity is pretty straight forward-

 **Senstivity = True Positives /Positives in Actual Dependent Variable**

Where, True positives are Positive events which are correctly classified as Positives.

**67) What is the importance of having a selection bias?**

Selection Bias occurs when there is no appropriate randomization acheived while selecting individuals, groups or data to be analysed.Selection bias implies that the obtained sample does not exactly represent the population that was actually intended to be analyzed.Selection bias consists of Sampling Bias, Data, Attribute and Time Interval.

**68) Give some situations where you will use an SVM over a RandomForest Machine Learning algorithm and vice-versa.**

SVM and Random Forest are both used in classification problems.

a)      If you are sure that your data is outlier free and clean then go for SVM. It is the opposite -   if your data might contain outliers then Random forest would be the best choice

b)      Generally, SVM consumes more computational power than Random Forest, so if you are constrained with memory go for Random Forest [machine learning algorithm](https://www.dezyre.com/article/top-10-machine-learning-algorithms/202).

**c)**  Random Forest gives you a very good idea of variable importance in your data, so if you want to have variable importance then choose Random Forest machine learning algorithm.

d)      Random Forest machine learning algorithms are preferred for multiclass problems.

e)     SVM is preferred in multi-dimensional problem set - like text classification

but as a good data scientist, you should experiment with both of them and test for accuracy or rather you can use ensemble of many Machine Learning techniques.

**69) What do you understand by feature vectors?**

**70) How do data management procedures like missing data handling make selection bias worse?**

Missing value treatment is one of the primary tasks which a data scientist is supposed to do before starting data analysis. There are multiple methods for missing value treatment. If not done properly, it could potentially result into selection bias. Let see few missing value treatment examples and their impact on selection-

**Complete Case Treatment:** Complete case treatment is when you remove entire row in data even if one value is missing. You could achieve a selection bias if your values are not missing at random and they have some pattern. Assume you are conducting a survey and few people didn’t specify their gender. Would you remove all those people? Can’t it tell a different story?

**Available case analysis:** Let say you are trying to calculate correlation matrix for data so you might remove the missing values from variables which are needed for that particular correlation coefficient. In this case your values will not be fully correct as they are coming from population sets.

**Mean Substitution:** In this method missing values are replaced with mean of other available values.This might make your distribution biased e.g., standard deviation, correlation and regression are mostly dependent on the mean value of variables.

Hence, various data management procedures might include selection bias in your data if not chosen correctly.

**71) What are the advantages and disadvantages of using regularization methods like Ridge Regression?**

**72) What do you understand by long and wide data formats?**

**73) What do you understand by outliers and inliers? What would you do if you find them in your dataset?**

**74) Write a program in Python which takes input as the diameter of a coin and weight of the coin and produces output as the money value of the coin.**

**75) What are the basic assumptions to be made for linear regression?**

Normality of error distribution, statistical independence of errors, linearity and additivity.

**76) Can you write the formula to calculat R-square?**

R-Square can be calculated using the below formular -

1 - (Residual Sum of Squares/ Total Sum of Squares)

**77) What is the advantage of performing dimensionality reduction before fitting an SVM?**

Support Vector Machine Learning Algorithm performs better in the reduced space. It is beneficial to perform dimensionality reduction before fitting an SVM if the number of features is large when compared to the number of observations.

**78) How will you assess the statistical significance of an insight whether it is a real insight or just by chance?**

Statistical importance of an insight can be accessed using Hypothesis Testing.

**79) How would you create a taxonomy to identify key customer trends in unstructured data?**

[](http://ctt.ec/sdqZ0)

The best way to approach this question is to mention that it is good to check with the business owner and understand their objectives before categorizing the data. Having done this, it is always good to follow an iterative approach by pulling new data samples and improving the model accordingly by validating it for accuracy by soliciting feedback from the stakeholders of the business. This helps ensure that your model is producing actionable results and improving over the time.

**80) How will you find the correlation between a categorical variable and a continuous variable ?**

You can use the analysis of covariance technqiue to find the correlation between a categorical variable and a continuous variable.

## **Data Science Puzzles-Brain Storming**

**1) How many Piano Tuners are there in Chicago?** 

To solve this kind of a problem, we need to know –

Can you tell if the equation given below is linear or not ?

Emp_sal= 2000+2.5(emp_age)2

Yes it is a linear equation as the coefficients are linear.

What will be the output of the following R programming code ?

var2<- c("I","Love,"DeZyre")

var2

 It will give an error.

How many Pianos are there in Chicago?

How often would a Piano require tuning?

How much time does it take for each tuning?

We need to build these estimates to solve this kind of a problem. Suppose, let’s assume Chicago has close to 10 million people and on an average there are 2 people in a house. For every 20 households there is 1 Piano. Now the question how many pianos are there can be answered. 1 in 20 households has a piano, so approximately 250,000 pianos are there in Chicago.

Now the next question is-“How often would a Piano require tuning? There is no exact answer to this question. It could be once a year or twice a year. You need to approach this question as the interviewer is trying to test your knowledge on whether you take this into consideration or not. Let’s suppose each piano requires tuning once a year so on the whole 250,000 piano tunings are required.

Let’s suppose that a piano tuner works for 50 weeks in a year considering a 5 day week. Thus a piano tuner works for 250 days in a year. Let’s suppose tuning a piano takes 2 hours then in an 8 hour workday the piano tuner would be able to tune only 4 pianos. Considering this rate, a piano tuner can tune 1000 pianos a year.

Thus, 250 piano tuners are required in Chicago considering the above estimates.

**2) There is a race track with five lanes. There are 25 horses of which you want to find out the three fastest horses. What is the minimal number of races needed to identify the 3 fastest horses of those 25?**

Divide the 25 horses into 5 groups where each group contains 5 horses. Race between all the 5 groups (5 races) will determine the winners of each group. A race between all the winners will determine the winner of the winners and must be the fastest horse. A final race between the 2nd and 3rd place from the winners group along with the 1st and 2nd place of thee second place group along with the third place horse will determine the second and third fastest horse from the group of 25.

**3) Estimate the number of french fries sold by McDonald's everyday.**

**4) How many times in a day does a clock’s hand overlap?**

**5) You have two beakers. The first beaker contains 4 litre of water and the second one contains 5 litres of water.How can you our exactly 7 litres of water into a bucket?**

**6) A coin is flipped 1000 times and 560 times heads show up. Do you think the coin is biased?**

**7) Estimate the number of tennis balls that can fit into a plane.**

**8) How many haircuts do you think happen in US every year?**

**9) In a city where residents prefer only boys, every family in the city continues to give birth to children until a boy is born. If a girl is born, they plan for another child. If a boy is born, they stop. Find out the proportion of boys to girls in the city.**

## **Probability Interview Questions for Data Science**

1. There are two companies manufacturing electronic chip. Company A is manufactures defective chips with a probability of 20% and good quality chips with a probability of 80%. Company B manufactures defective chips with a probability of 80% and good chips with a probability of 20%.If you get just one electronic chip, what is the probability that it is a good chip?
2. Suppose that you now get a pack of 2 electronic chips coming from the same company either A or B. When you test the first electronic chip it appears to be good. What is the probability that the second electronic chip you received is also good?
3. A dating site allows users to select 6 out of 25 adjectives to describe their likes and preferences. A match is said to be found between two users on the website if the match on atleast 5 adjectives. If Steve and On a dating site, users can select 5 out of 24 adjectives to describe themselves. A match is declared between two users if they match on at least 4 adjectives. If Brad and Angelina randomly pick adjectives, what is the probability that they will form a match?
4. A coin is tossed 10 times and the results are 2 tails and 8 heads. How will you analyse whether the coin is fair or not? What is the p-value for the same?
5. Continuation to the above question, if each coin is tossed 10 times (100 tosses are made in total). Will you modify your approach to the test the fairness of the coin or continue with the same?
6. An ant is placed on an infinitely long twig. The ant can move one step backward or one step forward with same probability during discrete time steps. Find out the probability with which the ant will return to the starting point.

## **Statistics Interview Questions for Data Science**

1. Explain the central limit theorem.
2. What is the relevance of central limit theorem to a class of freshmen in the social sciences who hardly have any knowledge about statistics?
3. Given a dataset, show me how Euclidean Distance works in three dimensions.
4. How will you prevent overfitting when creating a statistical model ?

## **Data Science Python Interview Questions and Answers**

**1) How can you build a simple logistic regression model in Python?**

**2) How can you train and interpret a linear regression model in SciKit learn?**

**3) Name a few libraries in Python used for Data Analysis and Scientific computations.**

NumPy, SciPy, Pandas, SciKit, Matplotlib, Seaborn

**4) Which library would you prefer for plotting in Python language: Seaborn or Matplotlib?**

Matplotlib is the python library used for plotting but it needs lot of fine-tuning to ensure that the plots look shiny. Seaborn helps data scientists create statistically and aesthetically appealing meaningful plots. The answer to this question varies based on the requirements for plotting data.

**5)  What is the main difference between a Pandas series and a single-column DataFrame in Python?**

**6) Write code to sort a DataFrame in Python in descending order.**

**7) How can you handle duplicate values in a dataset for a variable in Python?**

**8) Which Random Forest parameters can be tuned to enhance the predictive power of the model?**

**9) Which method in pandas.tools.plotting is used to create scatter plot matrix?**

​    `Scatter_matrix`

**10) How can you check if a data set or time series is Random?**

To check whether a dataset is random or not use the lag plot. If the lag plot for the given dataset does not show any structure then it is random.

**11) Can we create a DataFrame with multiple data types in Python? If yes, how can you do it?**

**12) Is it possible to plot histogram in Pandas without calling Matplotlib? If yes, then write the code to plot the histogram?**

**13) What are the possible ways to load an array from a text data file in Python? How can the efficiency of the code to load data file be improved?**

   `numpy.loadtxt ()`

**14) Which is the standard data missing marker used in Pandas?**

`NaN`

**15) Why you should use NumPy arrays instead of nested Python lists?**

**16)  What is the preferred method to check for an empty array in NumPy?**

**17) List down some evaluation metrics for regression problems.**

**18) Which Python library would you prefer to use for Data Munging?**

`Pandas`

**19) Write the code to sort an array in NumPy by the nth column?**

Using `argsort ()` function this can be achieved. If there is an array X and you would like to sort the nth column then code for this will be `x[x [: n-1].argsort ()]`

**20) How are NumPy and SciPy related?**

**21) Which python library is built on top of matplotlib and Pandas to ease data plotting?**

`Seaborn`

**22) Which plot will you use to access the uncertainty of a statistic?**

`Bootstrap`

**23) What are some features of Pandas that you like or dislike?**

**24) Which scientific libraries in SciPy have you worked with in your project?**

**25) What is pylab?**

A package that combines `NumPy`, `SciPy` and `Matplotlib` into a single namespace.

**26) Which python library is used for Machine Learning?**

  `SciKit-Learn`

## **Frequently Asked Open Ended Machine Learning Interview Questions for Data Scientists**

1. Which is your favourite machine learning algorithm and why?
2. In which libraries for Data Science in Python and R, does your strength lie?
3. What kind of data is important for specific business requirements and how, as a data scientist will you go about collecting that data?
4. Tell us about the biggest data set you have processed till date and for what kind of analysis.
5. Which data scientists you admire the most and why?
6. Suppose you are given a data set, what will  you do with it to find out if it suits the business needs of your project or not.
7. What were the business outcomes or decisions for the projects you worked on?
8. What unique skills you think can you add on to our data science team?
9. Which are your favorite data science startups?
10. Why do you want to pursue a career in data science?
11. What have you done to upgrade your skills in analytics?
12. What has been the most useful business insight or development you have found?
13. How will you explain an A/B test to an engineer who does not know statistics?
14. When does parallelism helps your algorithms run faster and when does it make them run slower?
15. How can you ensure that you don’t analyse something that ends up producing meaningless results?
16. How would you explain to the senior management in your organization as to why a particular data set is important?
17. Is more data always better?
18. What are your favourite imputation techniques to handle missing data?
19. What are your favorite data visualization tools?
20. Explain the life cycle of a data science project.