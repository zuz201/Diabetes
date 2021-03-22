# Diabetes
What is the probability that I can suffer from diabetes? The goal of the project is to create application which help women decide if they are in risk group.The Machine Learning model with the best accuracy in predicting diabetes disease was chosen and used.

## Table of contents
1. Introduction
2. Data Preprocessing 

## 1. Introduction

**International Diabetes Federation** refers that **463 million** people living with diabetes. There are couple of questions coming up to my mind. First of all how serious is this disease? What symptoms should pay my attention? Finally, the most important question -  what is the probability that I can suffer from diabetes? 

I decided to do some research on this and present key figures which I found on International Diabetes Federation site:

![Diabets facts & figures](https://github.com/zuz201/Diabetes/blob/master/d1.PNG) 




To answer to those questions we should explain what diabetes is. The situation when the **pancreas** is no longer able to make **insulin** leads to **chronic disease which is called diabetes**. 

There is another question what is the insulin and why it is so important for us? We know that to do some stuff we need energy, which is produce by our cells. How does it work? The cells need our help, they need some ‘fuel’. Ok, so let’s do this, we can eat some glucose-rich food. What is next? How our cells absorb it from food which we ate? This is the role for insulin, a hormone made by the pancreas, letting glucose from the food we eat pass from the bloodstream into the cells in the body. If we are not able to produce insulin, the glucose levels in the blood raised leading to hyperglycaemia. In the long term high glucose levels can lead to irreparable damage to our body. 


There are three types of diabetes:
- **Type 1** - body produces very little or no insulin, occurs most frequently in children and adolescents
- **Type 2** - body does not make good use of the insulin that it produces, more common in adults
- **Gestational diabetes (GDM)** - consists of high blood glucose during pregnancy and is associated with complications to both mother and child.


To answer the last question and calculate  the probability that I can suffer from diabetes, I decided to use Machine Learning to help me predict Diabetes. It is the Machine Learning classification problem with two clasess: diabetes and non-diabetes. To summarize the performance of a classification algorithm I will be using accuracy, confusion matrix, recall, precision, F-measure.

## 2. Data Preprocessing

### Data Description

The Diabetes dataset was gathered from National Institiute of Diabetes and Digestive and Kidney Diseases. All patients are females at least 21 years. This dataset consists of several medical variables: 
* **Pregnancies** - number of times pregnant <br>
* **Glucose** - plasma glucose concentration a 2 hours in an oral glucose tolerance test <br> 
* **Blood Pressure** - diastolic blood pressure (mm Hg) <br>
* **SkinThickness** - triceps skin fold thickness (mm) <br>
* **Insulin** - 2-Hour serum insulin (mu U/ml) <br>
* **BMI** - body mass index (weight in kg/(height in m)^2) <br>
* **DiabetesPedigreeFunction** - diabetes pedigree function <br>
* **Age** - age (years) <br>
<br>and one target variable named 'Outcome'. 


### Data Exploration

The most important step in Machine Learning is to explore dataset in order to get to know features and to see if data cleaning or data transformation is needed.
In this project Python and data science related packages such as _pandas_, _numpy_, _seaborn_, _matplotlib_ were used.

As was noticed using _info()_ function there are 768 entries without missing values, but to see more detail information _describe()_ function was useful.

Despite the fact that there are no missing values, the above statistics indicate potential errors. As was presented above majority of data have minimum value equal to 0.0 . In some cases this situation might be realistic for example number of pregnancies, but features like _Blood Pressure_ or _Skin Thickness_ equal cannot be equal to 0.0 .

### Data Preparation
