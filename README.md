# MLinAction
This project will realize all the algorithms in the book "Machine Learning in Action".


# Contents

## Part 1 -- Classification models
The classification algorithms within this part conclude:<br>

### 1. kNN  
>Here, kNN is used in two situation :  
>>1) classify the person from a meeting website,  
>>2) recognize the handwriting digits. 

### 2. Decision Tree(DT)
>Here, DT is constructed with the ID3 algorithm, it is used to :  
>>1) predicate the class of a ocean creature (whether it is a kind of fish),  
>>2) predicate the class of lenses.  

### 3. Naive Bayes(NB)
>Here, NB is applied in the following three situation :
>>1) Text classification : classify the class of speech (abusive or formal) on the posting board of dalmation fans,  
>>2) Spam detection : filter the spam from the email list,  
>>3) Area analysis : analysis the common top words in an area from the personal ads.  

### 4. Logistic Regression(LR)
>Here, LR use the (stochastic) gradient ascent methods to optimize the parameters. The trained LR model is used to :
>>1) classify the points in a plane,  
>>2) predicate the death ratio of ill horses from colic features.   


### 5. Support Vector Machine(SVM)
>Here, SVM uses the SMO(Sequential Minimization Optimization) algorithm to train the model. Kernel tricks are used to classify the data which is inseparable in low-dimension space. The trained svm model is used to : 
>>1) classify the points in a plane,  
>>2) classify the handwriting digits.


### 6. AdaBoost
>Here, AdaBoost utilizes the multiple weaker classifiers (decision stumps) to improve the classification performance of meta-algorithm. The trained AdaBoost model is used to :
>>1) classify the points in a plane,  
>>2) detect whether a horse has colic,  
>>3) plot the ROC curve for horse colic detection system.


## Part 2 -- Regression models
The regression algorithms within this part conclude:<br>

### 7. Linear Regression
>Here, we use the Standard Linear Regression (SLR) and Locally Weighted Linear Regression (LWLR) model to predicate numeric data. To understand the data better, some shrinkage methods are used, such as Ridge Regression (RR), Lasso Regression (LR), Forward Stepwise Linear Regression (FSLR) and so on. Besides, we use 10-fold cross validation to select the best trained model. The learned regression models are used to :
>> 1) computing a fitting line from points in a plane, 
>> 2) predicate the age of abalone, 
>> 3) predicate the price of Lego toy suits.


... To be continued ...


# License
```
  Copyright (C) 2017 Huang Chao @ KSE, Southeast University
```
