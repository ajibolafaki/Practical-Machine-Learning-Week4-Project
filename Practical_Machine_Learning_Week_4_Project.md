Introduction
------------

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

Load & Cleanse Data
-------------------

``` r
library(caret)
```

    ## Warning: package 'caret' was built under R version 3.4.4

``` r
library(tidyverse)
library(randomForest)
```

    ## Warning: package 'randomForest' was built under R version 3.4.4

``` r
library(rpart)
library(rpart.plot)
```

    ## Warning: package 'rpart.plot' was built under R version 3.4.4

``` r
library(RColorBrewer)
library(RGtk2)
```

    ## Warning: package 'RGtk2' was built under R version 3.4.4

``` r
library(rattle)
```

    ## Warning: package 'rattle' was built under R version 3.4.4

``` r
trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training_df <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
testing_df  <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))

# Exclude first seven variables because they are not predictors
#str(training_df)
training_df <- training_df[, -c(1:7)]

# Exclude columns with NAs from training dataset
training_df <- training_df[, colSums(is.na(training_df)) == 0]

# Check for near zero variance in training dataset
training_nzv <- nzv(training_df[,-ncol(training_df)],saveMetrics=TRUE)
#rownames(training_nzv)


# partition training dataset further into train and test data sets
inTrain  <- createDataPartition(training_df$classe, p=0.75, list=FALSE)
train_df <- training_df[inTrain, ]
test_df  <- training_df[-inTrain, ]

dim(train_df)
dim(test_df)
```

Modeling : Decision Trees
-------------------------

``` r
set.seed(21251)
fit_dt <- rpart(classe ~ ., data = train_df, method="class")
fancyRpartPlot(fit_dt)
```

    ## Warning: labs do not fit even at cex 0.15, there may be some overplotting

![](Practical_Machine_Learning_Week_4_Project_files/figure-markdown_github/unnamed-chunk-2-1.png)

``` r
predict_dt <- predict(fit_dt, test_df, type = "class")
confusionMatrix(predict_dt, test_df$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1130  145   21   44   27
    ##          B  114  606   76   82  119
    ##          C   38  124  746  165  113
    ##          D  100   33   12  495   35
    ##          E   13   41    0   18  607
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7308          
    ##                  95% CI : (0.7182, 0.7432)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6598          
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8100   0.6386   0.8725   0.6157   0.6737
    ## Specificity            0.9325   0.9011   0.8913   0.9561   0.9820
    ## Pos Pred Value         0.8266   0.6078   0.6290   0.7333   0.8940
    ## Neg Pred Value         0.9251   0.9122   0.9707   0.9269   0.9304
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2304   0.1236   0.1521   0.1009   0.1238
    ## Detection Prevalence   0.2788   0.2033   0.2418   0.1376   0.1385
    ## Balanced Accuracy      0.8712   0.7699   0.8819   0.7859   0.8279

Modeling : Random Forest
------------------------

In this section, we build a random forest model on the test set. We expect a small out of sample error estimate

``` r
set.seed(21212)
fit_rf <- randomForest(classe ~ ., data = train_df, ntree = 1000)
predict_rf <- predict(fit_rf, test_df, type = "class")
confusionMatrix(predict_rf, test_df$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1395    1    0    0    0
    ##          B    0  945    6    0    0
    ##          C    0    3  849    6    0
    ##          D    0    0    0  797    0
    ##          E    0    0    0    1  901
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.9965         
    ##                  95% CI : (0.9945, 0.998)
    ##     No Information Rate : 0.2845         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.9956         
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9958   0.9930   0.9913   1.0000
    ## Specificity            0.9997   0.9985   0.9978   1.0000   0.9998
    ## Pos Pred Value         0.9993   0.9937   0.9895   1.0000   0.9989
    ## Neg Pred Value         1.0000   0.9990   0.9985   0.9983   1.0000
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2845   0.1927   0.1731   0.1625   0.1837
    ## Detection Prevalence   0.2847   0.1939   0.1750   0.1625   0.1839
    ## Balanced Accuracy      0.9999   0.9971   0.9954   0.9956   0.9999

Modeling : GBM
--------------

In this section, we build a GBM model on the test set.

``` r
set.seed(21237)
fit_gbm <- train(classe ~ ., data = train_df, 
                             trControl = trainControl(method = "repeatedcv", number = 5, repeats = 1),
                             method = "gbm")
```

``` r
predict_gbm <- predict(fit_gbm, test_df)
confusionMatrix(predict_gbm, test_df$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1380   38    0    0    1
    ##          B   10  878   26    1   10
    ##          C    4   31  816   27    5
    ##          D    1    2   12  774   17
    ##          E    0    0    1    2  868
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9617          
    ##                  95% CI : (0.9559, 0.9669)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9515          
    ##  Mcnemar's Test P-Value : 6.32e-08        
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9892   0.9252   0.9544   0.9627   0.9634
    ## Specificity            0.9889   0.9881   0.9835   0.9922   0.9993
    ## Pos Pred Value         0.9725   0.9492   0.9241   0.9603   0.9966
    ## Neg Pred Value         0.9957   0.9822   0.9903   0.9927   0.9918
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2814   0.1790   0.1664   0.1578   0.1770
    ## Detection Prevalence   0.2894   0.1886   0.1801   0.1644   0.1776
    ## Balanced Accuracy      0.9891   0.9567   0.9689   0.9774   0.9813

Since the random forest model performed better than GBM and Decision Trees, we run predicions on the test set using the random forest model.

``` r
predict_rf_final <- predict(fit_rf, testing_df, type = "class")
predict_rf_final
```

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E
