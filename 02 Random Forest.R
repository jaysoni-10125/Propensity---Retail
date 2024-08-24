## KE XU - Modelling Customer Behaviour - Random Forest

#UCI Machine Learning Repository
#Bank Marketing Dataset
#http://archive.ics.uci.edu/ml/datasets/Bank+Marketing
#-----------------------------


install.packages("randomForest")
install.packages("ROCR")
install.packages("gplots")
install.packages("randomForestExplainer")
install.packages("inTrees")
install.packages("e1071")
install.packages("ggRandomForests")

library(randomForest)
library(ROCR)
library(caret)
library(ggplot2)
library(ggRandomForests)

#Randomly select 1000 people out of 4521 without replacement.
#Omit columns of contact methods, campaign information and etc.
bankdata<- bank[sample(nrow(bank),1000,replace = FALSE),c(1:4,6:8,12,17)]
sapply(train, class)

head(bankdata)
str(bankdata)
summary(bankdata)
##y in first attampt
##no:875
##yes:125

bankdata$y = as.factor(bankdata$y)

train_rf<- train
test_rf<- test
train_rf$y<- as.factor(train_rf$y)
test_rf$y<- as.factor(test_rf$y)

#Random forest model
#No need of a testing dataset due to OOB estimates.
modelrf<- randomForest(y ~ ., data = train_rf, importance = TRUE, do.trace = 50, mtry = 4, ntree = 500)
modelrf
print(modelrf)
### tree 500
#-- Call:
#--   randomForest(formula = y ~ ., data = train_rf, importance = TRUE) 
#-- Type of random forest: classification
#-- Number of trees: 500
#-- No. of variables tried at each split: 4

#-- OOB estimate of  error rate: 9.87%
#-- Confusion matrix:
#--       no yes class.error
#-- no  3097 103   0.0321875
#-- yes  254 162   0.6105769

### tree 300
#-- Call:
#--  randomForest(formula = y ~ ., data = train_rf, importance = TRUE,      do.trace = 50, mtry = 4, , ntree = 300) 
#-- Type of random forest: classification
#-- Number of trees: 300
#-- No. of variables tried at each split: 4

#-- OOB estimate of  error rate: 9.82%
#-- Confusion matrix:
#--       no yes class.error
#-- no  3094 106   0.0331250
#-- yes  249 167   0.5985577

plot(modelrf)

plot(modelrf$err.rate[,1])

plot(gg_rfsrc(modelrf))

modelrf2<- randomForest(y ~ ., data = train_rf, importance = TRUE, do.trace = 50, mtry = 17, ntree = 500)
modelrf2


#predicitons
predictions_rf <- predict(modelrf, test_rf, type = "class")
confusion.matrix <- prop.table(table(predictions_rf, test_rf$y))
confusion.matrix
accuracy <- confusion.matrix[1,1] + confusion.matrix[2,2] 

accuracy
#-- 0.8928177
print(  
  confusionMatrix(data=predictions_rf,  
                  reference=test_rf$y, positive='yes'))
## ntree = 500
#--             Reference
#-- Prediction  no yes
#--        no  773  66
#--        yes  27  39



summary(test_rf$y)
#--  no yes 
#-- 800 105 

mtry <- tuneRF(train_rf[,1:16], train_rf$y, ntreeTry=500,
               stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
print(best.m)


#Evaluate the importance of each variable.
#MeanDecreaseAccuracy is how much the model accuracy drops if
#leaving out this variable. The higher the value, the more important it is.
importance(modelrf)
varImpPlot(modelrf)


#----------------------------------
#ROC plot. The closer the curve the the up-left corner, the more accurate the model is.
pred2 = predict(modelrf,type = "prob")
pred2
perf = prediction(pred2[,2], bankdata$y)
perf
# 1. Area under curve
auc = performance(perf, "auc")
auc
# 2. True Positive and Negative Rate
pred3 = performance(perf, "tpr","fpr")
# 3. Plot the ROC curve
plot(pred3,main="ROC Curve for Random Forest",col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")



#---------------------------------
#Observation on the dataset gives that most customers chose "no" (4000/4521).
#Data is not very balanced.
#Randomly select 500 people chose "no" and 500 people chose "yes" to retrain the model.

library(dplyr)
bankre <- bank %>% group_by(y) %>% sample_n(500)
bankre<- bankre[sample(nrow(bankre),1000),]

summary(bankre)
##y
##no:500
##yes:500


#Random forest model retrain
modelrf2<- randomForest(y ~ ., data = train, importance = TRUE)
print(modelrf2)

#OBB estimate of error rate:11.62%
#The error rate does not drop significantly.

######################################
###### permutation importance #######
####################################

rf <- randomForest(hyper-parameters..., importance=T)
imp <- importance(rf, type=1, scale = F) # permutation importances





