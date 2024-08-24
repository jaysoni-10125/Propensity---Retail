## KE XU - Modelling Customer Behaviour - UCI Data Mining

#UCI Machine Learning Repository
#Bank Marketing Dataset
#http://archive.ics.uci.edu/ml/datasets/Bank+Marketing
#----------------------------------------------------

install.packages("ggplot2")
install.packages("tidyverse")
install.packages("gridExtra")
install.packages("ggpubr")
install.packages("mlbench")


hist(subset(bank,y=='yes')$age, xlab='Age', main='Figure')


#plotting ages distribution for class "yes" and "no"
#need to change the bg colour to white. Also change the labels to only one side.

library(ggplot2)
library(grid)
library(ggpubr)

require(gridExtra)
plot2 <- ggplot(bank, aes(age, fill = y)) + 
  geom_histogram(alpha = 0.5, position = 'identity') + ggtitle("Frequency Plot of Age")+ theme_bw()
plot1 <- ggplot(bank, aes(age, fill = y)) + 
  geom_histogram(alpha = 0.5, aes(y = ..density..), position = 'identity') + ggtitle("Density Plot of Age")+ theme_bw()

ggarrange(plot1, plot2, nrow=1, common.legend = TRUE)

##alpha is transparent colour

##lr model only depending on age. See how age influence the model
model_age<- glm(train$y ~ age,family=binomial,data=train)
summary(model_age)

model_duration<- glm(train$y ~ duration,family=binomial,data=train)
summary(model_duration)

model
#-- Degrees of Freedom: 3615 Total (i.e. Null);  3573 Residual
#-- Null Deviance:	    2581 
#-- Residual Deviance: 1718 	AIC: 1804

model_reduce<- step(model, trace=0)
step(model,trace=0)
step(model,trace=2)
summary(model_reduce)

anova(model_reduce,model)
#-- Null deviance: 2581.3  on 3615  degrees of freedom
#-- Residual deviance: 1738.7  on 3589  degrees of freedom
#-- AIC: 1792.7

predictions_reduce <- predict(model_reduce, test, "response")

library(pROC)
roc_obj_reduce <- roc(test$y, predictions_reduce)
coords(roc_obj_reduce, "best", "threshold")
#--   threshold specificity sensitivity 
#-- 0.08260592  0.77875000  0.87619048

glm.pred.reduce <- ifelse(predictions_reduce > 0.08260592, "yes", "no")

Error.reduce <- mean(glm.pred.reduce != test$y)
print(paste('Accuracy',1-Error.reduce))
#-- Accuracy 0.790055248618785

library(ROCR)

pred <- prediction(predictions, test$y)
pred.reduce <- prediction(predictions_reduce, test$y)
perf <- performance(pred,"tpr","fpr")
perf_re <- performance(pred.reduce,"tpr","fpr")

plot(perf, colorize=TRUE)
plot(perf_re, add=TRUE,colorize=FALSE)
abline(0,1,xlim=c(0,1),ylim=c(0,1),lty="dotted",col="black")








##############################
##### AFTER DATA MINING ######
##############################
bank2<- bank[sample(nrow(bank),replace = FALSE),c(2,3,7:9,11:17)]

train2<- train[sample(nrow(train),replace = FALSE),c(2,3,7:9,11:17)]
test2<- test[sample(nrow(test),replace = FALSE),c(2,3,7:9,11:17)]

##Logistic regression model
model2 <- glm(train2$y ~.,family=binomial,data=train2)
summary(model2)

require(stargazer)
stargazer(model2,type="text")


anova(model2, test="Chisq")
#varImp(model)

##Predictions using testing set
predictions_2 <- predict(model2, test2, "response")

#------------------------------------
#Trying to find the output specificity and sensitivity and the best threshold
library(pROC)
roc_obj_2 <- roc(test$y, predictions_2)

roc_obj_2
#--Call:
#--  roc.default(response = test$y, predictor = predictions_2)

#--Data: predictions_2 in 800 controls (test$y no) > 105 cases (test$y yes).
#--Area under the curve: 0.521

plot(roc_obj_2,xlim=c(0,1),ylim=c(0,1)) 

coords(roc_obj_2, "best", "threshold")
#-- threshold specificity sensitivity 
#-- 0.0906031   0.8012500   0.8285714 

##Set threshold to 0.2213995
glm.pred.2 <- ifelse(predictions_2 > 0.2213995, "yes", "no")

Error <- mean(glm.pred.2 != test$y)
print(paste('Accuracy',1-Error))
#--"Accuracy 0.80442"


##Confusion matrix for threshold == 0.0906031 on testing set.
TP<- sum(glm.pred=="yes" & test$y=="yes")
FP<- sum(glm.pred=="yes" & test$y=="no")

TN<- sum(glm.pred=="no"& test$y=="no")
FN<- sum(glm.pred=="no" & test$y=="yes")

confMat<-matrix(c(TP,FP,FN,TN),ncol = 2)
colnames(confMat)<-c("pred yes","pred no")
rownames(confMat)<-c("actual yes","actual no")

print(confMat)
#-- pred yes pred no
#-- actual yes       87      18
#-- actual no       159     641


summary(test)
#-- no :800
#-- yes: 105


sensitivity<- TP/(TP+FN)
sensitivity
#-- sensitivity = 0.82867

specificity<- TN/(TN+FP)
specificity
#-- specificity = 0.80125

#----------------------------------------
#----------------------------------------
#----------------------------------------
## ROC Curve

pred <- prediction(predictions, test$y)
pred
perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)
abline(0,1,xlim=c(0,1),ylim=c(0,1),lty="dotted",col="black")


#Set the threshold to 0.5
glm.pred <- ifelse(predictions > 0.5, "yes", "no")

Error <- mean(glm.pred != test$y)
print(paste('Accuracy',1-Error))
##"Accuracy 0.87"




###############################################
########## rf variable importance plot ########
##########          MDA MDG            ########
###############################################
importance(modelrf)
varImpPlot(modelrf)


#random forest model and predictions
modelrf_MDA<- randomForest(y ~ duration+month+poutcome+day+contact, data = train_rf, 
                       importance = TRUE, mtry = 2, ntree = 500)

test.forest.MDA = predict(modelrf_MDA, type = "prob", newdata = test_rf)

forestpred.MDA = prediction(test.forest.MDA[,2], test_rf$y)
forestperf.MDA = performance(forestpred.MDA, "tpr", "fpr")

modelrf_MDG<- randomForest(y ~ duration+month+balance+age+day+job+poutcome, data = train_rf, 
                           importance = TRUE, mtry = 2, ntree = 500)

test.forest.MDG = predict(modelrf_MDG, type = "prob", newdata = test_rf)

forestpred.MDG = prediction(test.forest.MDG[,2], test_rf$y)
forestperf.MDG = performance(forestpred.MDG, "tpr", "fpr")


###Plot two ROC curves together
plot(forestperf.MDA, col="red",lwd=2)
plot(forestperf.MDG, add=TRUE, col="orange", lwd=2)
plot(forestperf, add=TRUE, col="grey 56", lwd=2)
abline(0,1,lty=2)
legend(0.7, 0.3, legend=c("MDA", "MDG", "Full Model"), col=c("red", "orange", "grey 56"), lty=1, lwd=2)

library(pROC)
roc_obj_rf_MDA <- roc(test_rf$y, test.forest.MDA[,2])
roc_obj_rf_MDA

roc_obj_rf_MDG <- roc(test_rf$y,test.forest.MDG[,2])
roc_obj_rf_MDG

test.forest = predict(modelrf, type = "prob", newdata = test_rf)
roc_obj_rf<- roc(test_rf$y,test.forest[,2])
roc_obj_rf



##########################################
##### rf feature selection method 2 #####
########################################
set.seed(7)
# load the library
library(mlbench)
library(caret)
# load the data
# calculate correlation matrix
correlationMatrix <- cor(bank[,1:16])
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)

#-------The above code gives error that correlation can only
#-------be on numeric variables

##########################################
##### rf feature selection method 3 ######
##########################################
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(bank[,1:16], bank[,17], sizes=c(1:8), rfeControl=control)
# summarize the results# list the chosen features

print(results)
predictors(results)
# plot the results
plot(results, type=c("g", "o"), lwd=1)


######################################
### rf feature selection in logreg ###
######################################
##lr models
model_MDA <- glm(train$y ~ duration+month+poutcome+day+contact,family=binomial,data=train)
model_MDG <- glm(train$y ~ duration+month+balance+age+day+job+poutcome,family=binomial,data=train)
model_reduce<- step(model, trace=0)

##predictions
predictions_lr_MDA <- predict(model_MDA, test, "response")
predictions_lr_MDG <- predict(model_MDG, test, "response")
predictions_reduce <- predict(model_reduce, test, "response")

library(pROC)
library(ROCR)

######### MDA accuracy
roc_obj_lr_MDA<- roc(test$y, predictions_lr_MDA)
coords(roc_obj_lr_MDA, "best", "threshold")
#--   threshold specificity sensitivity 
#--  0.08937079  0.80750000  0.82857143 

glm.pred.lr.MDA <- ifelse(predictions_lr_MDA >  0.08937079, "yes", "no")
Error.lr.MDA <- mean(glm.pred.lr.MDA != test$y)
print(paste('Accuracy',1-Error.lr.MDA))
#-- Accuracy 0.809944751381215

######### MDG accuracy
roc_obj_lr_MDG<- roc(test$y, predictions_lr_MDG)
coords(roc_obj_lr_MDG, "best", "threshold")
#--  threshold specificity sensitivity 
#-- 0.05853191  0.68625000  0.91428571 

glm.pred.lr.MDG <- ifelse(predictions_lr_MDG >  0.05853191, "yes", "no")
Error.lr.MDG <- mean(glm.pred.lr.MDG != test$y)
print(paste('Accuracy',1-Error.lr.MDG))
#-- Accuracy 0.712707182320442

######### lr reduced model
roc_obj_reduce <- roc(test$y, predictions_reduce)
coords(roc_obj_reduce, "best", "threshold")
#--   threshold specificity sensitivity 
#-- 0.08260592  0.77875000  0.87619048

glm.pred.reduce <- ifelse(predictions_reduce > 0.08260592, "yes", "no")

Error.reduce <- mean(glm.pred.reduce != test$y)
print(paste('Accuracy',1-Error.reduce))
#-- Accuracy 0.790055248618785

######### plot three ROC curves together
library(ROCR)
pred.lr.MDA <- prediction(predictions_lr_MDA, test$y)
pred.lr.MDG <- prediction(predictions_lr_MDG, test$y)
pred.reduce <- prediction(predictions_reduce, test$y)

perf_MDA <- performance(pred.lr.MDA,"tpr","fpr")
perf_MDG <- performance(pred.lr.MDG,"tpr","fpr")
perf_re <- performance(pred.reduce,"tpr","fpr")

plot(perf_MDA, col="red", lwd=2)
plot(perf_MDG, col="orange", lwd=2, add=TRUE)
plot(perf_re, add=TRUE,col="deep sky blue", lwd=2)
abline(0,1,xlim=c(0,1),ylim=c(0,1),lty="dotted",col="black")
legend(0.7, 0.3, legend=c("MDA", "MDG", "step"), col=c("red", "orange", "deep sky blue"), lty=1, lwd=2)

########### AUC
performance(pred.lr.MDA, measure="auc")
#-- 0.8882024
performance(pred.lr.MDG, measure="auc")
#-- 0.8728571
performance(pred.reduce, measure="auc")
#-- 0.89325
