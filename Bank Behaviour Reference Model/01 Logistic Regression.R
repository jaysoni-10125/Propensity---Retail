## KE XU - Modelling Customer Behaviour - Logistic Regression

#UCI Machine Learning Repository
#Bank Marketing Dataset
#Aim:classify whether a customer will subscribe the term deposit or not.
#http://archive.ics.uci.edu/ml/datasets/Bank+Marketing
#-----------------------------

##change working directory by Session > Set Working Directory > Choose Directory
##or by the following
setwd("~/Desktop/bank")

install.packages("pROC")
install.packages("caret")
install.packages("stargazer")

##Randomly select 1000 people out of 4521 without replacement to have a quick look.
##Omit contact methods, campaign information and etc.
bank<- read.csv2("./bank.csv")
summary(bank)

bankdata<- bank[sample(nrow(bank),1000,replace=FALSE),c(1,2,3)]

summary(bank)
#--no: 40000
#--yes: 521

##Splitting dataset
##80% training set, 20% testing set
##Force the samplesets to have the same yes/no ratio as the full dataset: 521/4000
##First select out all yes's and all no's

yesdata<- subset(bank, y == 'yes')
nodata<- subset(bank, y == 'no')

##Set the trainset size
yestrainsize <- floor(0.8 * nrow(yesdata))
notrainsize <- floor(0.8 * nrow(nodata))

##Set the seed to make partition reproducible, (not sure what this does)
set.seed(123)
yes_train_ind <- sample(seq_len(nrow(yesdata)), size = yestrainsize)
no_train_ind <- sample(seq_len(nrow(nodata)), size =notrainsize)

yestrain <- yesdata[yes_train_ind, ]
notrain <- nodata[no_train_ind,]

yestest <- yesdata[-yes_train_ind, ]
notest <- nodata[-no_train_ind,]

#summary(yestrain)
#summary(notrain)
#summary(notest)
#summary(yestest)

##Combine yes and no dataframe and resample to make data randomly layout
train<- rbind(yestrain,notrain)
train<- train[sample(nrow(train),replace=FALSE),]
#summary(train)
#--no:3200
#--yes:416

test<- rbind(yestest,notest)
test<- test[sample(nrow(test),replace=FALSE),]
#summary(test)
#--no:800
#--yes:105


##Logistic regression model
model <- glm(train$y ~.,family=binomial,data=train)
summary(model)
require(stargazer)
stargazer(model,type="text")


anova(model, test="Chisq")
modelb<- glm(train$y ~ job+marital+housing+loan+contact+month+duration+campaign+pdays+previous+poutcome,family=binomial,data=train)
anova(model,modelb,test="Chi")

##drop 1 variable each time to test on significance
drop1(model,test="F")

modelc<- glm(train$y ~ loan+contact+day+month+duration+campaign+poutcome,family=binomial,data=train)
anova(modelc,modelb,model,test="Chi")
anova(modelb,model,test="Chi")

library(caret)
varImp(model)

##Predictions using testing set
predictions <- predict(modelc, test, "response")
predictions[1:50]
test$y[1:50]


#------------------------------------
#Trying to find the output specificity and sensitivity and the best threshold
library(pROC)
roc_obj <- roc(test$y, predictions)

roc_obj
#-- Call:
#--  roc.default(response = test$y, predictor = predictions)

#-- Data: predictions in 800 controls (test$y no) < 105 cases (test$y yes).
#-- Area under the curve: 0.8895

plot(roc_obj,xlim=c(0,1),ylim=c(0,1)) 

coords(roc_obj, "best", "threshold")[1,1]
#-- threshold specificity sensitivity 
#-- 0.0906031   0.8012500   0.8285714 



#------------------------------------
##Set threshold to 0.09...
glm.pred <- ifelse(predictions > 0.0906031, "yes", "no")

Error <- mean(glm.pred != test$y)

print(paste('Accuracy',1-Error))
#-- Accuracy 0.80442



#------------------------------------
##Confusion matrix for threshold == 0.0906031 on testing set.
TP<- sum(glm.pred=="yes" & test$y=="yes")
FP<- sum(glm.pred=="yes" & test$y=="no")

TN<- sum(glm.pred=="no"& test$y=="no")
FN<- sum(glm.pred=="no" & test$y=="yes")

confMat<-matrix(c(TP,FP,FN,TN),ncol = 2)
colnames(confMat)<-c("pred yes","pred no")
rownames(confMat)<-c("actual yes","actual no")

print(confMat)
#--             pred yes pred no
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
## ROC Curve plotting - use this one

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

