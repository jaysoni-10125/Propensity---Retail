## KE XU - Modelling Customer Behaviour - Example Plots

#UCI Machine Learning Repository
#Bank Marketing Dataset
#http://archive.ics.uci.edu/ml/datasets/Bank+Marketing
#----------------------------------------------------

#Example Plots - List:
#1. Linear regression for classification 
#2. Logistic regression on for classification
#3. LR and RF model ROC curves and error rate box plots
#4. Frequency and density plots of age
#5. Sensitivity and specificity
#6. Random forest variable importance plots
#7. RF error rates
#8. Imputation plots (Amelia and MICE)


## Declaration: Due to the break down of my laptop, 
## I could only find back-up of codes from Febuary.
## Hence codes of a few plots I generated later to improvethe appearance are lost.
## The original approaches are listed here to re-create plots.

library(dplyr)
library(ggplot2)
library(grid)
library(ggpubr)
library(Amelia)
library(randomForest)
library(mice)
require(gridExtra)

#----------------------------------------
#----------------------------------------
#----------------------------------------
##Sample 200 yes and 200 no
##Use the same data to obtain log reg model

linregdata2 <- bank %>% group_by(y) %>% sample_n(200)
linregdata2<- linregdata2[sample(nrow(linregdata2),400),c(12,17)]


##Change yes to 1 and no to 0 for plotting
lrdata2<- linregdata2
lrdata2$y <- as.character(lrdata2$y)
lrdata2$y[lrdata2$y == "yes"] <- "1"
lrdata2$y[lrdata2$y == "no"] <- "0"

##Plot data
lrdata2
plot(lrdata2,ylim = c(0:1),col='dark grey', xlab="Duration", ylab="Probability P(y|d)")

##Linear regression model and best fit line
fit<- lm(lrdata2$y ~ lrdata2$duration)

abline(fit$coef[1],fit$coef[2],col="red", lwd=2)


##Plot log reg curve with the same dataset
logregdata2<- linregdata2

##Log reg model
m2<- glm(logregdata2$y~., family = binomial,data = logregdata2)
m2
summary(m2)
range(logregdata2$duration)

##Predicitons
xplot<- seq(0,1800,0.1)
yplot<- predict(m2, list(duration = xplot),type="response")


X1_range <- seq(from=min(logregdata2$duration), to=max(logregdata2$duration), by=.01)

a_logits <- (-1.3055887)+ 0.0037738*X1_range

a_probs <- exp(a_logits)/(1 + exp(a_logits))


##Plot the curve
plot(X1_range, a_probs, 
     ylim=c(0,1),
     type="l", 
     lwd=3, 
     col="red", xlab="Duration", ylab="Probability P(y|d)")

##Plot data points
points(lrdata2,ylim = c(0:1),col='dark grey')


#------------------------------------------
####################################################
# This session plots 
# ROC curves of LR and RF models together
####################################################

##ROC curve plot for lr and rf models

library(ROCR)
library(randomForest)

#log reg model and predictions
model <- glm(train$y ~.,family=binomial,data=train)
predictions<- predict(model,test,"response")

pred <- prediction(predictions, test$y)

perf <- performance(pred,"tpr","fpr")

#random forest model and predictions
modelrf<- randomForest(y ~ ., data = train_rf, 
                       importance = TRUE, mtry = 4, ntree = 300)

test.forest = predict(modelrf, type = "prob", newdata = test_rf)

forestpred = prediction(test.forest[,2], test_rf$y)
forestperf = performance(forestpred, "tpr", "fpr")


###Plot two ROC curves together
plot(forestperf, col="red", lwd=2)
plot(perf, add=TRUE,col="dark blue", lwd=2)
abline(0,1,lty=2)
legend(0.5, 0.3, legend=c("random forest", "logistic regression"), col=c("red", "dark blue"), lty=1, lwd=2)

library(pROC)
roc_obj <- roc(test$y, predictions)
roc_obj

roc_obj_rf <- roc(test_rf$y,test.forest[,2])
roc_obj_rf

#------------------------------------------
####################################################
# Re-sample 100 training sets
# For each training set, apply both models on it
# Record error rates to give box plots
####################################################

#Set empty list for recording errors
error_lr_rec <- rep(0,100)
error_rf_rec <- rep(0,100)

#Force the training set have the same proportion of yes and no data
yestrainsize_loop <- floor(0.8 * nrow(yesdata))
notrainsize_loop <- floor(0.8 * nrow(nodata))


#Repeat sampling and modelling
for (i in 1:10){
  
  yes_train_ind_loop <- sample(seq_len(nrow(yesdata)), size = yestrainsize_loop)
  no_train_ind_loop <- sample(seq_len(nrow(nodata)), size =notrainsize_loop)
  
  yestrain_loop <- yesdata[yes_train_ind_loop, ]
  notrain_loop <- nodata[no_train_ind_loop,]
  
  yestest_loop <- yesdata[-yes_train_ind_loop, ]
  notest_loop <- nodata[-no_train_ind_loop,]
  
  train_loop<- rbind(yestrain_loop,notrain_loop)
  train_loop<- train_loop[sample(nrow(train_loop),replace=FALSE),]
  
  test_loop<- rbind(yestest_loop,notest_loop)
  test_loop<- test_loop[sample(nrow(test_loop),replace=FALSE),]
  
  #print(summary(train_loop$y))
  
  model_loop <- glm(train_loop$y ~.,family=binomial,data=train_loop)
  predictions_loop <- predict(model_loop, test_loop, "response")
  roc_obj_loop <- roc(test_loop$y, predictions_loop)
  
  c0_loop <- coords(roc_obj_loop, "best", "threshold")[1,1]
  
  glm.pred_loop <- ifelse(predictions_loop > c0_loop, "yes", "no")
  
  error_lr_loop <- mean(glm.pred_loop != test_loop$y)
  error_lr_rec[i] <- error_lr_loop
  
  
  #random forest model
  train_rf_loop<- train_loop
  test_rf_loop<- test_loop
  train_rf_loop$y<- as.factor(train_rf_loop$y)
  test_rf_loop$y<- as.factor(test_rf_loop$y)
  
  
  modelrf_loop<- randomForest(y ~ ., data = train_rf_loop, 
                              importance = TRUE, mtry = 4, ntree = 500)
  
  predictions_rf_loop <- predict(modelrf_loop, test_rf_loop, type = "class")
  conf.mat_loop <- prop.table(table(predictions_rf_loop, test_rf$y))
  error_rf_loop <- 1 - conf.mat_loop[1,1] - conf.mat_loop[2,2] 
  error_rf_rec[i] <- error_rf_loop
  
  
}

#--------------------------end of repeating---------------

error_lr_rec
error_rf_rec

conf.mat_loop
#Boxplot for both error rates
boxplot(error_rf_rec, error_lr_rec, col=c("sky blue","gold"),ylab="Error rates")

legend("topleft", inset=.02, c("random forest","logistic regression"), col=c("sky blue", "gold"), pch=15)

error_rec <- data.frame(error_lr_rec,error_rf_rec)

error_rec

#-------------loop again for 300 runs

#Set empty list for recording errors
error_lr_rec2 <- rep(0,300)
error_rf_rec2 <- rep(0,300)

for (i in 1:300){
  
  yes_train_ind_loop <- sample(seq_len(nrow(yesdata)), size = yestrainsize_loop)
  no_train_ind_loop <- sample(seq_len(nrow(nodata)), size =notrainsize_loop)
  
  yestrain_loop <- yesdata[yes_train_ind_loop, ]
  notrain_loop <- nodata[no_train_ind_loop,]
  
  yestest_loop <- yesdata[-yes_train_ind_loop, ]
  notest_loop <- nodata[-no_train_ind_loop,]
  
  train_loop<- rbind(yestrain_loop,notrain_loop)
  train_loop<- train_loop[sample(nrow(train_loop),replace=FALSE),]
  
  test_loop<- rbind(yestest_loop,notest_loop)
  test_loop<- test_loop[sample(nrow(test_loop),replace=FALSE),]
  
  #print(summary(train_loop$y))
  
  model_loop <- glm(train_loop$y ~.,family=binomial,data=train_loop)
  predictions_loop <- predict(model_loop, test_loop, "response")
  roc_obj_loop <- roc(test_loop$y, predictions_loop)
  
  c0_loop <- coords(roc_obj, "best", "threshold")[1,1]
  
  glm.pred_loop <- ifelse(predictions_loop > c0_loop, "yes", "no")
  
  error_lr_loop2 <- mean(glm.pred_loop != test_loop$y)
  error_lr_rec2[i] <- error_lr_loop2
  
  
  #random forest model
  train_rf_loop<- train_loop
  test_rf_loop<- test_loop
  train_rf_loop$y<- as.factor(train_rf_loop$y)
  test_rf_loop$y<- as.factor(test_rf_loop$y)
  
  
  modelrf_loop<- randomForest(y ~ ., data = train_rf_loop, 
                              importance = TRUE, mtry = 4, ntree = 300)
  
  predictions_rf_loop <- predict(modelrf_loop, test_rf_loop, type = "class")
  conf.mat_loop <- prop.table(table(predictions_rf_loop, test_rf$y))
  error_rf_loop2 <- 1 - conf.mat_loop[1,1] - conf.mat_loop[2,2] 
  error_rf_rec2[i] <- error_rf_loop2
  
  
}

error_rf_rec2
boxplot(error_rf_rec2)


#---------------------------------
########################################
##Frequency and density plots of age

require(gridExtra)
plot2 <- ggplot(bank, aes(age, fill = y)) + 
  geom_histogram(alpha = 0.5, position = 'identity') + ggtitle("Frequency Plot of Age")+ theme_bw()
plot1 <- ggplot(bank, aes(age, fill = y)) + 
  geom_histogram(alpha = 0.5, aes(y = ..density..), position = 'identity') + ggtitle("Density Plot of Age")+ theme_bw()

ggarrange(plot1, plot2, nrow=1, common.legend = TRUE)

##alpha is transparent colour


#----------------------------
##Plot sensitivity and specificity
performance(perf, "sens")
pred1 = predict(model)
length(pred1)
perflr = prediction(pred1, test$y)

plot(unlist(performance(perf, "sens")@x.values), unlist(performance(perf, "sens")@y.values), 
     type="l", lwd=2, ylab="Sensitivity", xlab="Cutoff")
par(mar=c(1,1,1,1))
par(new=TRUE)
plot(unlist(performance(perf, "spec")@x.values), unlist(performance(perf, "spec")@y.values), 
     type="l", lwd=2, col='red', ylab="", xlab="", add=TRUE)
axis(4, at=seq(0,1,0.2))
mtext("Specificity",side=4, padj=-2, col='red')


##Another aproach of spec and sens plot
library(ggplot2)
library(dplyr)
predictions<- predict(model,test,"response")
pred1<- predict(model)
pred1
perf<- prediction(pred1, test$y)
perf

sens <- data.frame(x=unlist(performance(perf, "sens")@x.values), 
                   y=unlist(performance(perf, "sens")@y.values))
spec <- data.frame(x=unlist(performance(perf, "spec")@x.values), 
                   y=unlist(performance(perf, "spec")@y.values))

sens %>% ggplot(aes(x,y)) + 
  geom_line() + 
  geom_line(data=spec, aes(x,y,col="red")) +
  scale_y_continuous(sec.axis = sec_axis(~., name = "Specificity")) +
  labs(x='Cutoff', y="Sensitivity") +
  theme(axis.title.y.right = element_text(colour = "red"), legend.position="none") 


#--------------------------
##Frequency plot of months

month_df<-  bank[,c(11,17)]
month_df

library(dplyr)

##sort months in order
month_df$month<-recode(month_df$month, 'jan'='1', 'feb'='2', 'mar'='3', 'apr'='4', 'may'='5', 'jun'='6', 'jul'='7', 'aug'='8', 'sep'='9', 'oct'='10', 'nov'='11', 'dec'='12')
month_df


##Frequency plot of month
plot_month <- ggplot(month_df, aes(as.factor(month), fill = y)) + 
  geom_bar(alpha = 1, position = 'identity') + ggtitle("Frequency Plot of Month")+ theme_minimal()+ 
  scale_x_discrete(limits=c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"))+
  scale_fill_brewer(palette="Paired")+
  labs(y="count", x = "month")

plot_month

##Percentage plot of month (percentage of yes and no in each month)
plot_month_perc<- month_df %>% 
  group_by(month) %>% 
  count(y) %>% 
  mutate(prop = n/sum(n)) %>% 
  ggplot(aes(x = month, y = prop)) +
  geom_col(aes(fill = y), position = "dodge") +
  geom_text(aes(label = scales::percent(prop), 
                y = prop, 
                group = month),
            position = position_dodge(width = 0.6),size=3,
            vjust = 1.5)+
  ggtitle("Percentage Plot of Month")+ theme_minimal()+ 
  scale_x_discrete(limits=c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"))+
  scale_fill_brewer(palette="Paired")+
  labs(y="percentage", x = "month") 

plot_month_perc

##if need to plot the two figures together
ggarrange(plot_month, plot_month_perc, nrow=1, common.legend = TRUE)


#-- record y==yes/y in each month

#nrow(bank[bank$month == "jan" & bank$y == 'yes',])/nrow(bank[bank$month == "jan",])

#-- jan: 0.1081081, feb: 0.1711712, mar: 0.4285714, apr:0.1911263
#-- may: 0.06652361, jun: 0.1035782, jul: 0.08640227, aug: 0.1248025
#-- sep: 0.3269231, oct: 0.4625, nov: 0.1002571, dec: 0.45

month_per<- c(0.1081081, 0.1711712, 0.4285714, 0.1911263, 0.06652361, 0.1035782, 0.08640227, 0.1248025, 
              0.3269231, 0.4625, 0.1002571, 0.45)


#-----------------------------
## rf variable importance plot of MDA MDG 
importance(modelrf)
varImpPlot(modelrf)


rf_MDA<- importance(modelrf)[,3]
rf_MDG<- importance(modelrf)[,4]

## make barplots

barplot1<- barplot(sort(rf_MDA),horiz=TRUE, las =1, xlim=c(0,100),col='salmon', main='Mean Decrease Accuracy')
text(x = round(sort(rf_MDA)), y = barplot1, label = round(sort(rf_MDA)), pos=4, cex = 1, col = "black")

barplot2<- barplot(sort(rf_MDG),horiz=TRUE, las =1,xlim=c(0,220), col='gold', main='Mean Decrease Gini')
text(x = round(sort(rf_MDG)), y = barplot2, label = round(sort(rf_MDG)), pos=4, cex = 1, col = "black")



#-------------------------------
##RF error rates plot
plot(modelrf)

plot(modelrf$err.rate[,1], type='l', lwd=2, col='sky blue')


#-------------------------------
##Other RF plots
##Interactoin plots

library(randomForestExplainer)
explain_forest(modelrf, interactions = TRUE, data = train_rf)

#variable interaction plot
plot_predict_interaction(modelrf, train_rf, "duration", "balance")


#-------------------------------
##Amelia imputation plots
## plot imputated and observed
plot(xk_bs_am, lwd=2, col = c("indianred", "dodgerblue"))
legend("topright", inset=.02, c("Imputed","Observed"), col=c("indianred", "dodgerblue"), lty=1, lwd=2)

##Mice imputed and observed
xk_bs_mice<- mice(xk_bs_data_new, m=5, method = "pmm", seed=500)
densityplot(xk_bs_mice)
xyplot(xk_bs_mice, def_flag ~ MZB + EBC + WMB, pch=18, cex=1)
