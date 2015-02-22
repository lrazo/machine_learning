library(caret)
library(parallel)
library(doParallel)
library(tseries)

#raw data load step 


if(!file.exists('pml-training.csv')){
  training <- read.csv(url('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'))
  write.csv(training,'pml-training.csv')
}else
  training <- read.csv('pml-training.csv', header = TRUE)  

if(!file.exists('pml-testing.csv')){
  testing <- read.csv(url('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'))
  write.csv(testing,'pml-testing.csv')
}else
  testing  <- read.csv('pml-testing.csv')

#The columns with not number over a 90% removed

nasPerCol <- apply(training,2,function(x) {sum(is.na(x))});
training  <- training[,which(nasPerCol <  nrow(training)*0.9)];  

#Near zero variance predictors removed

nearZeroCol <- nearZeroVar(training, saveMetrics = TRUE)
training <- training[, nearZeroCol$nzv==FALSE]

#Columns considered as not relevant for classification removed
#(x, user_name, raw time stamp 1  and 2, "new_window" and "num_window")

training<-training[,7:ncol(training)]

#class as factor

training$classe <- factor(training$classe)

trainIndex <- createDataPartition(y = training$classe, p=0.6,list=FALSE);
trainingPartition <- training[trainIndex,];
testingPartition <- training[-trainIndex,];

#random seed

set.seed(2423)

#parallel computing for multi-core

registerDoParallel(makeCluster(detectCores()))

#three models are generated:  

#random forest   ("rf"), 

model_rf <- train(classe ~ .,  method="rf", data=trainingPartition)    

#boosted trees ("gbm")

model_gbm <-train(classe ~ ., method = 'gbm', data = trainingPartition)

#linear discriminant analysis ("lda") model   

model_lda <-train(classe ~ ., method = 'lda', data = trainingPartition) 

print("Random forest accuracy ")
rf_accuracy<- predict(model_rf, testingPartition)
print(confusionMatrix(rf_accuracy, testingPartition$classe))
print("Boosted trees accuracy ")
gbm_accuracy<- predict(model_gbm , testingPartition)
print(confusionMatrix(gbm_accuracy, testingPartition$classe))
print("Linear discriminant analysis")
lda_accuracy<- predict(model_lda , testingPartition)
print(confusionMatrix(lda_accuracy, testingPartition$classe))


#random seed

set.seed(2423)

#parallel computing for multi-core

registerDoParallel(makeCluster(detectCores()))  

controlf <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
model_rf_CV <- train(classe ~ ., method="rf",  data=trainingPartition, trControl = controlf)

print("Random forest accuracy after CV")
rf_CV_accuracy<<- predict(model_rf_CV , testingPartition)
print(confusionMatrix(rf_CV_accuracy, testingPartition$classe))

print("Variables importance at the model")
vi = varImp(model_rf_CV$finalModel)
vi$var<-rownames(vi)
vi = as.data.frame(vi[with(vi, order(vi$Overall, decreasing=TRUE)), ])
rownames(vi) <- NULL
print(vi)

pml_create_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

predictionassign<- function(){
  prediction <- predict(model_rf_CV, testing)
  print(prediction)
  answers <- as.vector(prediction)
  pml_create_files(answers)
}

predictionassign()

