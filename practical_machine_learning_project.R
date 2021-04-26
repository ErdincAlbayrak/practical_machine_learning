rm(list=ls())
library(caret)
library(ggplot2)

dataset <- read.csv("~/Downloads/pml-training.csv")
truetest <- read.csv("~/Downloads/pml-testing.csv")

# pre processing
# trimming data from metadata and NA columns
dataset <- dataset[!sapply(dataset, function(x) all(x == "" || is.na(x)))]
truetest <- truetest[!sapply(truetest, function(x) all(x == "" || is.na(x)))]
# dropping metadata
dataset <- dataset[, -c(1:7)]
truetest <- truetest[, -c(1:7)]
# dividing into training validation and test sets with 80-10-10
trainingIndices <- createDataPartition(y=dataset$classe,p=0.80,list=FALSE)
training <- dataset[trainingIndices,]
dataset <- dataset[-trainingIndices,]
validationIndices <- createDataPartition(y=dataset$classe,p=0.50,list=FALSE)
validation <- dataset[validationIndices,]
testing <- dataset[-validationIndices,]
rm(list = c("validationIndices","trainingIndices","dataset"))

# try logistic regression
logisticfit <- train(classe~.,data=training,method="LogitBoost",trControl=trainControl(method="cv",number=3,verboseIter=F),tuneLength=5)
logisticpredictions <- predict(logisticfit,validation)
confusionMatrix(data = logisticpredictions,reference = validation$classe)

# naive bayes
naivefit <- train(classe~.,data=training,method="nb",trControl=trainControl(method="cv",number=3,verboseIter=F),tuneLength=5)
naivepredictions <- predict(naivefit,validation)
confusionMatrix(data = naivepredictions,reference = validation$classe)

# try decision tree
treefit <- train(classe~.,data=training,method="rpart",trControl=trainControl(method="cv",number=3,verboseIter=F),tuneLength=5)
treepredictions <- predict(treefit,validation)
confusionMatrix(data = treepredictions,reference = validation$classe)

# bagged tree
bagtreefit <- train(classe~.,data=training,method="treebag",trControl=trainControl(method="cv",number=3,verboseIter=F),tuneLength=5)
bagtreepredictions <- predict(bagtreefit,validation)
confusionMatrix(data = bagtreepredictions,reference = validation$classe)

# random forest
forestfit <- train(classe~.,data=training,method="rf",trControl=trainControl(method="cv",number=3,verboseIter=F),tuneLength=5)
forestpredictions <- predict(forestfit,validation)
confusionMatrix(data = forestpredictions,reference = validation$classe)

# chose bagged tree, use it on test set
testingpredictions <- predict(forestfit,testing)
confusionMatrix(data = testingpredictions,reference = testing$classe)

# finally predict quiz question set 
testpredictions <- predict(bagtreefit,truetest)