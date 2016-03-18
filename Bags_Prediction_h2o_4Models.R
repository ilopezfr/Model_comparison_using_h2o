#################################
##  PREDICTIVE MODELS IN H2O ####
#################################
# Model 1: Distributed Random Forest DRF
# Model 2: Gradient Boosting Machine GBM
# Model 3: Generalized Linear Model GLM
# Model 4: Deep Learning DL


#Go with 4th Iteration: K-Means with 10 numeric PCs: 20 clusters. 
#Combine ABPR, Categorical variables, clusterID and PCA dataset.

#  Iteration 1-- Include: "ARR_STN", "SCH_MILES",
# "DT_Year", "DT_Month", "DT_Season", "DT_Week", "DT_DayOfTheWeek", 
# "DT_Workingday", "DT_Hour", "DT_DaySegment", "holiday"


clusters_hex <- h2o.predict(object=kmeans_hex, newdata=pca.c.hex)
testing.hex <- as.h2o(testing)
dmm.result.hex <- h2o.cbind( testing.hex$ARR_STN, 
                             testing.hex$DT_Year, testing.hex$DT_Month, testing.hex$DT_Season,
                             testing.hex$DT_Week, testing.hex$DT_DayOfTheWeek , testing.hex$DT_Workingday, 
                             testing.hex$DT_Hour, testing.hex$DT_DaySegment, testing.hex$holiday,
                             pca.c.hex, clusters_hex, testing.hex$ABPR)

dmm.result <- as.data.frame(dmm.result.hex)  #convert to data frame
colnames(dmm.result)[length(dmm.result)-1] <- "clusterID"   #Add clusterID column
dmm.result$clusterID <- as.factor(dmm.result$clusterID)   #set clusterID as factor type

dmm.result
dmm.result["Count"] <- 1  #Add a new column "Count" that assigns 1 to each row. 


#check what flights are in which cluster
#dmm.result2.hex <- h2o.cbind( testing.hex$FLIGHT_ID, pca.c.hex, clusters_hex, testing.hex$ABPR)
#dmm_clust2 =as.data.frame(dmm.result2.hex)
#dmm_clust2[which(dmm_clust2$predict == 8),c("FLIGHT_ID", "ABPR") ]

#table(testing$ABPR, dmm.result$clusterID)


#install.packages(c("sqldf", "dplyr","RH2"));
library(dplyr); library(sqldf); #library(RH2)
#detach("package:RH2")
result_sup <- dmm.result

cluster_keep <- sqldf("select sum(Count) as clust_cnt, clusterID from result_sup group by Count, clusterID")
#cluster_keep <- sqldf("select clusterID from result_sup group by clusterID")
cluster_keep <- cluster_keep[which(cluster_keep$clust_cnt > 100),] #Consider only clusters with >100 obs
#cluster_keep <- data.frame(cluster_keep)
result_keep <- sqldf("select * from result_sup where clusterID in (select clusterID from cluster_keep)")
#result_keep$Diabetes0 <- NULL
#result_keep$ID <- NULL
result_keep$Count <- NULL
result_keep_hex <- as.h2o(result_keep)
summary(result_keep_hex)


# MODEL1: Distributed Random Forest by cluster (DRF)
# create a blank data frame where we insert each model important 
# variables along with key validation statistics by each cluster.
model_results_drf <- data.frame(No_Cluster=character(),
                                variable_importances=character(),
                                relative_importance=numeric(),
                                scaled_importance=numeric(),
                                percentage=numeric(),
                                Error_ABPR=numeric(),
                                stringsAsFactors=FALSE)
# We will now loop through each cluster (the ones we kept) and produce a random forest model 
# on each cluster, extracting each important feature and saving it to our blank data frame. 

#library(h2o)
#conn <- h2o.init(max_mem_size='5g')
#h2o.shutdown()

t1 <- Sys.time()
for (i in 1:nrow(cluster_keep)){
  clust_ID <- as.character(cluster_keep[i,2])
  clustDF <- result_keep_hex[result_keep_hex$clusterID==clust_ID,] 
  clustDF <- h2o.removeVecs(clustDF, c("clusterID","Count"))
  r <- h2o.runif(clustDF)
  train <- clustDF[r < 0.6,]
  test <- clustDF[(r >= 0.6) & (r < 0.9),]
  hold <- clustDF[r >= 0.9,]
  response <- "ABPR"
  predictors <- setdiff(names(clustDF), response)
  try(drf <- h2o.randomForest(x = predictors, y = response,
                              training_frame = train, validation_frame = test,
                              ntrees = 1000, balance_classes = T), silent = T)
  drf_var_variable_importances <- as.data.frame(drf@model$variable_importances) 
  perf_drf <- h2o.performance(drf, clustDF) 
#  drf_var_variable_importances$Error_ABPR <- h2o.mse(perf_drf)[1]
  drf_var_variable_importances$No_Cluster <- clust_ID
  model_results_drf <- rbind(model_results_drf,drf_var_variable_importances) 
}
t2 <- Sys.time() 
t2-t1

model_results_drf
perf_drf

#Visualize variables and importance for each cluster
model_results_drf_filtered <- sqldf("select * from model_results_drf where scaled_importance > .05")   #Select 40% and above in importance
plot_ly(model_results_drf_filtered, x = percentage, y = variable,mode = "markers", color = No_Cluster) %>% layout( 
        title = "DRF Variable Importance by Cluster",
        xaxis = list(title = "Percentage Importance"), 
        margin = list(l = 120))

#plot_ly(model_results_drf_filtered, x = percentage, y = variable, z=Error_ABPR, text = paste("Clusters: ", No_Cluster),
#        type="scatter3d", mode="markers", color = No_Cluster)%>%
#  layout(title = "DRF Variable Importance by Cluster (Importance & Error Rate)", xaxis = list(title = "Percentage Importance"),
#         margin = list(l = 20))


### MODEL 2: GBM
model_results_gbm <- data.frame(No_Cluster=character(), 
                                variable_importances=character(),
                                relative_importance=numeric(), 
                                scaled_importance=numeric(), 
                                percentage=numeric(), 
                                Error_ABPR=numeric(), 
                                stringsAsFactors=FALSE)
#result_keep_hex = result_sup_hex 

t1 <- Sys.time()
for (i in 1:nrow(cluster_keep)){
  clust_ID <- as.character(cluster_keep[i,2])
  clustDF <- result_keep_hex[result_keep_hex$clusterID==clust_ID,] 
  clustDF <- h2o.removeVecs(clustDF, c("clusterID","Count"))
  r <- h2o.runif(clustDF)
  train <- clustDF[r < 0.6,]
  test <- clustDF[(r >= 0.6) & (r < 0.9),]
  hold <- clustDF[r >= 0.9,]
  response <- "ABPR"
  predictors <- setdiff(names(clustDF), response)
  try(gbm <- h2o.gbm(x = predictors, 
                     y = response,
                     training_frame = train, 
                     validation_frame = test,
                     ntrees = 1000,
                     max_depth = 6,
                     learn_rate = 0.1,
                     #stopping_rounds = 1, 
                     #stopping_tolerance = 0.01, 
                     #stopping_metric = "MSE", 
                     #balance_classes = T,
                     seed = 2000000), 
                    silent = T)
  gbm_var_variable_importances <- as.data.frame(gbm@model$variable_importances) 
  perf_gbm <- h2o.performance(gbm, clustDF) 
  #gbm_var_variable_importances$Error_ABPR <- h2o.mse(perf_gbm)
  #gbm_var_variable_importances$Error_ABPR <-  perf_gbm@metrics$MSE
  gbm_var_variable_importances$No_Cluster <- clust_ID
  model_results_gbm <- rbind(model_results_gbm,gbm_var_variable_importances) 
}
t2 <- Sys.time()
t2-t1
summary(model_results_gbm)
perf_gbm


# Visualize the Importance Rate
model_results_gbm_filtered <- sqldf("select * from model_results_gbm where scaled_importance > .005") 
plot_ly(model_results_gbm_filtered, x = percentage, y = variable,
        mode = "markers", color = No_Cluster) %>% layout(
        title = "GBM Variable Importance by Cluster", 
        xaxis = list(title = "Percentage Importance"), 
        margin = list(l = 120))
                 
plot_ly(model_results_gbm_filtered, x = percentage, y = variable, z=Error_ABPR, 
        text = paste("Clusters: ", No_Cluster),
        type="scatter3d", mode="markers", color = No_Cluster)


#MODEL 3: GLM
model_results_glm <- data.frame(No_Cluster=character(), 
                                variable_importances=character(),
                                coefficients=numeric(), 
                                sign=numeric(), 
                                Error_ABPR=numeric(),
                                stringsAsFactors=FALSE)
t1 <- Sys.time()
for (i in 1:nrow(cluster_keep)){
  clust_ID <- as.character(cluster_keep[i,2])
  clustDF <- result_keep_hex[result_keep_hex$clusterID==clust_ID,] 
  clustDF <- h2o.removeVecs(clustDF, c("clusterID","Count"))
  r <- h2o.runif(clustDF)
  train <- clustDF[r < 0.6,]
  test <- clustDF[(r >= 0.6) & (r < 0.9),]
  hold <- clustDF[r >= 0.9,]
  response <- "ABPR"
  predictors <- setdiff(names(clustDF), response)
  try(glm <- h2o.glm(x = predictors, y = response,
                     training_frame = train, validation_frame = test,
                     nfolds = 5, #lambda_search=TRUE, alpha=0.5,
                     family = "gaussian"), silent = T)
  glm_var_variable_importances <- as.data.frame(glm@model$standardized_coefficient_magnitudes) 
  perf_glm <- h2o.performance(glm, clustDF)
  #glm_var_variable_importances$Error_ABPR <- h2o.mse(perf_glm)
  glm_var_variable_importances$No_Cluster <- clust_ID
  model_results_glm <- rbind(model_results_glm,glm_var_variable_importances) 
}
t2 <- Sys.time()
t2-t1 
head(model_results_glm,10)
perf_glm

# Visualize the Importance Rate
model_results_glm_filtered <- sqldf("select * from model_results_glm where coefficients > .005") # Try different values for coefficient
plot_ly(model_results_glm_filtered, x = coefficients, y = names,mode = "markers", color = No_Cluster) %>% layout(
        title = "GLM Variable Importance by Cluster", 
        xaxis = list(title = "Scaled Coefficients"), 
        margin = list(l = 120)
        )
plot_ly(model_results_glm_filtered, x = coefficients, y = names, z=Error_ABPR, 
        text = paste("Clusters: ", No_Cluster),
        type="scatter3d", mode="markers", color = No_Cluster)


#MODEL 4: DEEP LEARNING
model_results_dl <- data.frame(No_Cluster=character(), 
                               variable_importances=character(),
                               relative_importance=numeric(), 
                               scaled_importance=numeric(), 
                               percentage=numeric(), 
                               variable_included=character(), 
                               Error_ABPR=numeric(), 
                               stringsAsFactors=FALSE)
t1 <- Sys.time()
for (i in 1:nrow(cluster_keep)){
  clust_ID <- as.character(cluster_keep[i,2])
  clustDF <- result_keep_hex[result_keep_hex$clusterID==clust_ID,] 
  clustDF <- h2o.removeVecs(clustDF, c("clusterID","Count"))
  r <- h2o.runif(clustDF)
  train <- clustDF[r < 0.6,]
  test <- clustDF[(r >= 0.6) & (r < 0.9),]
  hold <- clustDF[r >= 0.9,]
  response <- "ABPR"
  predictors <- setdiff(names(clustDF), response)
  try(dl <- h2o.deeplearning(x = predictors, 
                             y = response,
                             training_frame = train,
                             activation = "Tanh",
                             #balance_classes = TRUE, 
                             input_dropout_ratio = 0.2,
                             hidden = c(100, 100, 100, 100, 100), 
                              epochs = 10,
                              variable_importances = T), 
                             silent = T)
  dl_var_variable_importances <- as.data.frame(dl@model$variable_importances) 
  perf_dl <- h2o.performance(dl, clustDF) 
  #dl_var_variable_importances$Error_ABPR <- h2o.confusionMatrix(perf_dl)[2,3] 
  dl_var_variable_importances$No_Cluster <- clust_ID
  model_results_dl <- rbind(model_results_dl,dl_var_variable_importances) }
t2 <- Sys.time()
t2-t1 
perf_dl
head(model_results_dl,10)

# Visualize
model_results_dl_filtered <- sqldf("select * from model_results_dl where scaled_importance > .98") # 0.98?
plot_ly(model_results_dl_filtered, x = percentage, y = variable, mode = "markers", color = No_Cluster) %>% layout(
  title = "DL Variable Importance by Cluster", xaxis = list(title = "Percentage Importance"), margin = list(l = 120)
  )
plot_ly(model_results_dl_filtered, x = percentage, y = variable, z=Error_Diabetes, 
        text = paste("Clusters: ", No_Cluster),
        type="scatter3d", mode="markers", color = No_Cluster)

### PREDICT (h2o) ####
#1. Using clusterID as predictor
#2. Without using clusterID
summary(result_keep_hex)
r <- h2o.runif(result_keep_hex)
train <- result_keep_hex[r < 0.6,]
test <- result_keep_hex[(r >= 0.6) & (r < 0.9),]
hold <- result_keep_hex[r >= 0.9,]
response <- "ABPR"
predictors <- setdiff(names(result_keep_hex), response)


## *************** DRF *************************************** 

#2nd Iteration: Removing: holiday, DT_Year, DT_DaySegment, DT_Season
predictors_drf2 <- setdiff(predictors, c(response, "holiday", "DT_Year", "DT_DaySegment", "DT_Season"))
drf2
#3rd Iteration: Removing ClusterID
predictors_drf3 <- setdiff(predictors, c(response, "clusterID"))

drf3 <- h2o.randomForest(x = predictors_drf3, 
                        y = response, 
                        training_frame = train, 
                        #validation_frame = test,
                        nfolds = 10, 
                        ntrees = 1000,
                        stopping_metric = MSE,
                        stopping_rounds = 4, 
                        #balance_classes = T)

drf3
dl_var_variable_importances <- as.data.frame(drf2@model$variable_importances) #See variable Importance
dl_var_variable_importances

drf_pred_hex1 <- h2o.predict(drf3, newdata=hold)  #predict on hold
perf_drf1 <- h2o.performance(drf3, hold); perf_drf1

drf_pred_hex2 <- h2o.predict(drf3, newdata=result_keep_hex) #predict on the entire data set (?)
perf_drf2 <- h2o.performance(drf3, result_keep_hex); perf_drf2

hold$DRF_predict1 <- drf_pred_hex1$predict   #Save predicted column in hold data
result_keep_hex$DRF_predict2 <- drf_pred_hex2$predict   #Save predicted column in dataset

#New Ranfom Forest:
<- h2o.randomForest(x = predictors_drf3, 
                    y = response, 
                    training_frame = train, 
                    #validation_frame = test,  #Do I realy need this? It's only to create Confussion Matrix in classification problems 
                    importance=T, 
                    stat.type = "GINI",
                    ntree = 1000, 
                    depth = 50, 
                    nodesize = 5, 
                    #oobee = T, ?
                    classification = FALSE, 
                    #type = "BigData" ?
                    )  


## *************** GBM *************************************** 
gbm2 <- h2o.gbm(x = predictors,
               y = response, 
               training_frame = train, 
               validation_frame = test,
               ntrees = 1000,
               max_depth = 6,
               learn_rate = 0.1,  #same as shrinkage
               distribution="gaussian",
               #stopping_rounds = 1, 
               #stopping_tolerance = 0.01, 
               #stopping_metric = "misclassification", 
               #balance_classes = T,
               seed = 2000000) 
gbm1
gbm_pred_hex1 <- h2o.predict(gbm1, newdata=hold)     #hold 
perf_gbm1 <- h2o.performance(gbm1, hold) ; perf_gbm1

gbm_pred_hex2 <- h2o.predict(gbm1, newdata=result_keep_hex)   #result_keep_hex
perf_gbm2 <- h2o.performance(gbm1, result_keep_hex) ; perf_gbm2

hold$GBM_predict1 <- gbm_pred_hex1$predict   #Save predicted column in hold data
result_keep_hex$GBM_predict2 <- gbm_pred_hex2$predict  #Save predicted column in result_keep_hex


## *************** GLM *************************************** 
glm <- h2o.glm(x = predictors, 
               y = response,
               training_frame = train, 
               validation_frame = test, 
               nfolds = 5,
               family = "gaussian")
glm
glm_pred_hex1 <- h2o.predict(glm, newdata=hold)
perf_glm1 <- h2o.performance(glm, hold) ; perf_glm1

glm_pred_hex2 <- h2o.predict(glm, newdata=result_keep_hex) 
perf_glm2 <- h2o.performance(glm, result_keep_hex) ; perf_glm2

hold$GLM_predict1 <- glm_pred_hex1$predict   #Save predicted column in hold data
result_keep_hex$GLM_predict2 <- glm_pred_hex2$predict #Save predicted column in result_keep_hex

## *************** DEEP LEARNING *************************************** 
dl <- h2o.deeplearning(x = predictors,
                       y = response,
                       training_frame = train,
                       activation = "Tanh",
                       #balance_classes  = TRUE,
                       hidden = c(100, 100, 100, 100, 100),
                       epochs = 100)         
dl
dl_pred_hex1 <- h2o.predict(dl, newdata=hold)
perf_dl1 <- h2o.performance(dl, hold) ; perf_dl1

dl_pred_hex2 <- h2o.predict(dl, newdata=result_keep_hex) 
perf_dl2 <- h2o.performance(dl, result_keep_hex) ; perf_dl2

hold$DL_predict1 <- dl_pred_hex1$predict        #Save predicted column in hold data
result_keep_hex$DL_predict2 <- dl_pred_hex2$predict  #Save predicted column in result_keep_hex

hold_pred <- as.data.frame(hold); names(hold_pred)
result_pred <- as.data.frame(result_keep_hex)

write.csv(hold_pred, file="PREDICTIONS_hold_4Models_2.1.csv", row.names=FALSE)
write.csv(result_pred, file="PREDICTIONS_result_4Models_2.csv", row.names=FALSE)

?write.csv

## *************** Viz Clusters **************************** 
DF_Final_Viz <- as.data.frame(result_keep_hex) 
str(DF_Final_Viz)

agg_nhanes_Viz <- sqldf("select clusterID, 
                        avg(ABPR) as ABPR,
                        avg(DRF_predict) as DRF,
                        avg(GBM_predict) as GBM, 
                        avg(GLM_predict) as GLM, 
                     
                        from DF_Final_Viz group by clusterID") 

library(reshape)
agg_nhanes_Viz2 <- melt(agg_nhanes_Viz, id="clusterID")
plot_ly(agg_nhanes_Viz2, x = clusterID, y = value, type = "bar", color = variable)

#write.csv(DF_Final_Viz, file="4_Models_H2O_vs_Actual_ABPR.csv")
