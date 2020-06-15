#import relevant library
library(dplyr)
library(stringr)
library(recommenderlab)
library(reshape2)
library(purrr)
library(ggplot2)
library(tidyverse)
library(doBy)
library(caret)
library(VIM)

#import movies and rating datasets
movies = read.csv('movies.csv')
ratings = read.csv('ratings_small.csv')

#merge two datasets by common variable 'movieId'
mergedf <- merge(ratings,movies,by.x = "movieId")

#retain userid,movieid,rating, movie title and genre columns
mergedf <- mergedf %>% 
  select('movieId','userId',"rating","title","genres")

#keep comedy movies because this project only build comedy movie recommender system
comedy <- mergedf %>% 
  filter(str_detect(genres,'Comedy'))

#summary of comedy dataset
summary(comedy)

#structure of comedy dataset
glimpse(comedy)

#identify the number of users and movie in comedy dataset
length(unique(comedy$movieId))
length(unique(comedy$userId))

#identify the number of users and movie in original dataset
length(unique(mergedf$movieId))
length(unique(mergedf$userId))

#check the missing value
any(is.na(comedy))
aggr(comedy)

#summary for rating variable in comedy dataset
summary(comedy$rating)

#histogram of rating variable
hist(comedy$rating, breaks = 5, xlim = c(0,5), xlab = "Rating", 
     main = "Histogram for Rating", col = "blue")

boxplot(comedy$rating, ylab = "Rating", main = "Boxplot of Rating",
        col = "blue")

#number of rating and mean rating for each movie
comedy %>% 
  group_by(title) %>% summarise(count=n(),mean=mean(rating)) %>% 
  arrange(desc(count),desc(mean)) %>% head(15)

#number of comedy movies that only have 1 rating
comedy %>% 
  group_by(title) %>% summarise(count=n()) %>% filter(count==1) %>% 
  count()

#extract year from title
comedy$title=as.character(comedy$title)

substrRight <- function(x, n){
  substr(x, nchar(x)-n+1, nchar(x))
}

y=str_sub(substrRight(comedy$title, 5),1,4)

comedy_year <- comedy %>% 
  mutate(year=y)

comedy_year$year[comedy_year$year == "986)"] <- "1986"

plot(table(comedy_year$year),type='l',
     xlab='Year',ylab='Number of Comedy Movie',main='Number of Comedy Movie by Year')


#create rating matrix
ratingmat <- dcast(comedy, userId~title, value.var = "rating", na.rm=FALSE)

#remove userid
ratingmat <- as.matrix(ratingmat[,-1]) 

## coerce into a realRatingMAtrix
realratingmat <- as(ratingmat, "realRatingMatrix")


## get some information about the real rating matrix
##number of ratings per item
views_per_movie <- colCounts(realratingmat) 
table_views <- data.frame(movie = names(views_per_movie),
                          views = views_per_movie) 
table_views_desc <- table_views[order(table_views$views, 
                                 decreasing = TRUE),] # sort by number of views

#plot top 10 comedy movies by count of rating
table_views_desc[1:15,]
ggplot(table_views_desc[1:15, ], aes(x = movie, y = views)) +
  geom_bar(stat="identity") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  ggtitle("Top 15 Comedy Movies") +
  xlab("Comedy Movie") + ylab("Number of Ratings")


##average rating for each comedy movie
average_ratings <- colMeans(realratingmat)

qplot(average_ratings) + 
  stat_bin(binwidth = 0.1) +
  ggtitle("Distribution of the Average Comedy Movie Rating") +
  xlab("Average Rating") + ylab("Counts")

#only get the average rating for the comedy movie with more than 50 ratings
average_ratings_relevant <- average_ratings[colCounts(realratingmat) > 50] 

qplot(average_ratings_relevant) + 
  stat_bin(binwidth = 0.1) +
  ggtitle("Distribution of the Relevant Average Comedy Movie Rating") +
  xlab("Average Rating") + ylab("Counts")


##average rating per user
average_ratings_per_user <- rowMeans(realratingmat)

qplot(average_ratings_per_user) +
  stat_bin(binwidth = 0.1) +
  ggtitle("Distribution of the Average Rating per User") +
  xlab("Average Rating per User") + ylab("Counts")


## explore the value of rating
vector_ratings <- as.vector(realratingmat@data)
table_ratings <- table(vector_ratings) # what is the count of each rating value
table_ratings

vector_ratings <- vector_ratings[vector_ratings != 0] # rating == 0 are NA values
vector_ratings <- factor(vector_ratings)

qplot(vector_ratings) + 
  ggtitle("Distribution of the Ratings")+xlab("Rating Score")+ylab('Count')
  

### use only users with more than 5 ratings
realratingmat5<- realratingmat[rowCounts(realratingmat)>5,]
realratingmat5

#build the recommender with an evaluation scheme using evaluationScheme() function. 
set.seed(123)
comedy_movies<-evaluationScheme(data = realratingmat5, 
                                method = "split", 
                                train = 0.80,
                                given = 5, 
                                goodRating = 4,
                                k = 1)
comedy_movies

#training set
train_comedymovie<-getData(comedy_movies, "train")

#set with the items used to build the recommendations
known_comedymovie<-getData(comedy_movies, "known") 

#set with the items used to test the recommendations
unknown_comedymovie<-getData(comedy_movies, "unknown") 



#build a memory-based UBCF model with cosine similarity 
ub_cosine <- train_comedymovie %>%
  Recommender(method = "UBCF",param=list(method="Cosine"))

#build a memory-based UBCF model with Euclidean distance 
ub_euclidean <- train_comedymovie %>%
  Recommender(method = "UBCF",param=list(method="Euclidean"))



#build a memory-based IBCF model with cosine similarity 
ib_cosine <- train_comedymovie %>%
  Recommender(method = "IBCF",param=list(method="Cosine"))

#build a memory-based IBCF model with Euclidean distance 
ib_euclidean <- train_comedymovie %>%
  Recommender(method = "IBCF",param=list(method="Euclidean"))



#build a model-based SVD model
svd <- train_comedymovie %>%
  Recommender(method="SVD",param=list(k=30))

#build a model-based Funk SVD model
svdf <- train_comedymovie %>%
  Recommender(method="SVDF",param=list(k=30))


#hybrid recommender model 
#build a UB_SVD_cosine model 
ub_svd_cosine <- HybridRecommender(Recommender(train_comedymovie,method='UBCF',param=list(method="Cosine")),
                                  Recommender(train_comedymovie,method='SVD',param=list(k=30)),
                                  weights = c(0.50,0.50))

#build a UB_SVD_euclidean model 
ub_svd_euclidean <- HybridRecommender(Recommender(train_comedymovie,method='UBCF',param=list(method="Euclidean")),
                                   Recommender(train_comedymovie,method='SVD',param=list(k=30)),
                                   weights = c(0.50,0.50))

#build a IB_SVD_cosine model 
ib_svd_cosine<- HybridRecommender(Recommender(train_comedymovie,method='IBCF',param=list(method="Cosine")),
                                      Recommender(train_comedymovie,method='SVD',param=list(k=30)),
                                      weights = c(0.50,0.50))

#build a IB_SVD_euclidean model 
ib_svd_euclidean <- HybridRecommender(Recommender(train_comedymovie,method='IBCF',param=list(method="Euclidean")),
                                      Recommender(train_comedymovie,method='SVD',param=list(k=30)),
                                      weights = c(0.50,0.50))

#build a UB_SVDF_cosine model 
ub_svdf_cosine <- HybridRecommender(Recommender(train_comedymovie,method='UBCF',param=list(method="Cosine")),
                                   Recommender(train_comedymovie,method='SVDF',param=list(k=30)),
                                   weights = c(0.50,0.50))

#build a UB_SVDF_euclidean model 
ub_svdf_euclidean <- HybridRecommender(Recommender(train_comedymovie,method='UBCF',param=list(method="Euclidean")),
                                      Recommender(train_comedymovie,method='SVDF',param=list(k=30)),
                                      weights = c(0.50,0.50))

#build a IB_SVDF_cosine model 
ib_svdf_cosine<- HybridRecommender(Recommender(train_comedymovie,method='IBCF',param=list(method="Cosine")),
                                  Recommender(train_comedymovie,method='SVDF',param=list(k=30)),
                                  weights = c(0.50,0.50))

#build a IB_SVDF_euclidean model 
ib_svdf_euclidean <- HybridRecommender(Recommender(train_comedymovie,method='IBCF',param=list(method="Euclidean")),
                                      Recommender(train_comedymovie,method='SVDF',param=list(k=30)),
                                      weights = c(0.50,0.50))



#evaluate UBCF_cosine model based on rating
ub_cosine_eval <- ub_cosine %>% 
  predict(known_comedymovie, type = "ratings") %>% 
  calcPredictionAccuracy(unknown_comedymovie)

#evaluate UBCF_euclidean model based on rating
ub_euclidean_eval <- ub_euclidean %>% 
  predict(known_comedymovie, type = "ratings") %>% 
  calcPredictionAccuracy(unknown_comedymovie)

#evaluate IBCF_cosine model based on rating
ib_cosine_eval <- ib_cosine %>% 
  predict(known_comedymovie, type = "ratings") %>% 
  calcPredictionAccuracy(unknown_comedymovie)

#evaluate IBCF_euclidean model based on rating
ib_euclidean_eval <- ib_euclidean %>% 
  predict(known_comedymovie, type = "ratings") %>% 
  calcPredictionAccuracy(unknown_comedymovie)

#evaluate SVD model based on rating
svd_eval <- svd %>% 
  predict(known_comedymovie, type = "ratings") %>% 
  calcPredictionAccuracy(unknown_comedymovie)

#evaluate Funk SVD model based on rating
svdf_eval <- svdf %>% 
  predict(known_comedymovie, type = "ratings") %>% 
  calcPredictionAccuracy(unknown_comedymovie)

#evaluate hybrid model based on rating
ub_svd_cosine_eval <- ub_svd_cosine %>% 
  predict(known_comedymovie, type = "ratings") %>% 
  calcPredictionAccuracy(unknown_comedymovie)

ub_svd_euclidean_eval <- ub_svd_euclidean %>% 
  predict(known_comedymovie, type = "ratings") %>% 
  calcPredictionAccuracy(unknown_comedymovie)

ib_svd_cosine_eval <- ib_svd_cosine %>% 
  predict(known_comedymovie, type = "ratings") %>% 
  calcPredictionAccuracy(unknown_comedymovie)

ib_svd_euclidean_eval <- ib_svd_euclidean %>% 
  predict(known_comedymovie, type = "ratings") %>% 
  calcPredictionAccuracy(unknown_comedymovie)

ub_svdf_cosine_eval <- ub_svdf_cosine %>% 
  predict(known_comedymovie, type = "ratings") %>% 
  calcPredictionAccuracy(unknown_comedymovie)

ub_svdf_euclidean_eval <- ub_svdf_euclidean %>% 
  predict(known_comedymovie, type = "ratings") %>% 
  calcPredictionAccuracy(unknown_comedymovie)

ib_svdf_cosine_eval <- ib_svdf_cosine %>% 
  predict(known_comedymovie, type = "ratings") %>% 
  calcPredictionAccuracy(unknown_comedymovie)

ib_svdf_euclidean_eval <- ib_svdf_euclidean %>% 
  predict(known_comedymovie, type = "ratings") %>% 
  calcPredictionAccuracy(unknown_comedymovie)




#combine all the evaluation result 
rating_eval <- rbind(ub_cosine_eval,
                     ub_euclidean_eval,
                     ib_cosine_eval,
                     ib_euclidean_eval,
                     svd_eval,
                     svdf_eval,
                     ub_svdf_cosine_eval,
                     ub_svdf_euclidean_eval,                  
                     ub_svd_cosine_eval,
                     ub_svd_euclidean_eval,
                     ib_svdf_cosine_eval,
                     ib_svdf_euclidean_eval,
                     ib_svd_cosine_eval,
                     ib_svd_euclidean_eval)


rating_eval


#create top n recommended movie for first user in test set
pred_ub_cosine <- predict(ub_cosine,unknown_comedymovie[1], n=5)
as(pred_ub_cosine, "list")

pred_ub_euclidean <- predict(ub_euclidean,unknown_comedymovie[1], n=5)
as(pred_ub_euclidean, "list")

pred_ib_cosine <- predict(ib_cosine,unknown_comedymovie[1], n=5)
as(pred_ib_cosine, "list")

pred_ib_euclidean <- predict(ib_euclidean,unknown_comedymovie[1], n=5)
as(pred_ib_euclidean, "list")

pred_svd <- predict(svd,unknown_comedymovie[1], n=5)
as(pred_svd, "list")

pred_svdf <- predict(svdf,unknown_comedymovie[1], n=5)
as(pred_svdf, "list")

pred_ub_svdf_cosine <- predict(ub_svdf_cosine,unknown_comedymovie[1], n=5)
as(pred_ub_svdf_cosine, "list")

pred_ub_svdf_euclidean <- predict(ub_svdf_euclidean,unknown_comedymovie[1], n=5)
as(pred_ub_svdf_euclidean, "list")

pred_ub_svd_cosine <- predict(ub_svd_cosine,unknown_comedymovie[1], n=5)
as(pred_ub_svd_cosine, "list")

pred_ub_svd_euclidean <- predict(ub_svd_euclidean,unknown_comedymovie[1], n=5)
as(pred_ub_svd_euclidean, "list")

pred_ib_svdf_cosine <- predict(ib_svdf_cosine,unknown_comedymovie[1], n=5)
as(pred_ib_svdf_cosine, "list")

pred_ib_svdf_euclidean <- predict(ib_svdf_euclidean,unknown_comedymovie[1], n=5)
as(pred_ib_svdf_euclidean, "list")

pred_ib_svd_cosine <- predict(ib_svd_cosine,unknown_comedymovie[1], n=5)
as(pred_ib_svd_cosine, "list")

pred_ib_svd_euclidean <- predict(ib_svd_euclidean,unknown_comedymovie[1], n=5)
as(pred_ib_svd_euclidean, "list")




