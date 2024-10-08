#' Simulated data for angleTL
#' @name data_angleTL
#' @docType data
#' @format A list with target x and y, source estimator
NULL

# library(MASS)
# library(corpcor)
# library(glmnet)
#
# simulate_coef <- function(mean=0, var, rho, p){
#   beta = matrix(mvrnorm(n=p, mu=mean, Sigma=var[1]), p, 1)
#   sd_delta = sqrt(var[1]+var[2]-2*rho*sqrt(var[1]*var[2]))
#
#   w = NULL
#   for(i in 1:length(sd_delta)){
#     delta = rnorm(p, mean=mean, sd = sd_delta[i])
#     w_tmp = beta - delta
#     w = cbind(w, w_tmp)
#   }
#
#
#   return(list(beta=beta, w=w))
# }
#
# var = c(1,rep(1,5))
# rho=seq(0.4,0.6,0.05)
# p=1000
#
# coef = simulate_coef(mean=0, var, rho, p)
# w.src=coef$w
# beta.tar=coef$beta
#
# ###### continuous y #########
# simulate_data <- function(n, p, beta, error=0.5){
#   x = matrix(rnorm(n*p, 0, 1), n, p)
#   y = matrix(rnorm(n, x%*%beta, error), n, 1)
#   return(list(x=x, y=y))
# }
#
# n=1000
# data = simulate_data(n, p, beta.tar, error=0.5)
# X=data$x
# y=data$y
# w.src=coef$w
# data_angleTL=list(X=X,y=y,w.src=w.src)
#
# ###### binary y #########
# simulate_data_binary <- function(n, p, beta, error=0.5){
#   x = matrix(rnorm(n*p, 0, 1), n, p)
#
#   # Linear predictor
#   linear_pred = x %*% beta
#
#   # Logistic transformation to get probabilities
#   prob = 1 / (1 + exp(-linear_pred))
#
#   # Generate binary response
#   y = rbinom(n, size=1, prob=prob)
#
#   return(list(x=x, y=y))
# }
#
# n = 100
# data_binary = simulate_data_binary(n, p, beta.tar, error=0.5)
# X = data_binary$x
# y = data_binary$y
# w.src=coef$w
#
#
