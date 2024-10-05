#' @title CV_ridge_2D
#' @description 3-fold cross validation for selecting tunning parameters
#' @param x_train x for training
#' @param y_train y for training
#' @param w Source model parameter estimators
#' @param rho Correlation of target and source model parametor estimators
#' @param var Variance of target and source model parametor estimators
#' @return A list with the best MSE, lambda and eta
#' @export
#' @import glmnet
#' @importFrom stats rnorm
#' @importFrom utils head

### accommodating binary y
CV_ridge_2D <- function(x_train, y_train, w, rho, var){
  num = as.integer(nrow(x_train) / 3)
  ss = c(rep(1, num), rep(2, num), rep(3, nrow(x_train) - 2 * num))
  ss = sample(ss)

  # Determine if y is binary or continuous
  if (length(unique(y_train)) == 2) {
    response_type = "binary"
    family_type = "binomial"
  } else {
    response_type = "continuous"
    family_type = "gaussian"
  }

  # Define theoretical optimal lambda & eta
  opt_lam = 1 / (nrow(x_train) * var[1] * (1 - rho^2))
  opt_eta = rho * opt_lam * sqrt(var[1] / var[2])

  # Define 2D search grid
  # lam_2D_grid <- seq(as.numeric(opt_lam) / 10, as.numeric(opt_lam) * 1000, length.out = 5)
  eta_2D_grid <- seq(as.numeric(opt_eta) / 10, as.numeric(opt_eta) * 1000, length.out = 10)

  # Use glmnet to choose lam candidates
  cv_fit <- cv.glmnet(x_train, y_train, alpha = 0.02, family = family_type, standardize = FALSE)  # alpha = 0 for ridge regression
  lam_2D_grid <- sort(cv_fit$lambda[1:5], decreasing = TRUE)  # Top 5 lambda candidates from glmnet

  best_metric = if (response_type == "binary") 0 else Inf
  best_lambda = NULL
  best_eta = NULL
  best_beta = NULL

  # Grid search over lambda and eta values
  for (lambda in lam_2D_grid) {
    for (eta in eta_2D_grid) {
      metrics_fold = numeric(3)  # Assuming 3-fold CV

      for (k in 1:3) {
        x = x_train[ss != k,]
        y = y_train[ss != k]
        x_val = x_train[ss == k,]
        y_val = y_train[ss == k]

        # Call the updated ridge_closed_form_result function
        fit = ridge_closed_form_result(x, y, lambda, w, eta, x_val, y_val, standard = FALSE)

        # Collect the metric for each fold
        metrics_fold[k] = fit$metric
      }

      # Calculate the average metric across folds
      avg_metric = mean(metrics_fold)

      # Update best values based on response type
      if ((response_type == "binary" && avg_metric > best_metric) || (response_type == "continuous" && avg_metric < best_metric)) {
        best_metric = avg_metric
        best_lambda = lambda
        best_eta = eta
        best_beta = fit$beta  # Last fit's beta as the best, typically you might want a consensus
      }
    }
  }



  # fit ridge regression on the whole training set
  fit_ridge_2D = ridge_closed_form_result(x=x_train, y=y_train, lam=best_lambda, w=w, eta=best_eta, x_val=x_train, y_val=y_train, standard=FALSE)
  beta_2D_best = fit_ridge_2D$beta
  mse_2D_best = fit_ridge_2D$metric

  # Return the best lambda, eta, corresponding beta, and metric
  return(list(
    best_lambda = best_lambda,
    best_eta = best_eta,
    best_beta = beta_2D_best,
    best_metric = mse_2D_best
  ))
}

