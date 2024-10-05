#' @title CV_ridge_1D
#' @description 3-fold cross validation for selecting tunning parameters
#' @param x_train x for training
#' @param y_train y for training
#' @param w Source model parameter estimators
#' @param rho Correlation of target and source model parameter estimators
#' @param var Variance of target and source model parameter estimators
#' @return A list with the best MSE, beta and lambda
#' @export
#' @import glmnet

CV_ridge_1D <- function(x_train, y_train, w, rho, var){
  num = as.integer(nrow(x_train) / 3)
  ss = c(rep(1, num), rep(2, num), rep(3, nrow(x_train) - 2 * num))
  ss = sample(ss)

  # Determine if y is binary or continuous
  response_type <- ifelse(length(unique(y_train)) == 2, "binary", "continuous")

  # Define the optimal lambda
  opt_lam_1D = 1 / (nrow(x_train) * (1 - rho^2))
  lam_1D_grid = seq(as.numeric(opt_lam_1D) / 1000, as.numeric(opt_lam_1D) * 1000, length.out = 10)  # Adjust grid as necessary

  best_metric = if (response_type == "binary") 0 else Inf  # Initialize best metric
  best_lambda = NULL
  best_beta = NULL
  metric_list = numeric(length(lam_1D_grid))

  # Grid search over lambda values
  for (lambda in lam_1D_grid) {
    metrics_fold = numeric(3)  # Assuming 3-fold CV

    for (k in 1:3) {
      x = x_train[ss != k,]
      y = y_train[ss != k]
      x_val = x_train[ss == k,]
      y_val = y_train[ss == k]

      # Call the updated ridge_closed_form_result function
      fit = ridge_closed_form_result(x, y, lambda, w, lambda, x_val, y_val, standard = FALSE)

      # Collect the metric for each fold
      metrics_fold[k] = fit$metric
    }

    # Calculate the average metric across folds
    avg_metric = mean(metrics_fold)

    # Update best values based on response type
    if ((response_type == "binary" && avg_metric > best_metric) || (response_type == "continuous" && avg_metric < best_metric)) {
      best_metric = avg_metric
      best_lambda = lambda
      best_beta = fit$beta  # Last fit's beta as the best, typically you might want a consensus
    }
  }

  # fit ridge regression on the whole training set
  fit_ridge_1D = ridge_closed_form_result(x=x_train, y=y_train, lam=best_lambda, w=w, eta=best_lambda, x_val=x_train, y_val=y_train, standard=FALSE)
  beta_1D_best = fit_ridge_1D$beta
  mse_1D_best = fit_ridge_1D$metric

  # Calculate metrics and return results
  return(list(mse_1D_best = mse_1D_best, beta_1D_best = beta_1D_best, lam_1D_best = best_lambda))
}


