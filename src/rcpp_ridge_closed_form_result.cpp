// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

// [[Rcpp::export]]
Rcpp::List ridge_closed_form_result(const arma::mat& x,
                                    const arma::vec& y,
                                    const double lam,
                                    const arma::vec& w,
                                    const double eta,
                                    const arma::mat& x_val,
                                    const arma::vec& y_val,
                                    const bool standard) {
  int n = x.n_rows;
  int p = x.n_cols;
  int rr = std::min(n, p);

  arma::vec y_unique = arma::unique(y);
  bool is_binary = (y_unique.n_elem == 2);

  arma::vec beta_est;

  if (n > p) {
    if (standard) {
      arma::mat XtX = x.t() * x + n * lam * arma::eye(p, p);
      arma::vec XtY = x.t() * y;
      beta_est = arma::solve(XtX, XtY, arma::solve_opts::likely_sympd);
    } else {
      arma::vec w_vector = w;
      arma::mat XtX = x.t() * x + n * lam * arma::eye(p, p);
      arma::vec XtY = x.t() * y + n * eta * w;
      beta_est = arma::solve(XtX, XtY, arma::solve_opts::likely_sympd);
    }
  } else {
    arma::mat U, V;
    arma::vec s;
    arma::svd_econ(U, s, V, x);

    U = U.cols(0, rr - 1);
    s = s.subvec(0, rr - 1);
    V = V.cols(0, rr - 1);

    arma::vec d = s;

    arma::vec diag_Sigma1_inv = d / (arma::square(d) + n * lam);
    arma::mat Sigma1_inv = arma::diagmat(diag_Sigma1_inv);

    arma::vec diag_Sigma2_inv = 1 / (arma::square(d) + n * lam);
    arma::mat Sigma2_inv = arma::diagmat(diag_Sigma2_inv);

    if (standard) {
      beta_est = V * Sigma1_inv * (U.t() * y);
    } else {
      arma::mat Identity_p = arma::eye(p, p);
      arma::mat V_Vt = V * V.t();
      arma::mat tmp = V * Sigma2_inv * V.t() + (Identity_p - V_Vt) / (n * lam);
      beta_est = V * Sigma1_inv * (U.t() * y) + n * eta * tmp * w;
    }
  }

  double metric = 0.0;
  arma::vec predictions;

  if (is_binary) {
    // For binary response, use logistic regression probability estimates to compute AUC
    predictions = 1.0 / (1.0 + arma::exp(-x_val * beta_est));

    // Compute AUC using R's pROC package
    Rcpp::Environment pROC = Rcpp::Environment::namespace_env("pROC");
    Rcpp::Function roc = pROC["roc"];
    Rcpp::Function auc = pROC["auc"];
    Rcpp::List roc_obj = roc(Rcpp::_["response"] = y_val, Rcpp::_["predictor"] = predictions);
    Rcpp::NumericVector auc_value = auc(roc_obj);
    metric = auc_value[0];
  } else {
    // For continuous response, calculate MSE
    predictions = x_val * beta_est;
    metric = arma::mean(arma::square(predictions - y_val));
  }

  return Rcpp::List::create(Rcpp::Named("beta") = beta_est,
                            Rcpp::Named("metric") = metric);
}
