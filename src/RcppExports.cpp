// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// ridge_closed_form_result
Rcpp::List ridge_closed_form_result(const arma::mat& x, const arma::vec& y, const double lam, const arma::vec& w, const double eta, const arma::mat& x_val, const arma::vec& y_val, const bool standard);
RcppExport SEXP _angleTL_ridge_closed_form_result(SEXP xSEXP, SEXP ySEXP, SEXP lamSEXP, SEXP wSEXP, SEXP etaSEXP, SEXP x_valSEXP, SEXP y_valSEXP, SEXP standardSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const double >::type lam(lamSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type w(wSEXP);
    Rcpp::traits::input_parameter< const double >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type x_val(x_valSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y_val(y_valSEXP);
    Rcpp::traits::input_parameter< const bool >::type standard(standardSEXP);
    rcpp_result_gen = Rcpp::wrap(ridge_closed_form_result(x, y, lam, w, eta, x_val, y_val, standard));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_angleTL_ridge_closed_form_result", (DL_FUNC) &_angleTL_ridge_closed_form_result, 8},
    {NULL, NULL, 0}
};

RcppExport void R_init_angleTL(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
