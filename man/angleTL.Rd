% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/angleTL.R
\name{angleTL}
\alias{angleTL}
\title{angleTL}
\usage{
angleTL(X, y, w.src)
}
\arguments{
\item{X}{Variables from the target. The variables need to be completely the same set and in the same order as variables used in the model parameter estimators.}

\item{y}{Response from the target.}

\item{w.src}{A matrix of pre-trained source model parameter estimators.}
}
\value{
A list of effect estimator and tuning parameters lambda and eta from angleTL.
}
\description{
Angle-based transfer learning
}
