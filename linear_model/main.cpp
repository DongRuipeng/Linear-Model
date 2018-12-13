#include <iostream>
#include "Linear_Model.h"

int main()
{
	// generate data
	unsigned n = 100, p = 500;
	arma::mat X = arma::randn<arma::mat>(n, p);
	arma::vec beta = arma::randn<arma::vec>(p);
	beta.tail(p - 5) = arma::zeros(p - 5);
	arma::vec e = 0.1*arma::randn<arma::vec>(n);
	arma::vec y = X * beta + e;
	// initilize lambda
	double lambda = 0.01;
	// estimate 
	Linear_Regression model_ols(X, y);
	Linear_Regression model_co(X, y, lambda);
	Linear_Regression model_lar(X, y, lambda, "lars");
	Linear_Regression model_scal(X, y, lambda, "scaled");
	// print result
	std::cout << "ols \t" << arma::norm(beta - model_ols.get_estimator()) << "\n";
	std::cout << "coordinate \t" << arma::norm(beta - model_co.get_estimator()) << "\n";
	std::cout << "least angle \t" << arma::norm(beta - model_lar.get_estimator()) << "\n";
	std::cout << "scaled \t" << arma::norm(beta - model_scal.get_estimator()) << "\n";
}
