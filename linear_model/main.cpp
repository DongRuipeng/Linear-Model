#include <iostream>
#include "Linear_Model.h"

int main()
{
	unsigned n = 100, p = 5;
	arma::mat X = arma::randn<arma::mat>(n, p);
	arma::vec beta = arma::randn<arma::vec>(p);
	arma::vec e = 0.1*arma::randn<arma::vec>(n);
	arma::vec y = X * beta + e;
	double lambda = 0.01;
	Linear_Model model_co(X, y, lambda);
	model_co.show();
	Linear_Model model_lar(X, y, lambda, "lars");
	model_lar.show();
}
