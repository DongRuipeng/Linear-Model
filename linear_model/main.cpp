#include <iostream>
#include "Linear_Model.h"

int main()
{
	unsigned n = 100, p = 500;
	arma::mat X = arma::randn<arma::mat>(n, p);
	arma::vec beta = arma::randn<arma::vec>(p);
	beta.tail(p - 5) = arma::zeros(p - 5);
	arma::vec e = 0.1*arma::randn<arma::vec>(n);
	arma::vec y = X * beta + e;
	double lambda = 0.01;
	Linear_Model model_ols(X, y);
	Linear_Model model_co(X, y, lambda);
	Linear_Model model_lar(X, y, lambda, "lars");
	/*model_co.show();
	model_ols.show();
	model_lar.show();*/
	//model_lar.show();
	std::cout << "error \t ols \t coordinate \t lar" << "\n";
	std::cout << "\t " << arma::norm(beta - model_ols.get_estimator());
	std::cout << "\t " << arma::norm(beta - model_co.get_estimator());
	std::cout << "\t " << arma::norm(beta - model_lar.get_estimator()) << "\n";

	/*arma::vec a = arma::randn<arma::vec>(6);
	std::cout << a << std::endl;
	a.reset();
	std::cout << a << std::endl;
	a = arma::zeros(3);
	std::cout << a << std::endl;*/
}
