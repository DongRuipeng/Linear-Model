#include <iostream>
#include "Linear_Model.h"

int main()
{
	/*=====linear regression example=====*/
	//// generate data
	//unsigned n = 100, p = 500;
	//arma::mat X = arma::randn<arma::mat>(n, p);
	//arma::vec beta = arma::randn<arma::vec>(p);
	//beta.tail(p - 5) = arma::zeros(p - 5);
	//arma::vec e = 0.1*arma::randn<arma::vec>(n);
	//arma::vec y = X * beta + e;
	//// initilize lambda
	//double lambda = 0.01;
	//// estimate 
	//Linear_Regression model_ols(X, y);
	//Linear_Regression model_co(X, y, lambda);
	//Linear_Regression model_lar(X, y, lambda, "lars");
	//Linear_Regression model_scal(X, y, lambda, "scaled");
	//// print result
	//std::cout << "ols \t" << arma::norm(beta - model_ols.get_estimator()) << "\n";
	//std::cout << "coordinate \t" << arma::norm(beta - model_co.get_estimator()) << "\n";
	//std::cout << "least angle \t" << arma::norm(beta - model_lar.get_estimator()) << "\n";
	//std::cout << "scaled \t" << arma::norm(beta - model_scal.get_estimator()) << "\n";
	/*=====support vector machine example=====*/
	// generate data
	unsigned n = 50;
	arma::mat X = arma::randn<arma::mat>(2 * n, 2);
	X.tail_rows(n) = -X.head_rows(n);
	arma::vec y = arma::ones<arma::vec>(2 * n);
	y.subvec(n, 2 * n - 1) = -y.subvec(n, 2 * n - 1);
	arma::Row<double> position = { 30,30 };
	X.head_rows(n).each_row() + position;
	X.tail_rows(n).each_row() - position;
	// constructe svm
	SVM model(X, y, 1.0);
	model.show();
	std::cout << X.tail_rows(n).each_row() - position << std::endl;

	return 0;
}
