#pragma once
#include <armadillo>

class Linear_Model
{
public:
	Linear_Model(arma::mat X, arma::vec y);
	Linear_Model(arma::mat X, arma::vec y, double lambda);
	~Linear_Model();
	void estimator_print();
	void estimator_error(arma::vec beta);

private:
	arma::mat lars_path(arma::mat X, arma::vec y, double lambda);
	arma::vec coordinate_descent(arma::mat X, arma::vec y, double lambda);
	arma::vec get_sign(arma::vec x);
	double soft_threshold(double z, double lambda);
	arma::mat hbeta_path;
	arma::vec hbeta;
};

