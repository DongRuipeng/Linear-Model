#include "Linear_Model.h"


Linear_Model::Linear_Model(arma::mat X, arma::vec y)
{
	arma::mat G = X.t() * X;
	arma::vec eigval;
	arma::mat eigvec;
	arma::eig_sym(eigval, eigvec, G);
	for (unsigned i = 0; i < eigval.n_elem; i++)
	{
		if (eigval[i] != 0)
		{
			eigval[i] = 1 / eigval[i];
		}
		else
		{
			eigval[i] = 0;
		}
		arma::mat G_inv = eigvec * arma::diagmat(eigval) *eigvec.t();
		Linear_Model::hbeta = G_inv * X.t() * y;
	}
}

Linear_Model::Linear_Model(arma::mat X, arma::vec y, double lambda)
{
	Linear_Model::hbeta = coordinate_descent(X, y, lambda);
	Linear_Model::path = lars_path(X, y, lambda);
}

Linear_Model::~Linear_Model()
{
	std::cout << "delete the objetion ... \n";
}

Linear_Model::Solution_Path Linear_Model::lars_path(arma::mat X, arma::vec y, double lambda)
{
	unsigned r_x = arma::rank(X);
	// intilize hmu
	arma::vec hmu = arma::zeros(y.n_elem);
	// initilize the output: gamma
	arma::vec gama = arma::zeros(r_x);
	// initilize the correlation between X and y
	arma::vec hc = X.t() * y;
	// initilize the maximum correlation
	arma::vec max_c = arma::zeros(r_x);
	max_c[0] = arma::max(arma::abs(hc));
	// initilize the active set 
	arma::uvec active_set = arma::find(abs(hc) == max_c[0]);
	// initilize the complement of the active set
	arma::uvec active_set_c = arma::find(abs(hc) != max_c[0]);
	// initilize hbeta
	arma::mat hbeta = arma::zeros(X.n_cols, r_x + 1);
	// define the column number of hbeta;
	unsigned col_num_hbeta;

	for (unsigned i = 0; i < r_x; i++)
	{
		col_num_hbeta = i;
		// get the sign of the correlation in the active set
		arma::vec s = get_sign(hc(active_set));
		// make the angle between X_a and y less than pi/2
		arma::mat X_a = X.cols(active_set) * arma::diagmat(s);
		// get the Gram matrix of X_a
		arma::mat G_a = X_a.t() * X_a;
		// get the inverse of G_a
		arma::mat G_a_inv = arma::inv_sympd(G_a);
		// initilize a one vector
		arma::vec one_vec = arma::ones(active_set.n_elem);
		// get the bisection of X_a
		arma::mat A_a_temp = one_vec.t() * G_a_inv * one_vec;
		double A_a = A_a_temp[0, 0];
		A_a = 1 / sqrt(A_a);
		// get the bisection line of X_a
		arma::vec u_a = A_a * X_a * G_a_inv * one_vec;
		// initilize a temp variable
		arma::vec a = X.t() * u_a;
		// get the gama(i) in the i^th step
		arma::vec temp_gama_1 = (max_c[i] - hc(active_set_c)) / (A_a - a(active_set_c));
		temp_gama_1(arma::find(temp_gama_1 <= 0)).fill(INFINITY);
		arma::vec temp_gama_2 = (max_c[i] + hc(active_set_c)) / (A_a + a(active_set_c));
		temp_gama_2(arma::find(temp_gama_2 <= 0)).fill(INFINITY);
		arma::vec temp_gama = arma::zeros(active_set_c.n_rows);
		for (unsigned j = 0; j < active_set_c.n_rows; j++)
		{
			temp_gama[j] = std::min(temp_gama_1[j], temp_gama_2[j]);
		}
		gama[i] = temp_gama.min();
		arma::vec d = arma::diagmat(s) * A_a * G_a_inv * one_vec;
		arma::vec temp_d = -hbeta.submat(active_set, arma::uvec{ i }) / d;
		temp_d(arma::find(temp_d <= 0)).fill(INFINITY);
		if (gama[i] > temp_d.min())
		{
			gama[i] = temp_d.min();
			// update hbeta
			hbeta.submat(active_set, arma::uvec{ i + 1 }) = hbeta.submat(active_set, arma::uvec{ i }) + gama[i] * d;
			// update max_c
			max_c[i + 1] = max_c[i] - gama[i] * A_a;
			if (max_c[i + 1] <= lambda)
			{
				break;
			}
			// update hc and hmu
			hmu = hmu + gama[i] * u_a;
			hc = X.t() * (y - hmu);
			// update the active set and the complement
			arma::uvec index = arma::find(temp_d == temp_d.min());
			active_set_c.insert_rows(0, active_set.elem(index));
			for (unsigned k = 0; k < index.n_elem; k++)
			{
				active_set.shed_row(index[k]);
			}
			continue;
		}
		// update hbeta
		hbeta.submat(active_set, arma::uvec{ i + 1 }) = hbeta.submat(active_set, arma::uvec{ i }) + gama[i] *  d;

		// update max_c
		max_c[i + 1] = max_c[i] - gama[i] * A_a;
		if (max_c[i + 1] <= lambda)
		{
			break;
		}
		// update the active set and the complement
		arma::uvec new_x_set = arma::find(temp_gama == gama[i]);
		active_set.insert_rows(0, active_set_c.elem(new_x_set));
		for (unsigned k = 0; k < new_x_set.n_elem; k++)
		{
			active_set_c.shed_row(new_x_set[k]);
		}
		// update hc and hmu
		hmu = hmu + gama[i] * u_a;
		hc = X.t() * (y - hmu);
		if (active_set_c.n_rows == 0)
		{
			i = i + 1;
			// get the sign of the correlation in the active set
			arma::vec s = get_sign(hc(active_set));
			// make the angle between X_a and y less than pi/2
			arma::mat X_a = X.cols(active_set) * arma::diagmat(s);
			// get the Gram matrix of X_a
			arma::mat G_a = X_a.t() * X_a;
			// get the inverse of G_a
			arma::mat G_a_inv = arma::inv_sympd(G_a);
			// initilize a one vector
			arma::vec one_vec = arma::ones(active_set.n_elem);
			// get the bisection of X_a
			arma::mat A_a_temp = one_vec.t() * G_a_inv * one_vec;
			double A_a = A_a_temp[0, 0];
			A_a = 1 / sqrt(A_a);
			gama[i] = max_c[i] / A_a;
			// update max_c
			max_c[i + 1] = 0;
			// update hbeta
			hbeta.submat(active_set, arma::uvec{ i + 1 }) = hbeta.submat(active_set, arma::uvec{ i }) + gama[i] * A_a * arma::diagmat(s) * G_a_inv * one_vec;
			break;
		}
		// std::cout << max_c << std::endl;
	}
	col_num_hbeta = col_num_hbeta + 1;
	// std::cout << col_num_hbeta << std::endl;
	hbeta.shed_col(0);
	hbeta.shed_cols(col_num_hbeta,hbeta.n_cols - 1);
	max_c.shed_row(0);
	max_c.shed_rows(col_num_hbeta, max_c.n_rows - 1);
	// std::cout << max_c << std::endl;
	Linear_Model::Solution_Path path;
	path.C_path = max_c;
	path.hbeta_path = hbeta;
	return path;
}

arma::vec Linear_Model::coordinate_descent(arma::mat X, arma::vec y, double lambda)
{
	unsigned p = X.n_cols;
	unsigned MAX_ITERATION = 200;
	unsigned NUM_ITERATION;
	double eps = 1e-3;
	// initilize hbeta
	arma::vec hbeta = arma::zeros(p);
	// initilize r
	arma::vec r = y;
	for (unsigned i = 0; i < MAX_ITERATION; i++)
	{
		// update beta_old
		arma::vec hbeta_old = hbeta;
		for (unsigned j = 0; j < p; j++)
		{
			arma::mat temp_z = (X.col(j).t() * r) / (X.col(j).t() * X.col(j)) + hbeta_old[j];
			double z = temp_z[0, 0];
			// update beta_j
			hbeta[j] = soft_threshold(z, lambda);
			// update r
			r = r - (hbeta[j] - hbeta_old[j])*X.col(j);
		}
		NUM_ITERATION = i + 1;
		if (arma::norm(hbeta - hbeta_old) < eps)
		{
			break;
		}
	}
	if (NUM_ITERATION == MAX_ITERATION)
	{
		std::cout << "the number of the iteration is over MAX_ITERATION ! \n";
	}
	return hbeta;
}

arma::vec Linear_Model::get_sign(arma::vec x)
{
	arma::vec s;
	s.zeros(arma::size(x));
	s(arma::find(x >= 0)).fill(1);
	s(arma::find(x < 0)).fill(-1);
	return s;
}

double Linear_Model::soft_threshold(double z, double lambda)
{
	if (z > lambda)
	{
		z = z - lambda;
	}
	else if (z < -lambda)
	{
		z = z + lambda;
	}
	else
	{
		z = 0;
	}
	return z;
}

//void Linear_Model::estimator_print()
//{
//	if (Linear_Model::hbeta_path.is_empty())
//	{
//		std::cout << "there is no penalty, the OLS estimator : \n";
//		std::cout << Linear_Model::hbeta << std::endl;
//	}
//	else
//	{
//		std::cout << "there is a penalty, the sparse esitmator : \n";
//		std::cout << Linear_Model::hbeta << std::endl;
//		std::cout << "the path of least angle regression is : \n";
//		std::cout << Linear_Model::hbeta_path;
//	}
//}
//
//void Linear_Model::estimator_error(arma::vec beta)
//{
//	if (!Linear_Model::hbeta_path.is_empty())
//	{
//		std::cout << "the estimate error of the coordinate descend algorithm : \n";
//		std::cout << "\t\t" << arma::norm(Linear_Model::hbeta - beta) << "\n";
//		/*std::cout << "the estimate error of the least angle regression : \n";
//		std::cout << "\t\t" << arma::norm(Linear_Model::hbeta_path)*/
//	}
//}
