#include "Linear_Model.h"


SVM::SVM(arma::mat X, arma::vec y, double C)
{
	arma::mat Q = get_matrix_Q(X, y);
	SVM::alpha = solver(Q, y, C);
	SVM::w = X.t() * (SVM::alpha % y);
	SVM::w = SVM::w / arma::norm(SVM::w);
	/*std::cout << "alpha is \n";
	std::cout << SVM::alpha(arma::find(SVM::alpha != 0));*/
}

SVM::~SVM()
{
	std::cout << "delete the objection ... \n";
}

void SVM::show()
{
	std::cout << "w : \n" << SVM::w << std::endl;
	std::cout << "alpha :\n" << SVM::alpha << std::endl;
}

arma::vec SVM::solver(arma::mat Q, arma::vec y, double C)
{
	unsigned n = y.n_elem;
	// initilize tau
	double tau = 1e-3;
	// initilize one vector
	arma::vec e = arma::ones(n);
	// initilize alpha
	arma::vec alpha = arma::zeros(n);
	// define maximum iteration number and eps
	unsigned MAX_ITERATION = 200;
	double eps = 1e-5;
	// initiliize the convergence_rate and the number of iteration
	double convergence_rate = INFINITY;
	unsigned iter = 0;
	// initilize indicate vector of up and low set, 1 means up, 0 means low
	arma::uvec indicate = arma::zeros<arma::uvec>(n);
	for (unsigned t = 0; t < n; t++)
	{
		if ((alpha[t] < C && y[t] == 1) || (alpha[t] > 0 && y[t] == -1))
		{
			//I_up
			indicate[t] = 1;
		}
	}
	while (iter < MAX_ITERATION && convergence_rate > eps)
	{
		// update old alpha
		arma::vec alpha_old = alpha;
		// initilize index of up set and low set
		arma::uvec index_up = arma::find(indicate == 1);
		arma::uvec index_low = arma::find(indicate == 0);
		// selecte i1 and i2
		arma::vec kkt = -y % (Q*alpha - e);
		unsigned i1 = index_up(kkt(index_up).index_max());
		arma::uvec indicate_i2 = (indicate == 0) % (kkt < kkt[i1]);
		arma::uvec index_i2 = find(indicate_i2 == 1);
		if (index_i2.is_empty())
		{
			// std::cout << "index_i2 is empty ! \n";
			break;
		}
		arma::vec b_i = kkt[i1] - kkt(index_i2);
		arma::vec a_i = Q[i1, i1] + Q.diag() - 2*y[i1]*(Q.col(i1) % y);
		arma::vec bar_a_i = a_i;
		for (unsigned i = 0; i < n; i++)
		{
			if (a_i[i] <= 0)
			{
				bar_a_i[i] = tau;
			}
		}
		arma::vec temp_cond = -(b_i % b_i) / bar_a_i.elem(index_i2);
		unsigned i2 = index_i2[temp_cond.index_min()];
		// update U and L
		double U, L;
		if (y[i1] == 1 && y[i2] == 1)
		{
			U = std::min(C - alpha[i2], alpha[i1]);
			L = std::max(-alpha[i2], alpha[i1] - C);
		}
		else if (y[i1] == -1 && y[i2] == -1)
		{
			U = std::min(alpha[i2], C - alpha[i1]);
			L = std::max(-alpha[i1], alpha[i2] - C);
		}
		else if (y[i1] == 1 && y[i2] == -1)
		{
			L = std::max(alpha[i1] - C, alpha[i2] - C);
			U = std::min(alpha[i1], alpha[i2]);
		}
		else
		{
			L = std::max(-alpha[i1], -alpha[i2]);
			U = std::min(C - alpha[i1], C - alpha[i2]);
		}
		// update alpha_i1 and alpha_i2
		double d_i2 = (-kkt[i1] + kkt[i2]) / bar_a_i[i2];
		d_i2 = std::max(d_i2, L);
		alpha[i2] = alpha[i2] + y[i2] * d_i2;
		alpha[i1] = alpha[i1] - y[i1] * d_i2;
		// update indicate vector of up and low set
		if ((alpha[i2] < C && y[i2] == 1) || (alpha[i2] > 0 && y[i2] == -1))
		{
			indicate[i2] = 1;
		}
		else
		{
			indicate[i2] = 0;
		}
		if ((alpha[i1] < C && y[i1] == 1) || (alpha[i1] > 0 && y[i1] == -1))
		{
			indicate[i1] = 1;
		}
		else
		{
			indicate[i1] = 0;
		}
		// update iter and convergence rate
		iter = iter + 1;
		convergence_rate = arma::norm(alpha - alpha_old);
	}

	return alpha;
}

arma::mat SVM::get_matrix_Q(arma::mat X, arma::vec y)
{
	unsigned n = X.n_rows;
	arma::mat Q = arma::zeros<arma::mat>(n, n);
	for (unsigned i = 0; i < n; i++)
	{
		for (unsigned j = 0; j < n; j++)
		{
			Q[i, j] = kernel(X.row(i), X.row(j))*y[i]*y[j];
		}
	}
	return Q;
}

double SVM::kernel(arma::Row<double> a, arma::Row<double> b, std::string mode)
{
	return arma::dot(a, b);
}
