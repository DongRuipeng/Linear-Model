﻿// linear_model.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include "Linear_Model.h"

int main()
{
	unsigned n = 100, p = 15;
	arma::mat X = arma::randn<arma::mat>(n, p);
	arma::vec beta = arma::zeros(p);
	arma::uvec index = { 0,1,2,3,4,5,6,7,8,9};
	beta(index) = arma::randn<arma::vec>(index.n_rows);

	arma::vec e = 0.1*arma::zeros(n);
	arma::vec y = X * beta;// +e;

	
	
	/*std::cout << "the true coefficient is \n ";
	std::cout << beta << std::endl;*/
	Linear_Model model(X, y, 0.05);
	model.estimator_print();
	// model.estimator_error(beta);
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门提示: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
