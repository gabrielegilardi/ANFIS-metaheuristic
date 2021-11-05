/*
Headers for file "utils.cpp".

Copyright (c) 2021 Gabriele Gilardi
*/

#ifndef __UTILS_H_
#define __UTILS_H_

#include <random>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

ArrayXXd rnd(uniform_real_distribution<double> dist, mt19937_64& gen,
             int nr=1, int nc=1);
ArrayXXd rnd(normal_distribution<double> dist, mt19937_64& gen,
             int nr=1, int nc=1);
ArrayXi cumsum(ArrayXi X);
ArrayXi cumprod(ArrayXi X);
double stdev(ArrayXd X, int ddof=0);
double rmse(ArrayXd X, ArrayXd Y);
double accuracy(ArrayXd A, ArrayXd B, double tol=1.e-5);
double calc_corr(ArrayXd X, ArrayXd Y);
ArrayXd normalize(ArrayXd X, double mu=0.0, double sigma=1.0);
ArrayXd scale(ArrayXd X, double Xmin, double Xmax, double a=-1.0, double b=1.0);
ArrayXi shuffle(int nel, mt19937_64& gen);
VectorXd exact_sol(MatrixXd A, VectorXd b);

#endif
 