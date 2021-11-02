/*
Headers for file "utils.cpp".

Copyright (c) 2021 Gabriele Gilardi
*/

#ifndef __UTILS_H_
#define __UTILS_H_

double mean(double* X, int nel);
double stdev(double* X, int nel, int ddof=0);
double rmse(double* X, double* Y, int nel);
double rmse(double** X, double** Y, int nr, int nc);
double accuracy(double* A, double* B, int nel, double tol=1.e-5);
double accuracy(double** A, double** B, int nr, int nc, double tol=1.e-5);
double calc_corr(double* X, double* Y, int nel);
double calc_corr(double** X, double** Y, int nr, int nc);
double* normalize(double* X, int nel, double mu=0.0, double sigma=1.0);
double* scale(double* X, int nel, double Xmin, double Xmax, double a=-1.0,
              double b=+1.0);
void LU_dcmp(double** A, int nel, double tol, int* P, int& n_swaps);
void LU_bksb(double** LU, int nel, double* B, int* P, double* X);
void LU_inv(double** LU, int nel, int* P, double** invA);
double LU_det(double** LU, int nel, int swaps);
double* solve_AX(double** A, int nel, double* B, double tol);
double** A_inv(double** A, int nel, double tol);
double A_det(double** A, int nel, double tol);
double** MP_inv(double** A, int nr, int nc, double tol);
int* range(int a, int b);
int* range(int a);
int* shuffle(int nel, int seed=1234567890);

#endif
 