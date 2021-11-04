/*
Utility functions.

Copyright (c) 2021 Gabriele Gilardi
*/

/* Headers */
#include "utils.hpp"


/* Returns an array of random numbers (uniform, double) */
ArrayXXd rnd(uniform_real_distribution<double> dist, mt19937_64& generator,
             int nr, int nc)
{
    ArrayXXd rn;

    rn.setZero(nr, nc);
    for (int i=0; i<nr; i++) {
        for (int j=0; j<nc; j++) {
            rn(i, j) = dist(generator);
        }
    }

    return rn;
}

/* Returns an array of random numbers (normal, double) */
ArrayXXd rnd(normal_distribution<double> dist, mt19937_64& generator,
             int nr, int nc)
{
    ArrayXXd rn;

    rn.setZero(nr, nc);
    for (int i=0; i<nr; i++) {
        for (int j=0; j<nc; j++) {
            rn(i, j) = dist(generator);
        }
    }

    return rn;
}

/* Returns the cumulative sum of the elements in an array */
ArrayXi cumsum(ArrayXi X)
{
    int nel = X.size();
    ArrayXi Xsum = X;

    for (int i=1; i<nel; i++) {
        Xsum(i) = Xsum(i-1) + X(i);
    }

    return Xsum;
}

/* Returns the cumulative product of the elements in an array */
ArrayXi cumprod(ArrayXi X)
{
    int nel = X.size();
    ArrayXi Xprod = X;

    for (int i=1; i<nel; i++) {
        Xprod(i) = Xprod(i-1) * X(i);
    }

    return Xprod;
}

/* Returns the standard deviation of an array */
double stdev(ArrayXd X, int ddof)
{
    double mu = X.mean();
    ArrayXd d2 = (X - mu).square();

    if (ddof == 0) {
        return sqrt (d2.sum() / static_cast<double>(X.size()));            // Biased
    }
    else {
        return sqrt (d2.sum() / static_cast<double>(X.size() - 1));        // Unbiased
    }
}

/* Returns the root-mean-square-error (RMSE) between two arrays */
double rmse(ArrayXd X, ArrayXd Y)
{
    ArrayXd d2 = (X - Y).square();

    return sqrt(d2.sum() / static_cast<double>(X.size()));
}

/* Returns the accuracy (in percent) between two arrays */
double accuracy(ArrayXd A, ArrayXd B, double tol)
{
    ArrayXd d = (A - B).abs();
    int count = (d < tol).count();

    return 100.0 * static_cast<double>(count) / static_cast<double>(A.size());
}

/* Returns the Pearson correlation between two arrays */
double calc_corr(ArrayXd X, ArrayXd Y) 
{
    int nel = X.size();
    double xm = X.mean();
    double ym = Y.mean();

    double num = (X * Y).sum() - static_cast<double>(nel) * xm * ym;
    double den = sqrt((X * X).sum() - static_cast<double>(nel) * xm * xm) *
                 sqrt((Y * Y).sum() - static_cast<double>(nel) * ym * ym);
    
    return num / den;
}


/* Normalizes an array using mean <mu> and standard deviation <sigma> */
ArrayXd normalize(ArrayXd X, double mu, double sigma)
{
    return (X - mu) / sigma;
}

/* Scales an array in the interval [a, b] */
ArrayXd scale(ArrayXd X, double Xmin, double Xmax, double a, double b)
{
    double delta = (b - a)/ (Xmax - Xmin);

    return a + delta * (X - Xmin);
}

/* Returns a shuffled sequence of the indexes [0, nel-1] */
ArrayXi shuffle(int nel, int seed)
{
    // Init index sequence
    ArrayXi idx;
    idx.setZero(nel);
    for (int i=1; i<nel; i++) {
        idx(i) = i;
    }

    // Randomly swap two indexes
    mt19937_64 generator(seed);
    for (int i=nel-1; i>0; i--) {
        uniform_int_distribution<int> unif(0, i);
        swap(idx[i], idx[unif(generator)]);
    }

    return idx;
}

/* Exact least square solution */
VectorXd exact_sol(MatrixXd A, VectorXd b)
{
    MatrixXd AtA = (A.transpose() * A);
    VectorXd Atb = (A.transpose() * b);

    VectorXd x = AtA.partialPivLu().solve(Atb);

    return x;

}
 