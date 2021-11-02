/*
Utility functions.

Copyright (c) 2021 Gabriele Gilardi
*/

/* Headers */
#include <random>

#include "templates.hpp"          // Namespace "std" is in this file


/* Returns the mean of an array */
double mean(double* X, int nel)
{
    double s = 0.0;

    for (int i=0; i<nel; i++) {
        s += X[i];
    }

    return s / static_cast<double>(nel);
}

/* Returns the standard deviation of an array */
double stdev(double* X, int nel, int ddof=0)
{
    double mu = mean(X, nel);

    double s = 0.0;
    for (int i=0; i<nel; i++) {
        s += pow(X[i] - mu, 2.0);
    }

    if (ddof == 0) {
        return sqrt(s / static_cast<double>(nel));            // Biased
    }
    else {
        return sqrt(s / static_cast<double>(nel - 1));        // Unbiased
    }
}

/* Returns the root-mean-square-error (RMSE) between two arrays */
double rmse(double* X, double* Y, int nel)
{
    double s = 0.0;
    for (int i=0; i<nel; i++) {
        double d = X[i] - Y[i];
        s += d * d;
    }

    return sqrt(s / static_cast<double>(nel));
}

/* Returns the root-mean-square-error (RMSE) between two matrixes */
double rmse(double** X, double** Y, int nr, int nc)
{
    // Un-wrap the two matrixes
    double* Xv = matrix2array(X, nr, nc);
    double* Yv = matrix2array(Y, nr, nc);

    // Use array-type RMSE
    double s = rmse(Xv, Yv, nr * nc);

    // Delete dynamic arrays
    del_Array(Xv);
    del_Array(Yv);

    return s;
}

/* Returns the accuracy (in percent) between two arrays */
double accuracy(double* A, double* B, int nel, double tol=1.e-5)
{
    int count = 0;
    for (int i=0; i<nel; i++) {
        double d = fabs(A[i] - B[i]);
        if (d < tol) {
            count++;
        }
    }

    return 100.0 * static_cast<double>(count) / static_cast<double>(nel);
}

/* Returns the accuracy (in percent) between two matrixes */
double accuracy(double** A, double** B, int nr, int nc, double tol=1.e-5)
{
    // Un-wrap the two matrixes
    double* Av = matrix2array(A, nr, nc);
    double* Bv = matrix2array(B, nr, nc);

    // Use array-type accuracy
    double s = accuracy(Av, Bv, nr * nc, tol);

    // Delete dynamic arrays
    del_Array(Av);
    del_Array(Bv);

    return s;
}

/* Returns the Pearson correlation between two arrays */
double calc_corr(double* X, double* Y, int nel) 
{
    double xm = mean(X, nel);
    double ym = mean(Y, nel);

    double num = dot_prod(X, Y, nel) - static_cast<double>(nel) * xm * ym;
    double den = sqrt(dot_prod(X, X, nel) - static_cast<double>(nel) * xm * xm) *
                 sqrt(dot_prod(Y, Y, nel) - static_cast<double>(nel) * ym * ym);
    
    return (num / den);
}

/* Returns the Pearson correlation between two matrixes */
double calc_corr(double** X, double** Y, int nr, int nc) 
{
    // Un-wrap the two matrixes
    double* Xv = matrix2array(X, nr, nc);
    double* Yv = matrix2array(Y, nr, nc);

    // Use array-type Pearson correlation
    double s = calc_corr(Xv, Yv, nr * nc);

    // Delete dynamic arrays
    del_Array(Xv);
    del_Array(Yv);

    return s;
}

/* Normalizes an array using mean <mu> and standard deviation <sigma> */
double* normalize(double* X, int nel, double mu=0.0, double sigma=1.0)
{
    double* Xnorm = new_Array<double>(nel);
    
    for (int i=0; i<nel; i++) {
        Xnorm[i] = (X[i] - mu) / sigma;
    }

    return Xnorm;
}

/* Scales an array in the interval [a, b] */
double* scale(double* X, int nel, double X_min, double X_max, double a=-1.0,
              double b=+1.0)
{
    double* Xscale = new_Array<double>(nel);
    double delta = (b - a)/ (X_max - X_min);

    for (int i=0; i<nel; i++) {
        Xscale[i] = a + delta * (X[i]- X_min);
    }

    return Xscale;
}

/*
LU decomposition of a square matrix.

A           (nel, nel)          Square matrix to decompose
P           (nel, )             Permutation matrix (stored as an array)
tol         scalar              Small number to detect matrix degeneration
swaps       scalar              Number of swaps in the permutation matrix

Matrix A is returned as LU, with L a lower triangular matrix (all elements
above the diagonal are 0s, all elements on the diagonal are 1s) and with U
an upper triangular matrix (all elements below the diagonal are 0s).

L and U are stored in matrix A, as (L-I) + U, where I is the identity matrix.
The upper triangular part of LU (inlcuding the main diagonal) is U, while the
lower triangular part of LU (excluding the main diagonal) is L.

The permutation matrix P is not stored as a matrix, but as an array containing
the column indexes where the permutation matrix would have "1". Thus, P[i] = j
indicates that element (i, j) in the permutation matrix is 1.

The returned value <swaps> is the number of swaps, used for the computation of
the determinant.    

Ref.: http://en.wikipedia.org/wiki/LU_decomposition
*/
void LU_dcmp(double** A, int nel, double tol, int* P, int& n_swap)
{
    // Init the permutation matrix to the unit matrix (i.e. no permutations)
    for (int i=0; i<nel; i++) {
        P[i] = i;
    }

    // Loop over the columns
    n_swap = 0;
    for (int i=0; i<nel; i++) {
        double maxA = 0.0;
        int imax = i;

        // Search the row with highest (absolute) value
        for (int k=i; k<nel; k++) {
            double absA = fabs(A[k][i]);
            if (absA > maxA) { 
                maxA = absA;
                imax = k;
            }
        }

        // If all values are smaller than <tol> stops (degenerated matrix)
        if (maxA < tol) {
            printf("\n\nRow number %d, maxA = %10.6f", i, maxA);
            printf("\n--> Matrix is degenerate.\n");
            exit(EXIT_FAILURE);
        }

        // Swap the current row with the highest (absolute) value row
        if (imax != i) {
            swap(P[i], P[imax]);        // Swap the permutation matrix
            swap(A[i], A[imax]);        // Swap the pointer rows
            n_swap++;                   // Count the swaps
        }

        // Matrix factorization
        for (int j=i+1; j<nel; j++) {
            A[j][i] /= A[i][i];
            for (int k=i+1; k<nel; k++) {
                A[j][k] -= A[j][i] * A[i][k];
            }
        }
    }

    return;
}

/*
Solves the linear system A*X = B, with <A> given as LU decomposition.

LU          (nel, nel)          LU decomposition of matrix A
P           (nel, )             Permutation matrix (stored as an array)
B           (nel, )             Array on the RHS
X           (nel, )             Array solution to the linear system 

It solves the problem L*U*X = B in two steps: first solve L*Y = B, then solve
U*X = Y.

Ref.: http://en.wikipedia.org/wiki/LU_decomposition
*/
void LU_bksb(double** LU, int nel, double* B, int* P, double* X)
{
    // Solve L*Y = B using forward subtitution (array X is used as temporary
    // memory instead to create a new array Y)
    for (int i=0; i<nel; i++) {
        X[i] = B[P[i]];
        for (int k=0; k<i; k++) {
            X[i] -= LU[i][k] * X[k];
        }
    }

    // Solve U*X = Y using backward subtitution
    for (int i=nel-1; i>=0; i--) {
        for (int k=i+1; k<nel; k++) {
            X[i] -= LU[i][k] * X[k];
        }
        X[i] /= LU[i][i];
    }

    return;
}

/*
Finds the inverse of matrix A, with <A> given as LU decomposition.

LU          (nel, nel)          LU decomposition of matrix A
P           (nel, )             Permutation matrix (stored as an array)
invA        (nel, nel)          Inverse of matrix A

Solves the linear system A*X = I, with <A> given as LU decomposition and
<I> being the identity matrix of size <nel>. The solution is found using 
backsubtitution for each column of <I>.

Ref.: http://en.wikipedia.org/wiki/LU_decomposition
*/
void LU_inv(double** LU, int nel, int* P, double** invA)
{
    double* B = new_Array<double>(nel);
    double* X = new_Array<double>(nel);

    set_Array(B, nel, 0.0);
    
    for (int i=0; i<nel; i++) {

        B[i] = 1.0;                         // Set the RHS
        LU_bksb(LU, nel, B, P, X);          // Backsubtitution

        for (int j=0; j<nel; j++) {
            invA[j][i] = X[j];              // Copy the i-th solution
        }

        B[i] = 0.0;                         // Reset the RHS
    
    }

    del_Array(B);
    del_Array(X);
    
    return;
}

/*
Returns the determinant of matrix A, with <A> given as LU decomposition.

LU          (nel, nel)          LU decomposition of matrix A
swaps       scalar              Number of swaps in the permutation matrix

Ref.: http://en.wikipedia.org/wiki/LU_decomposition
*/
double LU_det(double** LU, int nel, int swaps)
{
    double det = LU[0][0];

    for (int i=1; i<nel; i++) {
        det *= LU[i][i];
    }

    if ((swaps % 2) != 0) {
        det = -det;
    }

    return det;
}

/*
Solves the linear system A*X = B using LU decomposition.

A           (nel, nel)        System matrix
B           (nel, )           Array on the RHS
tol         scalar            Small number to detect matrix degeneration
X           (nel, )           Array solution to the linear problem 

Note: <A> is not changed, a copy is used for the LU decomposition.
*/
double* solve_AX(double** A, int nel, double* B, double tol)
{
    int n_swap;
    int* P = new_Array<int>(nel);
    double ** LU = new_Array<double>(nel, nel);
    double* X = new_Array<double>(nel);

    copy_Array(LU, nel, nel, A);                // Use a copy
    LU_dcmp(LU, nel, tol, P, n_swap);           // Decomposition
    LU_bksb(LU, nel, B, P, X);                  // Backsubtitution

    del_Array(P);
    del_Array(LU, nel);

    return X;
}

/*
Returns the inverse of matrix A using LU decomposition.

A           (nel, nel)          Matrix to invert
tol         scalar              Small number to detect matrix degeneration
invA        (nel, nel)          Inverse of matrix A

Note: <A> is not changed, a copy is used for the LU decomposition.
*/
double** A_inv(double** A, int nel, double tol)
{
    int n_swap;
    int* P = new_Array<int>(nel);
    double** LU = new_Array<double>(nel, nel);
    double** invA = new_Array<double>(nel, nel);

    copy_Array(LU, nel, nel, A);                // Use a copy
    LU_dcmp(LU, nel, tol, P, n_swap);           // Decomposition
    LU_inv(LU, nel, P, invA);                   // Inverse

    del_Array(P);
    del_Array(LU, nel);

    return invA;
}

/*
Returns the determinant of matrix A using LU decomposition.

A           (nel, nel)          Matrix
tol         scalar              Small number to detect matrix degeneration

Note: <A> is not changed, a copy is used for the LU decomposition.
*/
double A_det(double** A, int nel, double tol)
{
    int n_swap;
    int* P = new_Array<int>(nel);
    double det;
    double** LU = new_Array<double>(nel, nel);

    copy_Array(LU, nel, nel, A);                // Use a copy
    LU_dcmp(LU, nel, tol, P, n_swap);           // Decomposition
    det = LU_det(LU, nel, n_swap);              // Determinant

    del_Array(P);
    del_Array(LU, nel);

    return det;
}

/*
Returns the Moore-Penrose inverse of matrix A , i.e. (A^T * A)^(-1) * A^t,
using LU decomposition.

A           (nr, nc)          Matrix (square or non-square)
tol         scalar            Small number to detect matrix degeneration
invA        (nc, nr)          Inverse of matrix A

Note: <A> is not changed.

Ref.: http://en.wikipedia.org/wiki/Moore-Penrose_inverse
*/
double** MP_inv(double** A, int nr, int nc, double tol)
{
    // A^T, shape (nc, nr)
    double** At = transpose(A, nr, nc);

    // A^T * A, shape (nc, nc)
    double** AtA = mult_AB(At, nc, nr, A, nc);

    // Decomposition of (A^T * A)
    int swaps;
    int* P = new_Array<int>(nc);
    LU_dcmp(AtA, nc, tol, P, swaps);

    // (A^T * A)^(-1), shape (nc, nc)
    double** AtAinv = new_Array<double>(nc, nc);
    LU_inv(AtA, nc, P, AtAinv);

    // (A^T * A)^(-1) * A^t, shape (nc, nr)
    double** invMP = mult_AB(AtAinv, nc, nc, At, nr);

    del_Array(P);
    del_Array(At, nc);
    del_Array(AtA, nc);
    del_Array(AtAinv, nc);

    return invMP;
}

/* Returns an array with the range [a, b) */
int* range(int a, int b)
{
    int nel = b - a;
    int* V = new_Array<int>(nel);
    
    for (int i=0; i<nel; i++) {
        V[i] = i + a;
    }

    return V;
}

/* Returns an array with the range [0, a) */
int* range(int a)
{
    int* V = new_Array<int>(a);
    
    for (int i=0; i<a; i++) {
        V[i] = i;
    }

    return V;
}

/* Returns a shuffled sequence of the indexes [0, nel-1] */
int* shuffle(int nel, int seed=1234567890)
{
    // Init index sequence
    int* idx = range(nel);

    // Randomly swap two indexes
    mt19937_64 generator(seed);
    for (int i=nel-1; i>0; i--) {
        uniform_int_distribution<int> unif(0, i);
        swap(idx[i], idx[unif(generator)]);
    }

    return idx;
}
