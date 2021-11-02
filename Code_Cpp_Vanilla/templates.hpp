/*
Template functions.

Copyright (c) 2021 Gabriele Gilardi
*/

#ifndef __TEMPLATES_H_
#define __TEMPLATES_H_

/* Headers and namespaces */
#include<new>

using namespace std;


/* Initializes an array to a scalar */
template <typename T>
void set_Array(T* V, int nel, T v0)
{
    for (int i=0; i<nel; i++) {
        V[i] = v0;
    }

    return;
}

/* Initializes a matrix to a scalar */
template <typename T>
void set_Array(T** A, int nr, int nc, T a0)
{
    for (int i=0; i<nr; i++) {
        for (int j=0; j<nc; j++) {
            A[i][j] = a0;
        }   
    }

    return;
}

/* Dynamically allocates an array */
template <typename T>
T* new_Array(int nel)
{
    T* V = new(nothrow) T [nel];
    if (!V) {
        printf("\nArray with %d elements", nel);
        printf("\n--> Memory not allocated.\n\n");
        exit(EXIT_FAILURE);
    }

    return V;
}

/* Dynamically allocates a matrix */
template <typename T>
T** new_Array(int nr, int nc)
{
    T** A = new(nothrow) T* [nr];
    if (!A) {
        printf("\nArray with %d rows and %d columns (1)", nr, nc);
        printf("\n--> Memory not allocated.\n\n");
        exit(EXIT_FAILURE);
    }
    else {
        for (int i=0; i<nr; i++) {
            A[i] = new(nothrow) T [nc];
            if (!A[i]) {
                printf("\nArray with %d rows and %d columns (2)", nr, nc);
                printf("\n--> Memory not allocated.\n\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    return A;
}

/* Copies an array into another */
template <typename T>
void copy_Array(T* V, int nel, T* V0, int i0=0)
{
    for (int i=0; i<nel; i++) {
        V[i] = V0[i+i0];
    }

    return;
}

/* Copies a matrix into another */
template <typename T>
void copy_Array(T** A, int nr, int nc, T** A0, int i0=0, int j0=0)
{
    for (int i=0; i<nr; i++) {
        for (int j=0; j<nc; j++) {
            A[i][j] = A0[i+i0][j+j0];
        }   
    }

    return;
}

/* De-allocates a dynamically allocated array */
template <typename T>
void del_Array(T* V)
{
    delete [] V;

    return;
}

/* De-allocates a dynamically allocated matrix */
template <typename T>
void del_Array(T** A, int nr)
{
    for (int i=0; i<nr; i++) {
        delete [] A[i];
    }
    
    delete [] A;

    return;
}

/* Reshapes an array into a matrix */
template <typename T>
T** array2matrix(T* V, int nr, int nc, char col='R')
{
    int idx = 0;
    T** A = new_Array<T>(nr, nc);

    // Unwrap along the columns
    if (col == 'C') {
        for (int j=0; j<nc; j++) {
            for (int i=0; i<nr; i++) {
                A[i][j] = V[idx];
                idx++;
            }
        }
    }

    // Unwrap along the rows (default)
    else {
        for (int i=0; i<nr; i++) {
            for (int j=0; j<nc; j++) {
                A[i][j] = V[idx];
                idx++;
            }
        }
    }

    return A;
}

/* Reshapes a matrix into an array */
template <typename T>
T* matrix2array(T** A, int nr, int nc, char col='R')
{
    int idx = 0;
    T* V = new_Array<T>(nr * nc);

    // Wrap along the columns
    if (col == 'C') {
        for (int j=0; j<nc; j++) {
            for (int i=0; i<nr; i++) {
                V[idx] = A[i][j];
                idx++;
            }
        }
    }

    // Wrap along the rows (default)
    else {
        for (int i=0; i<nr; i++) {
            for (int j=0; j<nc; j++) {
                V[idx] = A[i][j];
                idx++;
            }
        }
    }

    return V;
}

/* Returns the sum of the elements in an array */
template <typename T>
T sum(T* X, int nel)
{
    T s = X[0];
    for (int i=1; i<nel; i++) {
        s += X[i];
    }

    return s;
}

/* Returns the product of the elements in an array */
template <typename T>
T prod(T* X, int nel)
{
    T s = X[0];
    for (int i=1; i<nel; i++) {
        s *= X[i];
    }

    return s;
}

/* Returns the cumulative sum of the elements in an array */
template <typename T>
T* cumsum(T* X, int nel)
{
    T* Xsum = new_Array<T>(nel);

    Xsum[0] = X[0];
    for (int i=1; i<nel; i++) {
        Xsum[i] = Xsum[i-1] + X[i];
    }

    return Xsum;
}

/* Returns the cumulative product of the elements in an array */
template <typename T>
T* cumprod(T* X, int nel)
{
    T* Xprod = new_Array<T>(nel);

    Xprod[0] = X[0];
    for (int i=1; i<nel; i++) {
        Xprod[i] = Xprod[i-1] * X[i];
    }

    return Xprod;
}

/* Returns the minimum value of an array */
template <typename T>
T value_min(T X[], int nel)
{
    T Xmin = X[0];

    for (int i=1; i<nel; i++) {
        if (X[i] < Xmin) {
            Xmin = X[i];
        }
    }

    return Xmin;
}

/* Returns the maximum value of an array */
template <typename T>
T value_max(T X[], int nel)
{
    T Xmax = X[0];

    for (int i=1; i<nel; i++) {
        if (X[i] > Xmax) {
            Xmax = X[i];
        }
    }

    return Xmax;
}

/* Returns the index corresponding to the minimum value of an array */
template <typename T>
int idx_min(T X[], int nel)
{
    int idx = 0;
    T Xmin = X[idx];

    for (int i=1; i<nel; i++) {
        if (X[i] < Xmin) {
            Xmin = X[i];
            idx = i;
        }
    }

    return idx;
}

/* Returns the index corresponding to the maximum value of an array */
template <typename T>
int idx_max(T X[], int nel)
{
    int idx = 0;
    T Xmax = X[idx];

    for (int i=1; i<nel; i++) {
        if (X[i] > Xmax) {
            Xmax = X[i];
            idx = i;
        }
    }

    return idx;
}

/* Returns the dot-product between two arrays */
template <typename T>
T dot_prod(T X[], T Y[], int nel) 
{
    T dot = X[0] * Y[0];
    for (int i=1; i<nel; i++) {
        dot += X[i] * Y[i];
    }

    return dot;
}

/* Returns the transpose of a matrix */
template <typename T>
T** transpose(T** A, int nr, int nc)
{
    T** At = new_Array<T>(nc, nr);

    for (int i=0; i<nr; i++) {
        for (int j=0; j<nc; j++) {
            At[j][i] = A[i][j];
        }
    }
 
    return At;
}

/* Returns the product between two matrixes */
template <typename T>
T** mult_AB(T** A, int rowA, int colA, T** B, int colB)
{
    T** AB = new_Array<T>(rowA, colB);

    for (int i=0; i<rowA; i++) {
        for (int j=0; j<colB; j++) {
            T s = static_cast<T>(0);
            for (int k=0; k<colA; k++) {
                s += A[i][k] * B[k][j];
            }
            AB[i][j] = s;
        }
    }

    return AB;
}

/* Returns (as array) the product between a matrix and an array */
template <typename T>
T* mult_AV(T** A, int rowA, int colA, T* V)
{
    T* AV = new_Array<T>(rowA);

    for (int i=0; i<rowA; i++) {
        T s = static_cast<T>(0);
        for (int k=0; k<colA; k++) {
            s += A[i][k] * V[k];
        }
        AV[i] = s;
    }

    return AV; 
}


#endif
 