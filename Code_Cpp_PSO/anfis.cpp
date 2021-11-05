/*
Multivariate Regression and Classification Using an Adaptive Neuro-Fuzzy
Inference System (Takagi-Sugeno) and Particle Swarm Optimization.

Copyright (c) 2021 Gabriele Gilardi


X           (n_samples, n_inputs)           Input dataset (training)
Y           (n_samples, n_outputs)          Output dataset (training)
Xp          (n_samples, n_inputs)           Input dataset (prediction)
Yp          (n_samples, n_labels)           Output dataset (prediction)
theta       (n_var, )                       Unrolled parameters
mu          (n_pf, )                        Mean (premise MFs)
s           (n_pf, )                        Standard deviation (premise MFs)
c           (n_pf, )                        Exponent (premise MFs)
A           (n_inputs+1, n_cf*n_outputs)    Coefficients (consequent MFs)
pf          (n_pf, )                        Premise MFs value
W           (n_cf, )                        Firing strenght value
Wr          (n_cf, )                        Firing strenght ratios
cf          (n_cf, )                        Consequent MFs value
f           (n_samples, n_outputs)          ANFIS output
combs       (n_inputs, n_cf)                Combinations of premise MFs

n_samples           Number of samples
n_inputs            Number of features/inputs
n_outputs           Number of labels/classes/outputs
n_var               Number of variables
n_mf                Number of premise MFs of each feature
n_pf                Total number of premise MFs = sum(MFs)
n_cf                Total number of consequent MFs = prod(MFs)

Notes:
- MF stands for membership function.
- premise (membership) functions are generalize Bell function defined by mean
  <mu>, standard deviation <s>, and exponent <c>.
- consequent (membership) functions are hyperplanes defined by <n_inputs+1>
  coefficients each.
- In a classification problem the number of outputs is the same as the number
  of classes.
*/

/* Headers and namespaces */
#include "ANFIS.hpp"
#include "utils.hpp"

/* ===== Helper functions ===== */

/* Returns an array (table) will all the possible classes */
ArrayXd build_class_table(ArrayXd Y, double tol)
{
    int nel = Y.size();
    Array<bool, Dynamic, 1> flag;
    flag.setZero(nel);

    flag(0) = true;                     // 1st element is for sure unique

    // Identify the unique classes grouping the quantities base on <tol>
    int n_classes = 1;
    for (int i=1; i<nel; i++) {
        bool duplicate = false;

        // Check if the i-th element is unique 
        for (int j=0; j<i; j++) {
            if (flag(j) && (fabs(Y(j) - Y(i)) < tol)) {
                duplicate = true;       // Not unique 
                break;
            }
        }

        // If the element is unique count it and mark it
        if (!duplicate) {
            n_classes++;
            flag(i) = true;
        }
    }

    // Build the array (table) with all the unique elements
    int count = 0;
    ArrayXd table;
    table.setZero(n_classes);
    for (int i=0; i<nel; i++) {
        if (flag(i)) {
            table(count) = Y(i);
            count++;
        }
    }

    return table;
}

/*
Converts the output array of a classification problem to a number in the
interval [0, n_classes-1]
*/
ArrayXi get_classes(ArrayXd Y, ArrayXd table, double tol)
{
    int nel = Y.size();
    int n_classes = table.size();

    ArrayXi Yc;
    Yc.setZero(nel);

    for (int i=0; i<nel; i++) {
        bool found = false;

        // Search the i-th element in the class table
        for (int j=0; j<n_classes; j++) {
            if (fabs(table(j) - Y(i)) < tol) {
                Yc(i) = j;
                found = true;
                break;
            }
        }

        // Every element must be in the class table if the same <tol> is used
        if (!found) {
            printf("\nClass %g", Y(i));
            printf("\n--> Not found in the class table.\n\n");
            exit(EXIT_FAILURE);
        }
    }

    return Yc;
}


/* ===== ANFIS class ===== */

/* Constructor */
AnfisType::AnfisType() {

    // Nothing to do, everything is initialized in "init"

}

/* Initializes class. */
void AnfisType::init(ArrayXi MFs0, int n_outputs0)
{
    // Passed arguments
    n_inputs = MFs0.size();
    n_outputs = n_outputs0;
    MFs = MFs0;

    // Derived quantities
    n_pf = MFs.sum();
    n_cf = MFs.prod();
    n_var = 3 * n_pf + (n_inputs + 1) * n_cf * n_outputs;

    // Build all the possible combination of premise functions
    build_combinations();
    
    // Premise function parameters
    mu.setZero(n_pf); 
    s.setZero(n_pf); 
    c.setZero(n_pf);

    // Consequent function parameters
    A.setZero(n_inputs+1, n_cf*n_outputs);

    return;
}

/* Creates the model for a regression (continuos) problem */
double AnfisType::create_model(ArrayXd theta, ArrayXXd X, ArrayXXd Y)
{
    // Build parameters mu, s, c, and A
    build_param(theta);

    // Calculate output
    ArrayXXd f = forward_steps(X);    

    // Calculate cost function
    double err = ((f - Y).square()).sum();

    return (err / 2.0);
}

/* Creates the model for a classification problem */
double AnfisType::create_model(ArrayXd theta, ArrayXXd X, ArrayXi Yc)
{
    int n_samples = X.rows();

    // Build the matrix <Y> actually used as output. Array <Yc> has dimensions
    // (n_samples, ) while <Y> has dimension (n_samples, n_classes). In this 
    // matrix, Y[j,i] = 1 specifies that the i-th sample belongs to the j-th
    // class.
    ArrayXXd Y;
    Y.setZero(n_samples, n_outputs);
    for (int i=0; i<n_samples; i++) {
        Y(i, Yc(i)) = 1.0;
    }

    // Build parameters mu, s, c, and A
    build_param(theta);

    // Calculate output
    ArrayXXd f = forward_steps(X);

    // Calculate cost function (the activation value is determined in the
    // logsig function)
    double err = ((1.0 - Y) * f - logsig(f)).sum();

    return (err / static_cast<double>(n_samples));
}

/* Evaluates a model for a regression (continuos) problem */
ArrayXXd AnfisType::eval_data(ArrayXXd Xp)
{
    // Calculate output
    ArrayXXd Yp = forward_steps(Xp);

    return Yp;
}

/* Evaluates a model for a classification problem */
ArrayXXd AnfisType::eval_data(ArrayXXd Xp, ArrayXd table)
{
    int n_samples = Xp.rows();

    // Calculate output
    ArrayXXd f = forward_steps(Xp);

    // Loop over each sample
    ArrayXXd fa, Yp;
    Yp.setZero(n_samples, 1);
    for (int i=0; i<n_samples; i++) {

        // Activation values
        fa = f_activation(f.row(i));

        // Class with max. probability expressed as index in [0, n_classes-1]
        Index r_max, c_max;
        double prob = fa.maxCoeff(&r_max, &c_max);

        // Assign best result and return the original class (for consistency
        // Yp is created and returned as a column-row)
        Yp(i, 0) = table[c_max];
    }

    return Yp;
}

/* Show ANFIS info */
void AnfisType::info()
{
    printf("\n ");
    printf("\n\n===== ANFIS info =====");
    printf("\n- Inputs: %d", n_inputs);
    printf("\n- Outputs: %d", n_outputs);
    printf("\n- Variables: %d", n_var);
    printf("\n- MF Layout:");
    for (int i=0; i<n_inputs; i++) {
        printf(" %d ", MFs(i));
    }
    printf("\n- Number of PF: %d", n_pf);
    printf("\n- Number of CF: %d", n_cf);
    printf("\n ");

    return;
}

/* Class destructor */
AnfisType::~AnfisType()
{
    // Nothing to do
}

/* 
Build all the possible combination of premise functions

For example if <n_mf> = [3, 2], the MF indexes for the first feature would be
[0, 1, 2] and for the second feature would be [3, 4]. The resulting combinations
would be <combs> = [[0 0 1 1 2 2],
                    [3 4 3 4 3 4]].

*/
void AnfisType::build_combinations()
{
    ArrayXi ics = cumsum(MFs);
    ArrayXi icp = cumprod(MFs);
    combs.setZero(n_inputs, n_cf);

    // Build the first row
    int idx = 0;
    int steps = n_cf / icp(0);
    for (int k=0; k<ics(0); k++) {
        for (int j=0; j<steps; j++) {
            combs(0, idx) = k;
            idx++;
        }
    }

    // Recursively build the other rows (if any)
    for (int m=1; m<n_inputs; m++) {
        idx = 0;
        steps = n_cf / icp(m);
        for (int i=0; i<icp(m-1); i++) {
            for (int k=ics(m-1); k<ics(m); k++) {
                for (int j=0; j<steps; j++) {
                    combs(m, idx) = k;
                    idx++;
                }
            }
        }
    }

    return;
}

/* Builds the premise/consequent parameters mu, s, c, and A */
void AnfisType::build_param(ArrayXd theta)
{
    // Premise parameters
    mu = theta(seqN(0, n_pf));
    s = theta(seqN(n_pf, n_pf));
    c = theta(seqN(2*n_pf, n_pf));

    // Consequent parameters
    A = theta(seq(3*n_pf, last)).reshaped<RowMajor>(n_inputs+1, n_cf*n_outputs);

    return;
}

/*
Calculate the output of the ANFIS layers giving the premise and consequent
parameters and the input dataset.
*/
ArrayXXd AnfisType::forward_steps(ArrayXXd X)
{
    int n_samples = X.rows();

    ArrayXd pf, W, Wr, cf;
    ArrayXXd f;

    // Init arrays
    pf.setZero(n_pf);
    W.setZero(n_cf);
    Wr.setZero(n_cf);
    cf.setZero(n_cf);
    f.setZero(n_samples, n_outputs);
    
    // Cumulative sum
    ArrayXi ics = cumsum(MFs);

    // Loop over each samples
    for (int i=0; i<n_samples; i++) {

        // Layer 1: premise functions (pf)
        int idx = 0;
        for (int j=0; j<n_pf; j++) {
            if (j >= ics(idx)) {
                idx++;
            }
            double tmp = (X(i, idx) - mu(j)) / s(j);
            pf(j) = 1.0 / (1.0 + pow(tmp * tmp, c(j)));
        }

        // Layer 2: firing strenght (W)
        for (int j=0; j<n_cf; j++) {
            double tmp = 1.0;
            for (int k=0; k<n_inputs; k++) {
                tmp *= pf(combs(k, j));
            }
            W(j) = tmp;
        }

        // Layer 3: firing strenght ratios (Wr)
        Wr = W / W.sum();

        // Layer 4 and 5: consequent functions (cf) and output (f)
        for (int m=0; m<n_outputs; m++) {
            int idx = m * n_cf;
    
            // cf
            for (int j=0; j<n_cf; j++) {
                double tmp = A(0, j+idx);
                for (int k=1; k<=n_inputs; k++) {
                    tmp += X(i, k-1) * A(k, j+idx);
                }
                cf(j) = Wr(j) * tmp;
            }

            // f
            f(i, m) = cf.sum();
        }
    }

    return f;
}

/*
Numerically stable version of the sigmoid function.

Ref.: http://fa.bianp.net/blog/2019/evaluate_logistic/#sec3
*/
ArrayXXd AnfisType::f_activation(ArrayXXd Z)
{
    int nr = Z.rows();
    int nc = Z.cols();
    ArrayXXd A;
    A.setZero(nr, nc);

    for (int i=0; i<nr; i++) {
        for (int j=0; j<nc; j++) {
            double z = Z(i, j);

            // Value in [0, +inf)
            if (z >= 0.0) {
                A(i, j) = 1.0 / (1.0 + exp(-z));
            }

            // Value in (-inf, 0)
            else {
                A(i, j) = exp(z) / (1.0 + exp(z));
            }

        }
    }

    return A;
}

/*
Numerically stable version of the log-sigmoid function.

Ref.: http://fa.bianp.net/blog/2019/evaluate_logistic/#sec3
*/
ArrayXXd AnfisType::logsig(ArrayXXd Z)
{
    int nr = Z.rows();
    int nc = Z.cols();
    ArrayXXd A;
    A.setZero(nr, nc);

    for (int i=0; i<nr; i++) {
        for (int j=0; j<nc; j++) {
            double z = Z(i, j);

            // Value in (-inf, -33.3)
            if (z < -33.3) {
                A(i, j) = z;
            }

            // Value in [-33.3, -18.0)
            else if ((z >= -33.3) && (z < -18.0)) {
                A(i, j) = z - exp(z);
            }

            // Value in [-18.0, +37.0)
            else if ((z >= -18.0) & (z < 37.0)) {
                A(i, j) = - log(1.0 + exp(-z));
            }

            // Value in [+37.0, +inf)
            else {
                A(i, j) = - exp(-z);
            }

        }
    }

    return A;
}
