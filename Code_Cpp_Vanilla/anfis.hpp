/*
Headers for file "anfis.cpp".

Copyright (c) 2021 Gabriele Gilardi
*/


#ifndef __ANFIS_H_
#define __ANFIS_H_

/* Helper functions */
double* build_class_table(double* Y, int nel, int& n_class, double tol=1.e-5);
int* get_classes(double* Y, int nel, double* table, int n_class, double tol=1.e-5);

/* ANFIS class */
class AnfisType {

    public:

        // Data
        int n_pf;                   // Number of premise MFs
        int n_cf;                   // Number of consequent MFs
        int n_var;                  // Number of variables
        double* mu;                 // Premise parameters mu
        double* s;                  // Premise parameters s
        double* c;                  // Premise parameters c
        double** A;                 // Consequent parameters A

        // Functions
        AnfisType();
        void init(int n_inputs, int n_outputs, int* MFs);
        double create_model(double* theta, double** X, double** Y, int n_samples);
        double create_model(double* theta, double** X, int* Yc, int n_samples);
        double** eval_data(double** Xp, int np_samples);
        double** eval_data(double** Xp, int np_samples, double* table);
        void info();
        ~AnfisType();
    
    private:

        // Data
        int n_outputs;              // Number of labels/classes
        int n_inputs;               // Number of features/inputs
        int* MFs;                   // Number of MFs in each feature/input
        int** combs;                // Combinations of premise MFs

        // Functions
        void build_combinations();
        void build_param(double* theta);
        double** forward_steps(double** X, int n_samples);
        double f_activation(double z);
        double logsig(double z);
};

#endif
 