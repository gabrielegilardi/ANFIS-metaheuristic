/*
Headers for file "anfis.cpp".

Copyright (c) 2021 Gabriele Gilardi
*/

#ifndef __ANFIS_H_
#define __ANFIS_H_

#include <Eigen/Dense>

using namespace Eigen;

/* Helper functions */
ArrayXd build_class_table(ArrayXd Y, double tol=1.e-5);
ArrayXi get_classes(ArrayXd Y, ArrayXd table, double tol=1.e-5);

/* ANFIS class */
class AnfisType {

    public:

        // Data
        int n_pf;                   // Number of premise MFs
        int n_cf;                   // Number of consequent MFs
        int n_var;                  // Number of variables
        ArrayXd mu;                 // Premise parameters mu
        ArrayXd s;                  // Premise parameters s
        ArrayXd c;                  // Premise parameters c
        ArrayXXd A;                 // Consequent parameters A

        // Functions
        AnfisType();
        void init(ArrayXi MFs, int n_outputs);
        double create_model(ArrayXd theta, ArrayXXd X, ArrayXXd Y);
        double create_model(ArrayXd theta, ArrayXXd X, ArrayXi Yc);
        ArrayXXd eval_data(ArrayXXd Xp);
        ArrayXXd eval_data(ArrayXXd Xp, ArrayXd table);
        void info();
        ~AnfisType();
    
    private:

        // Data
        int n_outputs;              // Number of labels/classes
        int n_inputs;               // Number of features/inputs
        ArrayXi MFs;                   // Number of MFs in each feature/input
        ArrayXXi combs;                // Combinations of premise MFs

        // Functions
        void build_combinations();
        void build_param(ArrayXd theta);
        ArrayXXd forward_steps(ArrayXXd X);
        ArrayXXd f_activation(ArrayXXd z);
        ArrayXXd logsig(ArrayXXd z);
};

#endif
 