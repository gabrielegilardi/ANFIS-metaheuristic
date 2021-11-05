/*
Multivariate Regression and Classification Using an Adaptive Neuro-Fuzzy
Inference System (Takagi-Sugeno) and Simulated Annealing Optimization.

Copyright (c) 2021 Gabriele Gilardi


=== Arguments ===
func            Function to minimize
LB              Lower boundaries of the search space
UB              Upper boundaries of the search space
nVar            Number of variables
p               Structure with the solver parameters
args            Structure with the ANFIS data
gen             Random number generator

=== Solver parameters ===
nPop            Number of agents (population)
epochs          Number of iterations
nMove           Number of neighbours of a state evaluated at each epoch
T0              Initial temperature
alphaT          Temperature reduction rate
sigma0          Initial standard deviation used to search the neighboroud
                of a state (given as a fraction of the search space)
alphaS          Standard deviation reduction rate
prob            Probability the dimension of a state is changed
normalize       Specifies if the search space should be normalized
IntVar          List of indexes specifying which variable should be treated
                as integer
nIntVar         Number of variables that should be treated as integers
seed            Seed for the random number generator

=== Dimensions ===
(nVar, 1)       LB, UB, best_pos, sigma, LB_orig, UB_orig
(nPop, nVar)    agent_pos, neigh_pos, agent_pos_orig, neigh_pos_orig   
(nPop, 1)       agent_cost, neigh_cost
(epochs)        F
(0-nVar)        IntVar
*/


/* Headers */
#include <random>

#include "templates.hpp"          // Namespace "std" is in this file
#include "anfis.hpp"

/* Structure to pass the parameters to the SA solver */
struct Parameters {
    int nPop;
    int epochs;
    int nMove;
    double T0;
    double alphaT;
    double sigma0;
    double alphaS;
    double prob;
    bool normalize;
    int* IntVar;
    int nIntVar;
    int seed;
};

/* Structure to pass the data to the ANFIS */
struct Arguments {
    double** X;                 // Training dataset inputs/features
    double** Y;                 // Training dataset labels/outputs (regression)
    int* Yc;                    // Training dataset classes (classification)
    int n_samples;              // Number of samples training dataset
    AnfisType* agents;          // Array of ANFIS agents
};


/* Minimizes a function using simulated annealing */
double* sa(double (*func)(double*, int, Arguments), double* LB, double* UB,
           int nVar, Parameters p, Arguments args, mt19937_64& gen)
{
    /* Random generator and probability distributions */
    uniform_real_distribution<double> unif(0.0, 1.0);
    normal_distribution<double> norm(0.0, 1.0);

    double* F = new_Array<double>(p.epochs);            // Best cost for each epoch
    double* sigma = new_Array<double>(nVar);            // Standard deviation
    double* best_pos = new_Array<double>(nVar);         // Best position

    // Agent's & neighbour's position
    double** agent_pos = new_Array<double>(p.nPop, nVar);
    double** neigh_pos = new_Array<double>(p.nPop, nVar);

    // Agent's & neighbour's cost
    double* agent_cost = new_Array<double>(p.nPop);
    double* neigh_cost = new_Array<double>(p.nPop);

    /* Normalize search space */
    double* LB_orig;
    double* UB_orig;
    double** agent_pos_orig;
    double** neigh_pos_orig;
    if (p.normalize) {              
        LB_orig = new_Array<double>(nVar);
        UB_orig = new_Array<double>(nVar);
    	agent_pos_orig = new_Array<double>(p.nPop, nVar);
        neigh_pos_orig = new_Array<double>(p.nPop, nVar);
        copy_Array(LB_orig, nVar, LB);
        copy_Array(UB_orig, nVar, UB);
        set_Array(LB, nVar, 0.0);
        set_Array(UB, nVar, 1.0);
    }

    /* Initial temperature and standard deviation */
    double T = p.T0;
    for (int j=0; j<nVar; j++) {
        sigma[j] = p.sigma0 * (UB[j] - LB[j]);
    }

    /* Initial position of each agent */
    for (int i=0; i<p.nPop; i++) {
        for (int j=0; j<nVar; j++) {
            agent_pos[i][j] = LB[j] + unif(gen) * (UB[j] - LB[j]);
        }
    }

    /* Round any integer variable */
    for (int j=0; j<p.nIntVar; j++) {
        int idx_int = p.IntVar[j];
        for (int i=0; i<p.nPop; i++) {
            agent_pos[i][idx_int] = round(agent_pos[i][idx_int]);
        }
    }

    /* Initial cost of each agent */
    if (p.normalize) {
        for (int i=0; i<p.nPop; i++) {
            for (int j=0; j<nVar; j++) {
                agent_pos_orig[i][j] = LB_orig[j] + agent_pos[i][j] *
                                       (UB_orig[j] - LB_orig[j]);
            }
            agent_cost[i] = func(agent_pos_orig[i], i, args);
        }
    }
    else {
        for (int i=0; i<p.nPop; i++) {
            agent_cost[i] = func(agent_pos[i], i, args);
        }
    }

    /* Initial (overall) best position and cost */
    int i_min = idx_min(agent_cost, p.nPop);
    double best_cost = agent_cost[i_min];
    copy_Array(best_pos, nVar, agent_pos[i_min]);

    /* Main loop (T = const) */
    for (int epoch=0; epoch<p.epochs; epoch++) {

        /* Sub-loop (search the neighboroud of an agent) */
        for (int move=0; move<p.nMove; move++) {

            /* Randomly create agent's neighbours */
            for (int i=0; i<p.nPop; i++) {
                for (int j=0; j<nVar; j++) {
                    if (unif(gen) <= p.prob) {
                        neigh_pos[i][j] = agent_pos[i][j] +
                                          norm(gen) * sigma[j];
                    }
                    else {
                        neigh_pos[i][j] = agent_pos[i][j];
                    }
                }
            }

            /* Round any integer variable */
            for (int j=0; j<p.nIntVar; j++) {
                int idx_int = p.IntVar[j];
                for (int i=0; i<p.nPop; i++) {
                    neigh_pos[i][idx_int] = round(neigh_pos[i][idx_int]);
                }
            }

            /* Impose position boundaries */
            for (int i=0; i<p.nPop; i++) {
                for (int j=0; j<nVar; j++) {
                    neigh_pos[i][j] = max(neigh_pos[i][j], LB[j]);
                    neigh_pos[i][j] = min(neigh_pos[i][j], UB[j]);
                }
            }

            /* Impose position boundaries on any integer variable */
            for (int j=0; j<p.nIntVar; j++) {
                int idx_int = p.IntVar[j];
                for (int i=0; i<p.nPop; i++) {
                    neigh_pos[i][idx_int] = max(neigh_pos[i][idx_int],
                                                ceil(LB[idx_int]));
                    neigh_pos[i][idx_int] = min(neigh_pos[i][idx_int],
                                                floor(UB[idx_int]));
                }
            }

            /* Evaluate the cost of each agent's neighbour */
            if (p.normalize) {
                for (int i=0; i<p.nPop; i++) {
                    for (int j=0; j<nVar; j++) {
                        neigh_pos_orig[i][j] = LB_orig[j] + neigh_pos[i][j] *
                                               (UB_orig[j] - LB_orig[j]);
                    }
                    neigh_cost[i] = func(neigh_pos_orig[i], i, args);
                }
            }
            else {
                for (int i=0; i<p.nPop; i++) {
                    neigh_cost[i] = func(neigh_pos[i], i, args);
                }
            }

            /* Decide if each agent will change its state */
            for (int i=0; i<p.nPop; i++) {

                /* Swap states if the neighbour state is better ... */
                if (neigh_cost[i] <= agent_cost[i]) {
                    agent_cost[i] = neigh_cost[i];
                    copy_Array(agent_pos[i], nVar, neigh_pos[i]);
                }

                /* ... or decide probabilistically */
                else {

                    /* Acceptance probability */
                    double delta = (neigh_cost[i] - agent_cost[i]) /
                                    agent_cost[i];
                    double prob_swap = exp(-delta / T);

                    /* Randomly swap states */
                    if (unif(gen) <= prob_swap) {
                        agent_cost[i] = neigh_cost[i];
                        copy_Array(agent_pos[i], nVar, neigh_pos[i]);
                    }
                }  

                /* Update the (overall) best position and cost */
                i_min = idx_min(agent_cost, p.nPop);
                best_cost = agent_cost[i_min];
                copy_Array(best_pos, nVar, agent_pos[i_min]);
            }
        }

        /* Save the best cost for this epoch */
        F[epoch] = best_cost;

        /* Update the (cooling) temperature */
        T = p.alphaT * T;
 
        /* Update the standard deviation */
        for (int j=0; j<nVar; j++) {
            sigma[j] = p.alphaS * sigma[j];
        }
    }

    /* De-normalize */
    if (p.normalize) {
        for (int j=0; j<nVar; j++) {
            best_pos[j] = LB_orig[j] + best_pos[j] * (UB_orig[j] - LB_orig[j]);
            sigma[j] = sigma[j] * (UB_orig[j] - LB_orig[j]);
        }
    }

    // De-allocate the dynamically allocated arrays and matrices
    del_Array(F);
    del_Array(sigma);
    del_Array(agent_pos, p.nPop);
    del_Array(neigh_pos, p.nPop);
    del_Array(agent_cost);
    del_Array(neigh_cost);
    if (p.normalize) {
        del_Array(LB_orig);
        del_Array(UB_orig);
        del_Array(agent_pos_orig, p.nPop);
        del_Array(neigh_pos_orig, p.nPop);
    }

    return best_pos;
}
