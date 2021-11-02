/*
Multivariate Regression and Classification Using an Adaptive Neuro-Fuzzy
Inference System (Takagi-Sugeno) and Simulated Annealing Optimization.

Copyright (c) 2021 Gabriele Gilardi


=== Arguments ===
func            Function to minimize
LB              Lower boundaries of the search space
UB              Upper boundaries of the search space
p               Structure with the solver parameters
args            Structure with the ANFIS data

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
seed            Seed for the random number generator

=== Dimensions ===
(nVar, 1)       LB, UB, best_pos
(nPop, nVar)    LBe, UBe, LBe_orig, UBe_orig, sigma, agent_pos, neigh_pos, 
                agent_pos_orig, neigh_pos_orig, rn, flips   
(nPop, 1)       agent_cost, neigh_cost
(epochs)        F
(0-nVar)        IntVar
*/


/* Headers */
#include "anfis.hpp"
#include "utils.hpp"

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
    ArrayXi IntVar;
    int seed;
};

/* Structure to pass the data to the ANFIS */
struct Arguments {
    ArrayXXd X;                 // Training dataset inputs/features
    ArrayXXd Y;                 // Training dataset labels/outputs (regression)
    ArrayXi Yc;                 // Training dataset classes (classification)
    AnfisType* agents;          // Array of ANFIS agents
};


/* Minimize a function using simulated annealing */
ArrayXd sa(ArrayXd (*func)(ArrayXXd, Arguments), ArrayXd LB, ArrayXd UB,
           Parameters p, Arguments args)
{
    /* Random generator and probability distributions */
    mt19937_64 generator(p.seed);
    uniform_real_distribution<double> unif(0.0, 1.0);
    normal_distribution<double> norm(0.0, 1.0);

    int nVar = LB.size();
    int nIntVar = p.IntVar.size();

    /* Create boundary matrixes for all agents */
    ArrayXXd LBe = LB.matrix().transpose().replicate(p.nPop, 1);
    ArrayXXd UBe = UB.matrix().transpose().replicate(p.nPop, 1);

    /* Normalize search space */
    ArrayXXd LBe_orig, UBe_orig;
    if (p.normalize) {
        LBe_orig = LBe;
        UBe_orig = UBe;
        LBe.setZero(p.nPop, nVar);
        UBe.setOnes(p.nPop, nVar);
    }

    // Temperature and initial standard deviation
    double T = p.T0;
    ArrayXXd sigma = p.sigma0 * (UBe - LBe);

    /* Initial position of each agent */
    ArrayXXd rn = rnd(unif, generator, p.nPop, nVar);
    ArrayXXd agent_pos = LBe + rn * (UBe - LBe);

    /* Correct for any integer variable */
    for (int j=0; j<nIntVar; j++) {
        int idx = p.IntVar(j);
        agent_pos.col(idx) = round(agent_pos.col(idx));
    }

    /* Initial cost of each agent */
    ArrayXd agent_cost;
    if (p.normalize) {
        ArrayXXd tmp = LBe_orig + agent_pos * (UBe_orig - LBe_orig);
        agent_cost = func(tmp, args);
    }
    else {
        agent_cost = func(agent_pos, args);
    }

    /* Initial (overall) best position/cost */
    Index r_min, c_min;
    double best_cost = agent_cost.minCoeff(&r_min, &c_min);
    ArrayXd best_pos = agent_pos.row(r_min);

    /* Main loop (T = const) */
    ArrayXd F;
    F.setZero(p.epochs);
    for (int epoch=0; epoch<p.epochs; epoch++) {

        /* Sub-loop (search the neighboroud of a state) */
        for (int move=0; move<p.nMove; move++) {

            /* Randomly decide in which dimension to search */
            rn = rnd(unif, generator, p.nPop, nVar);
            ArrayXXd flips = (rn <= p.prob).cast<double>();

            /* Create each agent's neighbours */
            rn = rnd(norm, generator, p.nPop, nVar);
            ArrayXXd neigh_pos = agent_pos + flips * rn * sigma;

            /* Correct for any integer variable */
            for (int j=0; j<nIntVar; j++) {
                int idx = p.IntVar(j);
                neigh_pos.col(idx) = round(neigh_pos.col(idx));
            }

            /* Impose position boundaries */
            neigh_pos = neigh_pos.max(LBe);
            neigh_pos = neigh_pos.min(UBe);

            for (int j=0; j<nIntVar; j++) {
                int idx = p.IntVar(j);
                neigh_pos.col(idx) = neigh_pos.col(idx).max(ceil(LBe.col(idx)));
                neigh_pos.col(idx) = neigh_pos.col(idx).min(floor(UBe.col(idx)));
            }

            /* Evaluate the cost of each agent's neighbour */
            ArrayXd neigh_cost;
            if (p.normalize) {
                ArrayXXd tmp = LBe_orig + neigh_pos * (UBe_orig - LBe_orig);
                neigh_cost = func(tmp, args);
            }
            else {
                neigh_cost = func(neigh_pos, args);
            }

            /* Decide if each agent will change its state */
            for (int i=0; i<p.nPop; i++) {

                /* Swap states if the neighbour state is better ... */
                if (neigh_cost(i) <= agent_cost(i)) {
                    agent_cost(i) = neigh_cost(i);
                    agent_pos.row(i) = neigh_pos.row(i);
                }

                /* ... or decide probabilistically */
                else {

                    /* Acceptance probability */
                    double d = (neigh_cost(i) - agent_cost(i)) / agent_cost(i);
                    double prob_swap = exp(-d / T);

                    /* Randomly swap states */
                    if (unif(generator) <= prob_swap) {
                        agent_cost(i) = neigh_cost(i);
                        agent_pos.row(i) = neigh_pos.row(i);
                    }
                }  

                /* Update the (overall) best position/cost */
                best_cost = agent_cost.minCoeff(&r_min, &c_min);
                best_pos = agent_pos.row(r_min);
            }
        }

        /* Save the best cost for this epoch */
        F(epoch) = best_cost;

        /* Cooling scheduling */
        T = p.alphaT * T;
 
        /* Random neighboroud search schedule */
        sigma = p.alphaS * sigma;
    }

    /* De-normalize */
    if (p.normalize) {
        best_pos = LB + best_pos * (UB - LB);
        sigma = sigma * (UBe_orig - LBe_orig);
    }

    return best_pos;
}
