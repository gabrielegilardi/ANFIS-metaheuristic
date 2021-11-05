/*
Multivariate Regression and Classification Using an Adaptive Neuro-Fuzzy
Inference System (Takagi-Sugeno) and Particle Swarm Optimization.

Copyright (c) 2021 Gabriele Gilardi


=== Arguments ===
func            Function to minimize
LB              Lower boundaries of the search space
UB              Upper boundaries of the search space
p               Structure with the solver parameters
args            Structure with the ANFIS data
gen             Random number generator

=== Solver parameters ===
nPop            Number of agents (population)
epochs          Number of iterations
K               Average size of each agent's group of informants
phi             Coefficient to calculate the two confidence coefficients
vel_fact        Velocity factor to calculate the maximum and the minimum
                allowed velocities
conf_type       Confinement type (on the velocities)
normalize       Specifies if the search space should be normalized
IntVar          List of indexes specifying which variable should be treated
                as integer

=== Dimensions ===
(nVar, 1)       LB, UB, swarm_best_pos
(nPop, nVar)    LBe, UBe, LBe_orig, UBe_orig, vel_max, vel_min, agent_pos, u,
                agent_vel, agent_best_pos, group_best_pos, Gr, x_sphere, out,
                vel_conf, rn
(nPop, nPop)    informants, informants_cost
(nPop, 1)       agent_cost, agent_best_cost, p_equal_g, r_max, norm, r
(0-nVar)        IntVar
*/


/* Headers */
#include <limits>
#include "anfis.hpp"
#include "utils.hpp"

/* Structure to pass the parameters to the PSO solver */
struct Parameters {
    int nPop;
    int epochs;
    int K;
    double phi;
    double vel_fact;
    string conf_type;
    bool normalize;
    ArrayXi IntVar;
};

/* Structure to pass the data to the ANFIS */
struct Arguments {
    ArrayXXd X;                 // Training dataset inputs/features
    ArrayXXd Y;                 // Training dataset labels/outputs (regression)
    ArrayXi Yc;                 // Training dataset classes (classification)
    AnfisType* agents;          // Array of ANFIS agents
};


/* Local function prototypes*/
ArrayXXi create_group(int nPop, double p_informant, mt19937_64& gen);
ArrayXXd group_best(ArrayXXi informants, ArrayXXd agent_best_pos,
                    ArrayXd agent_best_cost, ArrayXd& p_equal_g);
ArrayXXd hypersphere_point(ArrayXXd Gr, ArrayXXd agent_pos, mt19937_64& gen);
ArrayXXd hyperbolic_conf(ArrayXXi out, ArrayXXd agent_pos, ArrayXXd agent_vel,
                         ArrayXXd UBe, ArrayXXd LBe);
ArrayXXd random_back_conf(ArrayXXi out, ArrayXXd agent_vel, mt19937_64& gen);
ArrayXXd mixed_conf(ArrayXXi out, ArrayXXd agent_pos, ArrayXXd agent_vel,
                    ArrayXXd UBe, ArrayXXd LBe, mt19937_64& gen);


/* Minimize a function using particle swarm optimization */
ArrayXd pso(ArrayXd (*func)(ArrayXXd, Arguments), ArrayXd LB, ArrayXd UB,
            Parameters p, Arguments args, mt19937_64& gen)
{
    /* Random probability distributions */
    uniform_real_distribution<double> uniform(0.0, 1.0);
    normal_distribution<double> normal(0.0, 1.0);

    /* Parameters and coefficients*/
    int nVar = LB.size();
    int nIntVar = p.IntVar.size();      // Equal to 0 if no integer variables
    double w = 1.0 / (p.phi - 1.0 + sqrt(p.phi * (p.phi - 2.0)));
    double cmax = w * p.phi;

    /* Create boundary matrixes for all agents */
    ArrayXXd LBe = LB.matrix().transpose().replicate(p.nPop, 1);
    ArrayXXd UBe = UB.matrix().transpose().replicate(p.nPop, 1);

    // Max. allowed velocities
    ArrayXXd vel_max = p.vel_fact * (UBe - LBe);
    ArrayXXd vel_min = - vel_max;

    // Probability an agent is an informant
    double base = 1.0 - 1.0 / static_cast<double>(p.nPop);
    double exponent = static_cast<double>(p.K);
    double p_informant = 1.0 - pow(base, exponent);

    /* Normalize search space */
    ArrayXXd LBe_orig, UBe_orig;
    if (p.normalize) {
        LBe_orig = LBe;
        UBe_orig = UBe;
        LBe.setZero(p.nPop, nVar);
        UBe.setOnes(p.nPop, nVar);
    }

    /* Initial position of each agent */
    ArrayXXd agent_pos = LBe + rnd(uniform, gen, p.nPop, nVar) * (UBe - LBe);
    for (int i=0; i<nIntVar; i++) {
        int idx = p.IntVar(i);
        agent_pos.col(idx) = round(agent_pos.col(idx));
    }

    // Initial velocity of each agent (with velocity limits)
    ArrayXXd agent_vel = (LBe - agent_pos) + 
                         rnd(uniform, gen, p.nPop, nVar) * (UBe - LBe);
    agent_vel = (agent_vel.max(vel_min)).min(vel_max);

    /* Initial cost of each agent */
    ArrayXd agent_cost;
    if (p.normalize) {
        ArrayXXd tmp = LBe_orig + agent_pos * (UBe_orig - LBe_orig);
        agent_cost = func(tmp, args);
    }
    else {
        agent_cost = func(agent_pos, args);
    }

    // Initial best position and cost of each agent
    ArrayXXd agent_best_pos = agent_pos;
    ArrayXd agent_best_cost = agent_cost;

    /* Initial (swarm) best position and cost */
    Index r_min, c_min;
    double swarm_best_cost = agent_best_cost.minCoeff(&r_min, &c_min);
    ArrayXd swarm_best_pos = agent_best_pos.row(r_min);

    // Initial best position of each agent
    ArrayXXi informants;
    ArrayXd p_equal_g;
    ArrayXXd group_best_pos;
    if (p.K == 0) {         // ... using the swarm
        group_best_pos = swarm_best_pos.matrix().transpose().replicate(p.nPop, 1);
        p_equal_g.setOnes(p.nPop);
        p_equal_g(r_min) = 0.75;
    }
    else {                  // ... using informants
        informants = create_group(p.nPop, p_informant, gen);
        group_best_pos = group_best(informants, agent_best_pos,
                                    agent_best_cost, p_equal_g);
    }

    // Main loop
    for (int epoch=0; epoch<p.epochs; epoch++) {

        // Determine the updated velocity for each agent
        ArrayXXd Gr = agent_pos + (1.0 / 3.0) * cmax * ((agent_best_pos +
                      group_best_pos - 2.0 * agent_pos).colwise() *
                      p_equal_g);
        ArrayXXd x_sphere = hypersphere_point(Gr, agent_pos, gen);
        agent_vel = w * agent_vel + Gr + x_sphere - agent_pos;

        // Impose velocity limits
        agent_vel = (agent_vel.max(vel_min)).min(vel_max);

        // Temporarly update the position of each agent to check if they are
        // outside the search space (0=out, 1=in)
        ArrayXXd agent_pos_tmp = agent_pos + agent_vel;
        for (int i=0; i<nIntVar; i++) {
            int idx = p.IntVar(i);
            agent_pos_tmp.col(idx) = round(agent_pos_tmp.col(idx));
        }
        ArrayXXi out = (agent_pos_tmp >= LBe).cast<int>() *
                       (agent_pos_tmp <= UBe).cast<int>();

        // Apply velocity confinement and update velocities (all confinement
        // velocities are smaller than the max. allowed velocity)
        if (p.conf_type == "HY") {
            agent_vel = hyperbolic_conf(out, agent_pos, agent_vel, UBe, LBe);
        }
        else if (p.conf_type == "MX") {
            agent_vel = mixed_conf(out, agent_pos, agent_vel, UBe, LBe, gen);
        }
        else {
            agent_vel = random_back_conf(out, agent_vel, gen);
        }

        // Update positions
        agent_pos = agent_pos + agent_vel;
        for (int i=0; i<nIntVar; i++) {
            int idx = p.IntVar(i);
            agent_pos.col(idx) = round(agent_pos.col(idx));
        }

        // Apply position confinement rules to agents outside the search space
        agent_pos = (agent_pos.max(LBe)).min(UBe);
        for (int i=0; i<nIntVar; i++) {
            int idx = p.IntVar(i);
            agent_pos.col(idx) = agent_pos.col(idx).max(ceil(LBe.col(idx)));
            agent_pos.col(idx) = agent_pos.col(idx).min(floor(UBe.col(idx)));
        }

        // Calculate new cost of each agent
        if (p.normalize) {
            ArrayXXd tmp = LBe_orig + agent_pos * (UBe_orig - LBe_orig);
            agent_cost = func(tmp, args);
        }
        else {
            agent_cost = func(agent_pos, args);
        }

        // Update best position and cost of each agent
        for (int i=0; i<p.nPop; i++) {
            if (agent_cost(i) < agent_best_cost(i)) {
                agent_best_pos.row(i) = agent_pos.row(i);
                agent_best_cost(i) = agent_cost(i);
            }
        }
        
        // Update best position and cost of the swarm
        double best_cost = agent_best_cost.minCoeff(&r_min, &c_min);
        if (best_cost < swarm_best_cost) {
            swarm_best_cost = best_cost;
            swarm_best_pos = agent_best_pos.row(r_min);
        }
        // If the best cost of the swarm did not improve when using informants
        // change informant groups
        else {
            if (p.K != 0) {
                informants = create_group(p.nPop, p_informant, gen);
            }
        }

        // Update best position of each agent using the swarm
        if (p.K == 0) {
            group_best_pos = swarm_best_pos.matrix().transpose().replicate(p.nPop, 1);
        }

        // Update best position of each agent using informants
        else {
            group_best_pos = group_best(informants, agent_best_pos,
                                        agent_best_cost, p_equal_g);
        }
    }

    /* De-normalize */
    if (p.normalize) {
        swarm_best_pos = LB + swarm_best_pos * (UB - LB);
    }
   
    return swarm_best_pos;
}

/* Randomly creates the group of informants for each agent */
ArrayXXi create_group(int nPop, double p_informant, mt19937_64& gen)
{
    ArrayXXi informants;
    informants.setZero(nPop, nPop);
    uniform_real_distribution<double> uniform(0.0, 1.0);

    // 1 = part of the group, 0 = not part of the group
    for (int i=0; i<nPop; i++) {
        for (int j=0; j<nPop; j++) {
            if (uniform(gen) < p_informant) {
                informants(i, j) = 1;
            }
        }
        informants(i, i) = 1;
    }

    return informants;
}

/*
Returns the group best position of each agent based on its informants.
*/
ArrayXXd group_best(ArrayXXi informants, ArrayXXd agent_best_pos,
                    ArrayXd agent_best_cost, ArrayXd& p_equal_g)
{
    double inf = numeric_limits<double>::infinity();
    int nPop = agent_best_pos.rows();
    int nVar = agent_best_pos.cols();

    // Determine the cost of each agent in each group (set the value for
    // agents that are not informants of the group to infinity)
    ArrayXXd informants_cost;
    informants_cost.setConstant(nPop, nPop, inf);
    for (int i=0; i<nPop; i++) {
        for (int j=0; j<nPop; j++) {
            if (informants(i, j) == 1) {
                informants_cost(i, j) = agent_best_cost(j);
            }
        }
    }

    // For each agent determine the agent with the best cost in the group
    // and assign its position to it.
    ArrayXXd group_best_pos;
    Index r_min, c_min;
    group_best_pos.setZero(nPop, nVar);
    p_equal_g.setOnes(nPop);
    for (int i=0; i<nPop; i++) {
        double tmp = informants_cost.row(i).minCoeff(&r_min, &c_min);
        group_best_pos.row(i) = agent_best_pos.row(c_min);
        // Build the vector to correct the velocity update for the corner
        // case where the agent is also the group best
        if (c_min == i) {
            p_equal_g(i) = 0.75;
        }
    }

    return group_best_pos;
}

/*
For each agent determines a random point inside the hypersphere (Gr,|Gr-X|),
where Gr is its center, |Gr-X| is its radius, and X is the agent position.
*/
ArrayXXd hypersphere_point(ArrayXXd Gr, ArrayXXd agent_pos, mt19937_64& gen)
{
    int nPop = agent_pos.rows();
    int nVar = agent_pos.cols();

    // Hypersphere radius of each agent
    ArrayXd r_max = (Gr - agent_pos).rowwise().norm();

    // Randomly pick a direction using a normal distribution
    normal_distribution<double> normal(0.0, 1.0);
    ArrayXXd u = rnd(normal, gen, nPop, nVar);
    ArrayXd norm = u.rowwise().norm();

    // Randomly pick a radius using a uniform distribution 
    uniform_real_distribution<double> uniform(0.0, 1.0);
    ArrayXd r = rnd(uniform, gen, nPop, 1) * r_max;

    // Coordinates of the point with direction <u> and at distance <r> from
    // the hypersphere center
    ArrayXXd x_sphere = u.colwise() * (r / norm);

    return x_sphere;
}

/* Applies hyperbolic confinement to the velocities */
ArrayXXd hyperbolic_conf(ArrayXXi out, ArrayXXd agent_pos, ArrayXXd agent_vel,
                         ArrayXXd UBe, ArrayXXd LBe)
{
    // If the agent velocity is > 0
    ArrayXXd vel_plus = agent_vel / (1.0 + (agent_vel / (UBe - agent_pos)).abs());

    // If the agent velocity is <= 0
    ArrayXXd vel_minus = agent_vel / (1.0 + (agent_vel / (agent_pos - LBe)).abs());

    // Confinement velocity for all agents
    ArrayXXd vel_conf_all = (agent_vel > 0.0).select(vel_plus, vel_minus);

    // Confinement velocity for the agents outside the search space
    ArrayXXd vel_conf = (1 - out).cast<double>() * vel_conf_all +
                        out.cast<double>() * agent_vel;

    return vel_conf;
}

/* Applies random-back confinement to the velocities */
ArrayXXd random_back_conf(ArrayXXi out, ArrayXXd agent_vel, mt19937_64& gen)
{
    uniform_real_distribution<double> uniform(0.0, 1.0);

    // Confinement velocity for all agents
    ArrayXXd rn = rnd(uniform, gen, out.rows(), out.cols());
    ArrayXXd vel_conf_all = - rn * agent_vel;

    // Confinement velocity for the agents outside the search space
    ArrayXXd vel_conf = (1 - out).cast<double>() * vel_conf_all +
                        out.cast<double>() * agent_vel;

    return vel_conf;
}

/* Applies (randomly) a mixed-type confinement to the velocities */
ArrayXXd mixed_conf(ArrayXXi out, ArrayXXd agent_pos, ArrayXXd agent_vel,
                    ArrayXXd UBe, ArrayXXd LBe, mt19937_64& gen)
{
    uniform_real_distribution<double> uniform(0.0, 1.0);

    // Hyperbolic confinement velocity
    ArrayXXd vel_conf_HY = hyperbolic_conf(out, agent_pos, agent_vel, UBe, LBe);
    
    // Random back confinement velocity
    ArrayXXd vel_conf_RB = random_back_conf(out, agent_vel, gen);

    // Confinement velocity for all agents
    ArrayXXd rn = rnd(uniform, gen, out.rows(), out.cols());
    ArrayXXd vel_conf_all = (rn >= 0.5).select(vel_conf_RB, vel_conf_HY);

    // Confinement velocity for the agents outside the search space
    ArrayXXd vel_conf = (1 - out).cast<double>() * vel_conf_all +
                        out.cast<double>() * agent_vel;

    return vel_conf;
}
