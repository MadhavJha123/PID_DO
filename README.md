# Genetic Algorithm for Dynamic Optimization of Chemical Processes

## Project Overview

This project implements a genetic algorithm to perform dynamic optimization of a chemical process, specifically targeting Problem 3 (Tubular Reactor Parallel Reaction Problem) from the paper *Dynamic Optimization of Chemical Processes Using Ant Colony Framework* by J. Rajesh et al., published in *Computers & Chemistry* (2001). The system dynamics are simulated using the 4th-order Runge-Kutta (RK4) method, and optimization is achieved through a genetic algorithm with mutation and selection.

## Problem Statement

The problem involves optimizing a tubular reactor where parallel reactions \( A \rightarrow B \) and \( A \rightarrow C \) occur. The goal is to maximize the yield of component B (\( x_2(t_f) \)) by determining the optimal control profile \( u(t) \).

- **System Dynamics**:
  \[
  \frac{dx_1}{dt} = -\left(u + 0.5 u^2\right) x_1
  \]
  \[
  \frac{dx_2}{dt} = u x_1
  \]
  where \( x_1 \) is the dimensionless concentration of A, \( x_2 \) is the dimensionless concentration of B, and \( u(t) \) is the control variable.
- **Initial Conditions**: \( x(0) = [1, 0] \).
- **Constraints**: \( 0 \leq u(t) \leq 5 \), \( t \in [0, t_f] \), \( t_f = 1 \).
- **Objective**: Maximize \( J = x_2(t_f) \).

## Methodology

### Control Profile Representation
- The control variable \( u(t) \) is discretized into a piecewise constant profile over \( [0, 1] \).
- Time is divided into 40 intervals (\( \Delta t = 1 / 40 = 0.025 \)), each with a constant control value.
- Each control vector has 40 values, constrained to \( [0, 5] \).

### System Simulation
- **Numerical Integration**: Uses RK4 with a step size of 0.008.
- **Simulation**:
  - Starts with \( x(0) = [1, 0] \).
  - For each interval, applies the control value and integrates the ODEs using RK4.
  - Computes \( x_2(t_f) \) as the fitness.

### Genetic Algorithm
- **Population**: 800 individuals, each a control vector of 40 values, randomly initialized in \( [0, 5] \).
- **Fitness**: \( x_2(t_f) \), to be maximized.
- **Selection**: Selects the top 50% of individuals based on fitness.
- **Mutation**: Applies with probability 0.05, adding a random perturbation in \( [-0.5, 0.5] \), clipped to \( [0, 5] \).
- **Reproduction**: Creates new individuals by mutating randomly selected parents.
- **Iteration**: Runs for 200 generations, tracking the best solution.

## Comparison with Previous Methods

### Ant Colony Framework (Paper)
- **Control Profile**: Uses a piecewise linear profile with 4 grid points (10 variables total).
- **Method**: Combines global search (crossover, mutation, trail diffusion) and local search, guided by pheromone trails.
- **Performance**:
  - Achieves \( x_2(t_f) = 0.57284 \), close to the global optimum 0.57353 (0.12% error).
  - 100% convergence success.
  - CPU time: 0.07 seconds on a Sun Enterprise 450 server.

### Genetic Algorithm (This Study)
- **Control Profile**: Uses 40 intervals, providing finer control but increasing computational cost.
- **Method**: A simpler GA with selection and mutation, lacking crossover and local search.
- **Performance**:
  - Final \( x_2(t_f) \) depends on the run; may underperform compared to 0.57284 if significantly lower.
  - Likely higher computational cost due to more intervals.
  - Convergence success not reported but may be lower without local search.

## Suggested Improvements

- **Add Crossover**: Introduce crossover to increase population diversity, e.g., combine segments of control vectors from two parents.
- **Implement Local Search**: Refine the top individuals by perturbing their control values slightly to improve convergence.
- **Reduce Intervals**: Decrease the number of intervals to 4 or 5, reducing computational cost while maintaining control flexibility.
- **Increase Diversity**: Increase mutation probability or perturbation range, or use an adaptive mutation rate.
- **Add Elitism**: Preserve the top individuals across generations to ensure the best solutions are not lost.
- **Benchmark Convergence**: Run 25 times to compute average performance, standard deviation, and success rate; measure CPU time.
- **Handle Constraints**: Use a penalty function for additional constraints, ensuring robustness.

## Prerequisites

- **Python 3.6+**
- **NumPy**: Install via `pip install numpy`

## Usage

### File Description
- `optimize_chemical_process.py`: Main script.

### Running the Code
1. Save the script as `optimize_chemical_process.py`.
2. Run:
   ```
   python optimize_chemical_process.py
   ```
3. Outputs:
   - Best fitness per generation.
   - Final \( x_1(t_f) \), \( x_2(t_f) \), and optimal control sequence.

### Parameters
- `population_size`: 800
- `num_generations`: 200
- `mutation_probability`: 0.05
- `num_intervals`: 40
- `h`: 0.008
- `tf`: 1.0
- `control_min, control_max`: 0.0, 5.0

## References
- J. Rajesh et al., *Dynamic Optimization of Chemical Processes Using Ant Colony Framework*, *Computers & Chemistry* 25 (2001) 583--595. DOI: [10.1016/S0097-8485(01)00081-X](https://doi.org/10.1016/S0097-8485(01)00081-X).
