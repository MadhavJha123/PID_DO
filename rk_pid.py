import numpy as np
import random

# Parameters for genetic algorithm and simulation
population_size = 30  # Number of individuals in the population
num_generations = 50  # Number of generations
mutation_probability = 0.05  # Probability of mutation
num_intervals = 12  # Number of control intervals
h = 0.01  # Step size for numerical integration
c = 100  # Cost penalty factor
u_min = 0  # Minimum control input value
eta1 = 0.6  # Learning rate for PID update
u_max = 12  # Maximum control input value

# PID control coefficients
Kp = 1.2  # Proportional gain
Ki = 1.5  # Integral gain
Kd = 0.8  # Derivative gain

# Function to calculate PID-based control update
def delta_u(etk, prev_etk, prev_prev_etk, r2, r3, r4):
    return (((Kp * r2 * (etk - prev_etk)) + (Ki * r3 * etk) + (Kd * r4 * (etk - 2 * prev_etk + prev_prev_etk))))

# Functions defining system dynamics
def g1(x2, x3):
    return (0.408 / (1 + x3 / 16)) * (x2 / (0.22 + x2))

def g2(x2, x3):
    return (1 / (1 + x3 / 71.5)) * (x2 / (0.44 + x2))

# Initialize random control parameters for PID
r2, r3, r4 = np.random.rand(3, population_size, num_intervals)

# Function representing system equations
def f(t, x1, x2, x3, x4, u_val):
    g1_val = g1(x2, x3)
    g2_val = g2(x2, x3)
    dx1_dt = g1_val * x1 - u_val * (x1 / x4)
    dx2_dt = -10 * g1_val * x1 + u_val * (150 - x2) / x4
    dx3_dt = g2_val * x1 - u_val * (x3 / x4)
    dx4_dt = u_val
    return np.array([dx1_dt, dx2_dt, dx3_dt, dx4_dt])

# Runge-Kutta 4th Order Method for solving ODEs
def rk4_step(t, x, h, u_val):
    k1 = f(t, *x, u_val)
    k2 = f(t + 0.5 * h, *(x + 0.5 * h * k1), u_val)
    k3 = f(t + 0.5 * h, *(x + 0.5 * h * k2), u_val)
    k4 = f(t + h, *(x + h * k3), u_val)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Fitness function evaluating the system performance
def calculate_fitness(x3, x4):
    return (x3 * x4) - c * abs(200 - x4)

# Function to simulate the system over multiple time intervals
def simulate(u_values):
    x = np.array([1, 150, 0, 10])  # Initial state variables x1, x2, x3, x4
    t = 0
    for i in range(num_intervals):
        u_val = u_values[i]
        for _ in range(int(5400 / num_intervals)):  # Run RK4 method over each interval
            x = rk4_step(t, x, h, u_val)
            t += h
    return x, u_val

# Function to create an individual (control sequence)
def create_individual():
    return [0.6 * (i + 1) for i in range(num_intervals)]

# Function to create an initial population
def create_population(size):
    return [create_individual() for _ in range(size)]

# Mutation function to introduce small changes in individuals
def mutate(individual):
    if random.random() < mutation_probability:
        if random.random() < 0.5:
            for i in range(len(individual)):
                individual[i] += 0.05 * abs(12 - individual[i])  # Small increase towards upper bound
        else:
            for i in range(len(individual)):
                individual[i] -= 0.05 * abs(individual[i] - 0)  # Small decrease towards lower bound
    return individual

# Function to get best individuals based on fitness
def get_best_individuals(population, fitnesses):
    sorted_population = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    return sorted_population

# Initialize population
population = create_population(population_size)
fitnesses = []  # List to store fitness values

# PID-related variables
etk = 0
prev_etk = 0
prev_prev_etk = 0
prev_x_star = 0
prev_prev_x_star = 0
x_star = 0

# Evolutionary loop over generations
for generation in range(num_generations):
    best_score = 0
    best_x3 = 0
    best_x4 = 0
    best_u = 0
    
    # Evaluate fitness for each individual
    fitnesses = []
    for i, individual in enumerate(population):
        final_state, u_val = simulate(individual)
        x3, x4 = final_state[2], final_state[3]
        fitness = calculate_fitness(x3, x4)
        fitnesses.append(fitness)
        if fitness > best_score:
            best_score = fitness
            best_x3 = x3
            best_x4 = x4
            x_star = individual
    
    # Compute error for PID control update
    etk = [np.array(x_star) - np.array(individual) for individual in population]
    if generation == 0:
        prev_etk = np.copy(etk)
        prev_prev_etk = np.copy(etk)
    else:
        prev_etk = np.copy(etk)
        prev_prev_etk = np.copy(prev_etk)
    
    # Apply PID-based adjustment
    delta_u_t = delta_u(etk, prev_etk, prev_prev_etk, r2, r3, r4)
    prev_population = population
    prev_prev_x_star = prev_x_star
    prev_x_star = x_star.copy()

    # Update population with new control values
    population += eta1 * delta_u_t
    population = [mutate(individual) for individual in population]
    population[0] = x_star  # Retain best individual
    prev_prev_population = prev_population
    
    # Print generation results
    print(f"x3 * x4 :{best_x3 * best_x4:.3f}, x4: {best_x4:.3f}, x3: {best_x3:.3f}, fitness:{best_score:.2f}")
