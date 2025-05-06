import numpy as np
import random

# Parameters
population_size = 60
num_generations = 100
mutation_probability = 0.05
num_intervals = 8
h = 0.01  # Time step
tf = 1.0  # Final time

# PID control coefficients
Kp = 1.5
Ki = 1.8
Kd = 0.8
eta1 = 0.6

# Random control parameters
r2, r3, r4 = np.random.rand(3, population_size, num_intervals)

def delta_u(etk, prev_etk, prev_prev_etk, r2, r3, r4):
    return (((Kp * r2 * (etk - prev_etk)) + 
             (Ki * r3 * etk) +
             (Kd * r4 * (etk - 2 * prev_etk + prev_prev_etk))))

def f(t, x1, x2, u_val):
    """System dynamics"""
    dx1_dt = u_val
    dx2_dt = x1**2 + u_val**2
    return np.array([dx1_dt, dx2_dt])

def rk4_step(t, x, h, u_val):
    """4th order Runge-Kutta method"""
    k1 = f(t, *x, u_val)
    k2 = f(t + 0.5 * h, *(x + 0.5 * h * k1), u_val)
    k3 = f(t + 0.5 * h, *(x + 0.5 * h * k2), u_val)
    k4 = f(t + h, *(x + h * k3), u_val)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def simulate(u_values):
    """Simulate the system with given control inputs"""
    x = np.array([1.0, 0.0])  # Initial conditions [x1(0), x2(0)]
    t = 0
    x_history = [x.copy()]
    t_history = [t]
    
    for i in range(num_intervals):
        u_val = u_values[i]
        steps_per_interval = int((tf / num_intervals) / h)
        
        for _ in range(steps_per_interval):
            x = rk4_step(t, x, h, u_val)
            t += h
            x_history.append(x.copy())
            t_history.append(t)
    return x, u_val, np.array(x_history), np.array(t_history)

def calculate_fitness(final_state):
    """Fitness function with penalty for x1(tf) not equal to 1"""
    x1_tf, x2_tf = final_state
    penalty = 10 * abs(x1_tf - 1)  # Large penalty for constraint violation
    return -x2_tf - penalty  # Negative because we want to minimize x2(tf)

def create_individual():
    """Create a random control sequence"""
    return np.random.uniform(-10, 10, num_intervals)

def create_population(size):
    return [create_individual() for _ in range(size)]

def mutate(individual):
    """Mutation operator"""
    for i in range(len(individual)):
        if random.random() < mutation_probability:
            if random.random() < 0.5:
                individual[i] += np.random.normal(0, 1)
            else:
                individual[i] -= np.random.normal(0, 1)
    return individual

# Initialize population
population = create_population(population_size)
best_solution = None
best_fitness = float('-inf')
best_trajectory = None
best_times = None

# Main optimization loop
etk = 0
prev_etk = 0
prev_prev_etk = 0
x_star = np.zeros(num_intervals)

for generation in range(num_generations):
    # Evaluate fitness for the population
    fitnesses = []
    trajectories = []
    times = []
    
    for individual in population:
        final_state, _, trajectory, time = simulate(individual)
        fitness = calculate_fitness(final_state)
        fitnesses.append(fitness)
        trajectories.append(trajectory)
        times.append(time)
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_solution = individual.copy()
            best_trajectory = trajectory
            best_times = time
    
    # Update PID error terms
    best_idx = np.argmax(fitnesses)
    x_star = population[best_idx].copy()
    etk = [np.array(x_star) - np.array(individual) for individual in population]
    
    if generation == 0:
        prev_etk = np.copy(etk)
        prev_prev_etk = np.copy(etk)
    else:
        prev_prev_etk = np.copy(prev_etk)
        prev_etk = np.copy(etk)
    
    # Update population using PID control
    delta_u_t = delta_u(etk, prev_etk, prev_prev_etk, r2, r3, r4)
    population = [p + eta1 * du for p, du in zip(population, delta_u_t)]
    
    # Apply mutation
    population = [mutate(individual) for individual in population]
    
    # Preserve best solution
    population[0] = x_star
    
    print(f"Generation {generation + 1}: Best fitness = {best_fitness:.6f}, Final x2(tf) = {-best_fitness:.6f}")

# Get final results
final_state, _, best_trajectory, best_times = simulate(best_solution)
print("\nOptimization Results:")
print(f"Final x2(tf) = {final_state[1]:.6f}")
print(f"Final x1(tf) = {final_state[0]:.6f}")