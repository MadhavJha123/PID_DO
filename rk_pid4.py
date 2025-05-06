import numpy as np
import random

# Parameters
population_size = 50
num_generations = 100
mutation_probability = 0.05
num_intervals = 15
h = 0.1  # Step size
zf = 12.0  # Final z

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

def f(z, x, u_val):
    """System dynamics for the reaction"""
    xA, xB = x
    dxA_dz = u_val * (10 * xB - xA)
    dxB_dz = -u_val * (10 * xB - xA) - (1 - u_val) * xB
    return np.array([dxA_dz, dxB_dz])

def rk4_step(z, x, h, u_val):
    """4th order Runge-Kutta method"""
    k1 = f(z, x, u_val)
    k2 = f(z + 0.5 * h, x + 0.5 * h * k1, u_val)
    k3 = f(z + 0.5 * h, x + 0.5 * h * k2, u_val)
    k4 = f(z + h, x + h * k3, u_val)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def simulate(u_values):
    """Simulate the system with given control inputs"""
    x = np.array([1.0, 0.0])  # Initial conditions [xA(0), xB(0)]
    z = 0
    x_history = [x.copy()]
    z_history = [z]
    
    for i in range(num_intervals):
        u_val = np.clip(u_values[i], 0, 1)  # Ensure u is within [0, 1]
        steps_per_interval = int((zf / num_intervals) / h)
        
        for _ in range(steps_per_interval):
            x = rk4_step(z, x, h, u_val)
            z += h
            x_history.append(x.copy())
            z_history.append(z)
    return x, u_val, np.array(x_history), np.array(z_history)

def calculate_fitness(x_final):
    """Fitness function based on the objective"""
    xA_final, xB_final = x_final
    return 1 - xA_final - xB_final  # Objective to maximize

def create_individual():
    """Create a random control sequence"""
    return np.random.uniform(0, 1, num_intervals)

def create_population(size):
    return [create_individual() for _ in range(size)]

def mutate(individual):
    """Mutation operator"""
    for i in range(len(individual)):
        if random.random() < mutation_probability:
            individual[i] += np.random.normal(0, 0.1)
            individual[i] = np.clip(individual[i], 0, 1)  # Keep within bounds
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
    
    print(f"Generation {generation + 1}: Best fitness = {best_fitness:.6f}, Final Objective = {best_fitness:.6f}")

# Get final results
final_state, _, best_trajectory, best_times = simulate(best_solution)
print("\nOptimization Results:")
print(f"Final Objective (J) = {best_fitness:.6f}")
print(f"Final xA(zf) = {final_state[0]:.6f}")
print(f"Final xB(zf) = {final_state[1]:.6f}")
