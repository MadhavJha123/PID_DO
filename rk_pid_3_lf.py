import numpy as np
import random
from scipy.special import gamma

# Parameters
population_size = 50
num_generations = 100
gmax = num_generations
num_intervals = 30  
h = 0.01  # Time step
tf = 1.0  # Final time
u_min, u_max = 0, 5

# PID control coefficients
Kp = 1.5
Ki = 1.8
Kd = 0.8
eta1 = 0.9
eta2 = 0.1
beta = 1.5

# Random control parameters
r2, r3, r4 = np.random.rand(3, population_size, num_intervals)

def delta_u(etk, prev_etk, prev_prev_etk, r2, r3, r4):
    return (((Kp * r2 * (etk - prev_etk)) + 
             (Ki * r3 * etk) + 
             (Kd * r4 * (etk - 2 * prev_etk + prev_prev_etk))))

def control_function(u):
    """Clip control values within [u_min, u_max]"""
    return np.clip(u, u_min, u_max)

def f(t, x1, x2, u_val):
    """System dynamics"""
    dx1_dt = - (u_val + 0.5 * u_val**2) * x1
    dx2_dt = u_val * x1
    return np.array([dx1_dt, dx2_dt])

def rk4_step(t, x, h, u_val):
    """4th order Runge-Kutta method"""
    k1 = f(t, *x, u_val)
    k2 = f(t + 0.5 * h, *(x + 0.5 * h * k1), u_val)
    k3 = f(t + 0.5 * h, *(x + 0.5 * h * k2), u_val)
    k4 = f(t + h, *(x + h * k3), u_val)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def levy_flight():
    u = np.random.standard_normal((1, num_intervals))
    v = np.random.standard_normal((1, num_intervals))
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    L = u * sigma / (np.abs(v) ** (1 / beta))
    
    # print(gamma())
    return L

def simulate(u_values):
    """Simulate the system with given control inputs"""
    x = np.array([1.0, 0.0])  # Initial conditions [x1(0), x2(0)]
    t = 0
    x_history = [x.copy()]
    t_history = [t]
    
    for i in range(num_intervals):
        u_val = control_function(u_values[i])  # Constrain control input
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
    return -x2_tf   # Negative because we want to minimize x2(tf)

def create_individual():
    """Create a random control sequence"""
    return np.random.uniform(u_min, u_max, num_intervals)

def create_population(size):
    return [create_individual() for _ in range(size)]


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

    L_values = np.array([levy_flight() for _ in range(population_size)]).reshape(population_size, num_intervals)
    r5_values = np.random.rand(population_size, num_intervals)
    lambda_val = (np.log(gmax - generation + 2) / np.log(gmax)) ** 2
    o_t = (np.cos(1 - generation / gmax) + lambda_val * r5_values * L_values) * etk
    
    # Update population using PID control
    delta_u_t = delta_u(etk, prev_etk, prev_prev_etk, r2, r3, r4)
    population += eta1 * delta_u_t + eta2 * o_t
    
    # Clip population to maintain control values within bounds
    population = [control_function(individual) for individual in population]
    
    # Preserve best solution
    population[0] = x_star
    
    print(f"Generation {generation + 1}: Best fitness = {best_fitness:.6f}, Final x2 = {-best_fitness:.6f}")

# Get final results
final_state, _, best_trajectory, best_times = simulate(best_solution)
print("\nOptimization Results:")
print(f"Final x2(tf) = {final_state[1]:.6f}")
print(f"Final x1(tf) = {final_state[0]:.6f}")
