import numpy as np
import random

# Parameters
population_size = 800
num_generations = 200
mutation_probability = 0.05
num_intervals = 40
h = 0.008
tf = 1.0
control_min, control_max = 0.0, 5.0

def f(t, x1, x2, u_val):
    """System dynamics"""
    dx1_dt = -(u_val + 0.5 * u_val**2) * x1
    dx2_dt = u_val * x1
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
    x = np.array([1.0, 0.0])
    t = 0 
    x_history = [x.copy()]
    t_history = [t]
    
    for i in range(num_intervals):
        u_val = np.clip(u_values[i], control_min, control_max)
        steps_per_interval = int((tf / num_intervals) / h)
        
        for _ in range(steps_per_interval):
            x = rk4_step(t, x, h, u_val)
            t += h
            x_history.append(x.copy())
            t_history.append(t)
    return x, np.array(x_history), np.array(t_history)

def calculate_fitness(x2_final):
    """Fitness function based on final value of x2"""
    return float(x2_final)  # Ensure scalar output

def create_individual():
    """Create a random control sequence"""
    return np.random.uniform(control_min, control_max, num_intervals)

def create_population(size):
    return [create_individual() for _ in range(size)]

def mutate(individual):
    """Mutation operator"""
    for i in range(len(individual)):
        if random.random() < mutation_probability:
            individual[i] += np.random.uniform(-0.5, 0.5)
            individual[i] = np.clip(individual[i], control_min, control_max)
    return individual

# Initialize population
population = create_population(population_size)
best_solution = None
best_fitness = float('-inf')
best_trajectory = None
best_times = None

# Main optimization loop
for generation in range(num_generations):
    # Store individuals with their fitness values
    generation_data = []
    
    for individual in population:
        final_state, trajectory, time = simulate(individual)
        fitness = calculate_fitness(final_state[1])
        generation_data.append((fitness, individual, trajectory, time))
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_solution = individual.copy()
            best_trajectory = trajectory
            best_times = time
    
    # Sort by fitness (first element of tuple)
    generation_data.sort(key=lambda x: x[0], reverse=True)
    
    # Select top 50% individuals
    population = [data[1] for data in generation_data[:population_size//2]]
    
    # Generate new population
    new_population = []
    while len(new_population) < population_size:
        parent1, parent2 = random.sample(population, 2)
        child1 = mutate(parent1.copy())
        child2 = mutate(parent2.copy())
        new_population.extend([child1, child2])

    population = new_population[:population_size]
    
    print(f"Generation {generation + 1}: Best fitness = {best_fitness:.6f}")

# Final Results
final_state, best_trajectory, best_times = simulate(best_solution)
print("\nOptimization Results:")
print(f"Final x2(tf) = {final_state[1]:.6f}")
print(f"Final x1(tf) = {final_state[0]:.6f}")
print("Optimal Control Sequence (u):", best_solution)