import numpy as np
import random

# Problem-Specific Parameters
population_size = 600  # As per the given problem
num_generations = 200
mutation_probability = 0.05
num_intervals = 40  # Number of time intervals
h = 0.008  # Time step for RK4
tf = 1.0  # Total time (in hours)
control_min, control_max = 298.0, 398.0  # Control temperature bounds (K)

# Constants for rate equations
A1, E1 = 4e3, 2500  # For k1
A2, E2 = 6.2e5, 5000  # For k2

def reaction_rates(T):
    """Calculate reaction rates k1 and k2 based on temperature T."""
    k1 = A1 * np.exp(-E1 / T)
    k2 = A2 * np.exp(-E2 / T)
    return k1, k2

def f(t, CA, CB, T):
    """System dynamics based on the provided equations."""
    k1, k2 = reaction_rates(T)
    dCA_dt = -k1 * CA**2
    dCB_dt = k1 * CA**2 - k2 * CB
    return np.array([dCA_dt, dCB_dt])

def rk4_step(t, x, h, T):
    """4th order Runge-Kutta method for solving ODEs."""
    k1 = f(t, *x, T)
    k2 = f(t + 0.5 * h, *(x + 0.5 * h * k1), T)
    k3 = f(t + 0.5 * h, *(x + 0.5 * h * k2), T)
    k4 = f(t + h, *(x + h * k3), T)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def simulate(T_values):
    """Simulate the system with the given temperature control inputs."""
    x = np.array([1.0, 0.0])  # Initial conditions: CA(0) = 1, CB(0) = 0
    t = 0  # Initial time
    x_history = [x.copy()]
    t_history = [t]

    for i in range(num_intervals):
        T = np.clip(T_values[i], control_min, control_max)
        steps_per_interval = int((tf / num_intervals) / h)

        for _ in range(steps_per_interval):
            x = rk4_step(t, x, h, T)
            t += h
            x_history.append(x.copy())
            t_history.append(t)
    return x, np.array(x_history), np.array(t_history)

def calculate_fitness(CB_final):
    """Fitness function based on final concentration of CB."""
    return float(CB_final)  # Ensure scalar output

def create_individual():
    """Create a random control sequence."""
    return np.random.uniform(control_min, control_max, num_intervals)

def create_population(size):
    return [create_individual() for _ in range(size)]

def mutate(individual):
    """Mutation operator."""
    for i in range(len(individual)):
        if random.random() < mutation_probability:
            individual[i] += np.random.uniform(-5.0, 5.0)  # Small random change in temperature
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
        fitness = calculate_fitness(final_state[1])  # CB(tf)
        generation_data.append((fitness, individual, trajectory, time))

        if fitness > best_fitness:
            best_fitness = fitness
            best_solution = individual.copy()
            best_trajectory = trajectory
            best_times = time

    # Sort by fitness (first element of tuple)
    generation_data.sort(key=lambda x: x[0], reverse=True)

    # Select top 50% individuals
    population = [data[1] for data in generation_data[:population_size // 2]]

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
print(f"Final CB(tf) = {final_state[1]:.6f}")
print(f"Final CA(tf) = {final_state[0]:.6f}")
print("Optimal Temperature Control Sequence (T):", best_solution)
