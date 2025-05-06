import random
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import mutual_info_classif

# Load and prepare dataset (keeping your existing setup)
df = pd.read_csv('Heart.csv')
warnings.filterwarnings("ignore", category=FutureWarning)
X = df.drop(columns='Sex')
y = df['Sex']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Population parameters
population_size = 30
num_features = X_train.shape[1]
threshold = 0.3
eta1 = 1
eta2 = 0.8
iterations = 100
gmax = iterations
c = 0.4

# PID control coefficients
Kp = 1.2
Ki = 1.5
Kd = 0.8

# New parameters for dynamic optimization
crossover_rate = 0.8
mutation_rate = 0.2
exploration_factor = 1.0
min_exploration = 0.2
stagnation_threshold = 0.001
stagnation_counter = 0

population = np.random.rand(population_size, num_features)

def convert_to_binary(pop):
    return np.where(pop >= threshold, 1, 0)

def evaluate_fitness(individual):
    selected_features = np.where(individual == 1)[0]
    if len(selected_features) == 0:
        return 0

    X_subset = X.iloc[:, selected_features]
    rf = RandomForestClassifier(random_state=42)
    scores = cross_val_score(rf, X_subset, y, cv=10)
    fitness = scores.mean() - c * (len(selected_features) / X.shape[1])
    return fitness

def dynamic_crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2
    return parent1.copy(), parent2.copy()

def adaptive_mutation(individual, mutation_rate, exploration_factor, generation, max_generations):
    # Increase mutation rate when stuck or in early generations
    adaptive_rate = mutation_rate * (1 + exploration_factor * (1 - generation / max_generations))
    
    for i in range(len(individual)):
        if random.random() < adaptive_rate:
            # Apply larger mutations early, smaller ones later
            mutation_strength = exploration_factor * (1 - generation / max_generations)
            individual[i] += np.random.normal(0, mutation_strength)
            individual[i] = np.clip(individual[i], 0, 1)
    
    return individual

def check_stagnation(current_best, prev_best, threshold):
    return abs(current_best - prev_best) < threshold

def delta_u(etk, prev_etk, prev_prev_etk, r2, r3, r4, exploration_factor):
    # Add exploration term when control signals become similar
    base_delta = ((Kp * r2 * (etk - prev_etk)) + 
                  (Ki * r3 * etk) + 
                  (Kd * r4 * (etk - 2 * prev_etk + prev_prev_etk)))
    
    # Check if control signals are becoming too similar
    signal_similarity = np.mean(np.abs(etk - prev_etk))
    
    if signal_similarity < stagnation_threshold:
        # Add exploration noise proportional to exploration factor
        exploration_noise = np.random.normal(0, 0.1 * exploration_factor, base_delta.shape)
        return base_delta + exploration_noise
    
    return base_delta

def tournament_selection(population, fitness_values, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        tournament_idx = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_idx]
        winner_idx = tournament_idx[np.argmax(tournament_fitness)]
        selected.append(population[winner_idx].copy())
    return np.array(selected)

# Main optimization loop
prev_prev_population = population.copy()
r2, r3, r4 = np.random.rand(3, population_size, num_features)
prev_prev_x_star = None
prev_x_star = None
x_star = None
max_fit = 0
prev_best_fitness = float('-inf')

for t in range(1, iterations + 1):
    print(f"Iteration: {t}")
    
    # Convert population to binary and evaluate fitness
    binary_population = convert_to_binary(population)
    fitness_values = np.array([evaluate_fitness(individual) for individual in binary_population])
    
    # Update best solution
    best_idx = np.argmax(fitness_values)
    current_best_fitness = evaluate_fitness(binary_population[best_idx])
    
    if t == 1:
        x_star = population[best_idx].copy()
        max_fit = current_best_fitness
    elif current_best_fitness > max_fit:
        x_star = population[best_idx].copy()
        max_fit = current_best_fitness
    
    # Check for stagnation
    if check_stagnation(max_fit, prev_best_fitness, stagnation_threshold):
        stagnation_counter += 1
        # Increase exploration when stuck
        exploration_factor = min(1.5, exploration_factor * 1.1)
    else:
        stagnation_counter = 0
        # Gradually reduce exploration when improving
        exploration_factor = max(min_exploration, exploration_factor * 0.95)
    
    # Calculate errors
    etk = x_star - population
    if t == 1:
        prev_etk = etk.copy()
        prev_prev_etk = etk.copy()
    else:
        prev_etk = prev_x_star - prev_population
        if t > 2:
            prev_prev_etk = prev_prev_x_star - prev_prev_population
    
    # Calculate PID control signal with exploration
    delta_u_t = delta_u(etk, prev_etk, prev_prev_etk, r2, r3, r4, exploration_factor)
    
    # Update population history
    prev_population = population.copy()
    prev_prev_x_star = prev_x_star
    prev_x_star = x_star.copy()
    
    # Apply PID control
    population += eta1 * delta_u_t
    
    # Apply selection, crossover, and mutation
    population = tournament_selection(population, fitness_values)
    
    # Apply crossover
    for i in range(0, population_size - 1, 2):
        population[i], population[i + 1] = dynamic_crossover(
            population[i], population[i + 1], crossover_rate
        )
    
    # Apply adaptive mutation
    for i in range(population_size):
        population[i] = adaptive_mutation(
            population[i], 
            mutation_rate, 
            exploration_factor, 
            t, 
            iterations
        )
    
    # Ensure best solution is preserved
    population[0] = x_star
    
    # Clip values to valid range
    population = np.clip(population, 0, 1)
    
    # Store current best fitness for next iteration
    prev_best_fitness = max_fit
    
    # Print current status
    num_selected_features = list(convert_to_binary(x_star)).count(1)
    print(f"Features selected: {num_selected_features}")
    print(f"Best fitness: {max_fit}")
    print(f"Best accuracy: {max_fit + c * (num_selected_features / df.shape[1])}")
    print(f"Exploration factor: {exploration_factor}")
    print("**** Next iteration ****")