import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from deap import base, creator, tools, algorithms
import random

# Step 1: Generate Dummy Data (3 inputs → target moisture content)
data = np.random.rand(100, 3)
target = data[:, 0]*0.5 + data[:, 1]*0.3 + data[:, 2]*0.2
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# Step 2: Define Fitness Function
def evaluate(ind):
    ind = [max(1, i) for i in ind]  # Ensure neuron counts ≥ 1
    model = MLPRegressor(hidden_layer_sizes=(ind[0], ind[1]), max_iter=2000)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return (mean_squared_error(y_test, pred),)

# Step 3: Genetic Algorithm Setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 1, 100)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, 2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("select", tools.selTournament, tournsize=3)

# Step 4: Custom Integer Mutation (no bit-level operations)
def mutate(ind):
    for i in range(len(ind)):
        if random.random() < 0.1:
            ind[i] = max(1, random.randint(1, 100))
    return ind,
toolbox.register("mutate", mutate)

# Step 5: Run Genetic Algorithm
pop = toolbox.population(n=20)
algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=30, verbose=True)

# Step 6: Output Best Neural Network Architecture
best = tools.selBest(pop, 1)[0]
print("Best hidden layers:", best)
