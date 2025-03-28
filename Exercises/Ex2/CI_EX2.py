import numpy as np
import random
import matplotlib.pyplot as plt



"""
Genetic algorithm to solve "Travel Salesman Problem"
   Looking for shortests distance

    City has a coordinate (x,y)

"""

n_population = 150
n_generations = 1000 #number of generations (number of loop time to do the algorithm)
mutation_rate = 0.02
alpha = 0.2 #parameteer indicating where to split the parents in the crossover
enable_prints = False

######### Create the population
def genesis(city_names, n_population):
    city_names = list(city_names)
    population_set = []
    for i in range (n_population):
        sol_i = ''.join(random.sample(city_names, len(city_names))) #randomly shuffle the city names
        population_set.append(sol_i)
    return population_set



######### Determine Fitness
def distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2)) #euclidean distance

#fitness of path. calcullates the total distance of the path. 
#path is what the travelers toke, cities list contains the coordinates of each city
def fitness_eval(path, cities_dict):
    n_cities = len(cities_dict)
    total = 0
    for i in range(1, n_cities): #ABC
        total += distance(cities_dict[path[i-1]], cities_dict[path[i]])
    return total

#calculates the fitness of all solutions in the population
def get_all_fitness(populattion, n_cities_dict):
    n_population = len(populattion)
    fitness_list = np.zeros(n_population)
    
    #looping over all solutions computing the fitness for each solution
    for i in range(n_population):
        fitness_list[i] = fitness_eval(populattion[i], n_cities_dict)
    return fitness_list



def mating_selection(population, fitness_list):
    total_fit = fitness_list.sum()
    prob_list = fitness_list / total_fit  # Normalize each fitness value
    if enable_prints:
        print(f"total_fit {total_fit}\nprob_list: {prob_list}")
    
    # Select random parents based on probabilities
    parents_list_a = np.random.choice(list(range(len(population))), size=len(population), p=prob_list, replace=True)
    parents_list_b = np.random.choice(list(range(len(population))), size=len(population), p=prob_list, replace=True)
    
    if enable_prints:
        print(f"parents_list_a: {parents_list_a}")
        print(f"parents_list_b: {parents_list_b}")
    
    # Convert indices to actual population members
    parents_list_a = [population[i] for i in parents_list_a]
    parents_list_b = [population[i] for i in parents_list_b]
    
    if enable_prints:
        print(f"parents_list_a: {parents_list_a}")
        print(f"parents_list_b: {parents_list_b}")
    
    return np.array([parents_list_a, parents_list_b])



## For mating we will swap the head of population a with the tail of b.
# Alpha will help us deside the location of the head and tail
# we also want to make sure that the same city did not appear twice in the same path
# we do soo by keeping a trck of the cities that have been swapped and replace them with the corresponding city from the other parent
def mate_parents(parents_a, parents_b):
    #allpha = np.random.random()
    head_a = parents_a[:int(alpha*len(parents_a))] # head is a new list from the begining until Alpha*len
    tail_a = parents_a[int(alpha*len(parents_a)):] # tail is a new list from the end until Alpha*len
    tail_b = parents_b[int(alpha*len(parents_b)):]
    
    mapping = {tail_b[i]: tail_a[i] for i in range(len(tail_a))} #value of the dict will be the tail of a
    if enable_prints:
        print(f"mapping: {mapping}")
    
    for i in range(len(head_a)):
        while head_a[i] in tail_b:
            if enable_prints:
                print(head_a, mapping[head_a[i]])
            head_a = head_a.replace(head_a[i], mapping[head_a[i]])  # Replace the city with the corresponding city from the other parent
    if enable_prints:
        print(head_a + tail_b)
    return head_a + tail_b
    
    
    

def mate_parents_list(mating_list):
    new_population = []
    for i in range(mating_list.shape[1]):
        parent_a = mating_list[0][i]
        parent_b = mating_list[1][i]
        if enable_prints:
            print(f"parents: {parent_a}, {parent_b}")
        offspring = mate_parents(parent_a, parent_b)
        new_population.append(offspring)
    
    return new_population




#we will swapp randomly between a city in the path with another city in the path
#this will be done n_cities*mutation_rate times
#if mutation_rate is big wee will have more swapping (more mutations)
## Question.. can a mutation have the same city in its path?
def mutate_offspring(offspring, n_cities):
    # print(f"offspring: {offspring}")
    offspring = list(offspring) #takes a path and breaks it down to cities
    # print(f"offspring list: {offspring}")
    for q in range(int(n_cities*mutation_rate)):
        a = np.random.randint(0, n_cities)
        b = np.random.randint(0, n_cities)
        # print(f"swapping {a} with {b}")
        #Swapping 
        offspring[a] = offspring[b]
        offspring[b] = offspring[a]
    
    return ''.join(offspring)



#GO over all new population and perform a mutation on each offspring
#n_cities is the number of cities in the problem
def mutate_population(new_population, n_cities):
    mutated_population = []
    for offspring in new_population:
        mutated_population.append(mutate_offspring(offspring, n_cities))
    return mutated_population





def solve(cities_list):
    n_cities = len(cities_list)
    population = genesis(cities_list.keys(), n_population)
    fitness_list = np.zeros(n_population)
    
    best_solutions = [-1, np.inf, []] # initial best solution where [index, fitness, path]
    BEST = []
    
    
    for i in range(n_generations): #number of generations
        if i % 100 == 0:
            print(i, fitness_list.min(), fitness_list.mean(), best_solutions[1])
            
            # fig = plt.figure(0)
            # fig.clf()
            # plt.plot(BEST, 'r-')
            # plot_path(cities_list, best_solutions[2], best_solutions[1])
            # plt.pause(0.1)
            
        fitness_list = get_all_fitness(population, cities_list) #calculate the distances of the paths
        
        #saving the best solution
        if fitness_list.min() < best_solutions[1]:
            best_solutions[0] = i           #Saving index for the next best solution
            best_solutions[1] = fitness_list.min() #saving minimum value of fitness found
            best_solutions[2] = population[i]  #saving the path of the best solution
            # best_solutions[2] = population[fitness_list.min() == fitness_list][0]  #saving the path of the best solution

        mating_list = mating_selection(population, fitness_list) #selecting the parents for mating
        new_population = mate_parents_list(mating_list) #mating the parents to create new population
        
        population = mutate_population(new_population, n_cities) #mutating the new population
        BEST.append(best_solutions[1])
        
        if i > 3000 and np.all(np.array(BEST[:-2000]) == BEST[-1:]): 
            # if the best solution is not changing for 2000 generations
            # we have made lots of generastions and the min value is the same 
            break
        
        
    return best_solutions #returning the best solution and the best fitness found





def plot_path(cities_list, path, fitness):
    fig = plt.figure(1, figsize=(6, 4))
    fig.clf()
    
    loc = np.array(list(cities_list.values()))
    plt.scatter(x=loc[:, 0], y=loc[:, 1], s=500, zorder=1, c='blue', marker='o')
    
    
    #plotting the cities
    for city in cities_list.values():
        plt.text(city[0], city[1], city, fontsize=12, ha='center', va='center')

    
    #plotting the cities
    for i in range(len(path) - 1):
        plt.plot([cities_list[path[i]][0], cities_list[path[i+1]][0]], 
                 [cities_list[path[i]][1], cities_list[path[i+1]][1]], 'k', zorder=0)

    plt.title(f"Visiting: {len(path)} cities in distance {fitness:.2f}",size=16)
    # plt.show()

example_cities_dict = {
    'A': [123, 456],
    'B': [789, 234],
    'C': [567, 890],
    'D': [345, 678],
    'E': [901, 123],
    'F': [234, 567],
    'G': [678, 345],
    'H': [890, 789],
    'I': [456, 901],
    'J': [123, 234],
    'K': [567, 678],
    'L': [789, 890],
    'M': [345, 123],
    'N': [901, 456],
    'O': [234, 789],
    'P': [678, 234],
    'Q': [890, 567],
    'R': [456, 678],
    'S': [123, 890],
    'T': [567, 345]
}

sol = solve(example_cities_dict)
print(f"Best solution: {sol}")
plot_path(example_cities_dict, sol[2], sol[1])
plt.show()

# example_pop = genesis(example_cities_dict.keys(), len(example_cities_dict))
# example_parents_list = mating_selection(example_pop, get_all_fitness(example_pop, example_cities_dict))
# example_cross = mate_parents_list(example_parents_list)
# example_mutate = mutate_population(example_cross, len(example_cross))

# print(f"example pop: {example_pop}")
# print(f"example fitness: {get_all_fitness(example_pop, example_cities_dict)}")
# print(f"parents list after selection: {example_parents_list}")
# print(f"example after crossover {example_cross}")
# print(f"example after mutation {example_mutate}")