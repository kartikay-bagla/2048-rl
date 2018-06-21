""" The structure of the NN will be 16x8x4
    The first 16 values will be weights for the 1st layer
    The next 128 values will be weights for layer 2
    The next 32 values will be weights for layer 3
    And the next 16 values will be biases for layer 1
    And the next 8 values will be biases for layer 2
    And the next 4 values will be biases for layer 3 
    
    We have taken an initial population of 50
    the best 7 will be taken for reproduction.
    The number of children then will be 7 * 6.
    To return to our original population, we take 8 random specimens from inital
    population to survive and move on to the next gen."""

import Game2048
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from drawnow import drawnow

random.seed(0)
total_params = 204
init_pop = 2000

maxs = []
grids = []

"""No of top specimens from current pop who can become parents"""
spec_to_save = 30

"""No of specimens to save from the top, i.e. with highest score"""
top_to_save = 620 

"""No of specimens to save randomly from current pop"""
random_selected = 200

"""No of new random specimens"""
new_random = 100
mutation_chance = 0.001

"""No of random specimens from current pop who can become parents"""
random_parents = 15

def create_init_pop(num):
    return [[random.uniform(-1.0, 1.0) for i in range(total_params)] for i in range(num)]

"""def reproduce(spec):
    for i in range(len(spec)):
        if random.random() <= mutation_chance:
            spec[i] = random.uniform(-1.0, 1.0)
    return spec"""

def reproduce(spec1, spec2):
    """Creates the  new child of spec1 and spec2 by taking equal amounts of their genes with a 0.1% chance of mutation"""
    a = [1,0]*(total_params//2)
    random.shuffle(a)

    """Mutation chance"""
    for i in range(len(a)):
        if random.random() <= mutation_chance:
            if a[i]==0:
                a[i] = 1
            else:
                a[i] = 0
    
    new_kid = []
    
    for i in range(len(spec1)):
        
        if a[i] == 0:
            new_kid.append(spec1[i])
        else:
            new_kid.append(spec2[i])
    return new_kid

def evolution(pop):
    best = sorted(pop, key= lambda x: x[-1])
    best.reverse()
    a = [i[:-1:] for i in best]
    b = [len(i) for i in pop]
    try:
        print(b.index(206))
    except:
        pass
    best = a[:spec_to_save:]
    list_for_evol = [(x, y) for x in best for y in best if x != y]
    new_pop = []
    for i in list_for_evol:
        new_pop.append(reproduce(i[0], i[1]))
    random_for_evol = random.sample(a, random_parents)
    list_for_evol = [(x, y) for x in random_for_evol for y in random_for_evol if x != y]
    for i in list_for_evol:
        new_pop.append(reproduce(i[0], i[1]))
    random_saved = random.sample(a, random_selected)
    
    
    a = a[:top_to_save]
    new_pop += random_saved
    new_pop += a
    new_pop += create_init_pop(new_random)
    
    return new_pop

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def relu(x):
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = 0
    return x

def matmul(X, Y):
    result = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*Y)] for X_row in X]
    return result

def play(spec, game = Game2048.Game2048()):
    
    global maxs
    global grids
    current_game = []
    s = game.reset()
    current_game.append(s)
    s = [s]
    w1 = spec[:16:]

    w2 = spec[16:144:]
    w2 = [w2[i*8:(i+1)*8] for i in range(0,16)]
    
    w3 = spec[144:176:]
    w3 = [w3[i*4:(i+1)*4] for i in range(0,8)]
    
    b1 = spec[176:192:]
    b2 = spec[192:200:]
    b3 = spec[200::]

    r = 0
    while not game.get_game_over():
        input1 = matmul(s, w2)
        activation1 = [relu(input1[0])]
        input2 = matmul(activation1, w3)
        activation2 = [softmax(input2[0]).tolist()]
        pred = np.argmax(activation2[0])
        s, reward, _ = game.step(pred)
        current_game.append(s)
        s = [s]
        r += reward
    
    new_spec = spec[::]
    new_spec.append(r)
    maxs.append(max(game.get_grid()))
    grids.append(current_game)
    return new_spec

def test(pop, game = Game2048.Game2048()):
    global maxs
    maxs = []
    new_pop = []
    for i in range(len(pop)):
        new_pop.append(play(pop[i], game))
    return new_pop, maxs

max_scores = []
avg_scores = []

def plotter():
    plt.scatter(range(len(avg_scores)), avg_scores, marker=".")
    plt.scatter(range(len(max_scores)), max_scores, marker=".")

plt.ion()
fig = plt.figure()

pop = create_init_pop(init_pop)
game = Game2048.Game2048()
for i in range(1000):
    print("Generation:", i+1)
    
    pop_score, list_max = test(pop, game)
    
    ind_max_num = np.argmax(list_max)
    print ("Highest Number:", list_max[ind_max_num], ", Score at Max Num:", pop_score[ind_max_num][-1])

    scores = [i[-1] for i in pop_score]

    c = max(scores)
    max_scores.append(c)

    max_index = np.argmax(scores)

    with open("games.dat", "ab") as f:
        a = grids[max_index]
        b = (i, scores[max_index], c, a)
        pickle.dump(b, f)
        print("File Saved")

    avg_score = sum(scores)/len(scores)
    avg_scores.append(avg_score)

    print ("Max Score:", c, ", Avg Score:", avg_score)
    print ()
    drawnow(plotter)
    pop = evolution(pop_score)
