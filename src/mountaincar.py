import numpy as np
import math
import random
import pygame
import sys
import time
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
from collections import Counter

def gen_points(amount, min, max, w, h, tb):
    pts = []
    step_size = (abs(max)+abs(min))/amount
    for i in range(0,amount-1):
        pts = pts + [(int(w*i/amount),tb+int(h*(1-math.sin(3 * (step_size*i+min)))/2))]
    return pts

#TODO: Implement Deep Q learning and DynaQ learning and compare to standard Q

#CAR VARIABLES
init_pos = -0.5 #Initial position
min_pos = -1.2 #Minimum allowed position
max_pos = 0.6 #Maximum allowed position
init_vel = 0 #Initial velocity
acc = 0.001 #Acceleration
max_vel = 0.07 #Max velocity
car_size = 50 #Length of car in pixels

#ENVIRONMENT VARIABLES
goal = 0.5  #Finish flag
g_force = -0.0025 #Gravity

#Q-LEARNING VARIABLES
alpha = 0.1
reward = -1 #Reward for each timestep
discount = 0.618 #Discount factor
num_divs = 30 #Number of division for Q-matrix
iter_max = 1000 #Maximum learning iterations
threshold_timesteps = 250
max_timesteps = 10000
num_tests = 5 #Number of visual tests after learning

# DISPLAY VARIABLES
pygame.display.set_caption('MountainCarV1.0')
width = 640 #Screen width
height = 430 #screen height
top_buffer = 50 #Buffer, for drawing purposes

fps = 60 #Frames per second
framerate = 1. / fps #seconds per frame

# create a surface object, image is drawn on it.
flag = pygame.transform.scale(pygame.image.load('checkerflag.png'),(40, 40))
flag_length = 30
flag_bottom = (int(width * ((goal + abs(min_pos)) / (max_pos - min_pos))), top_buffer + int(height * (1 - math.sin(3 * goal)) / 2))
flag_top = (int(width * ((goal + abs(min_pos)) / (max_pos - min_pos))),
            top_buffer + int(height * (1 - math.sin(3 * goal)) / 2) - flag_length)
lines = gen_points(500, min_pos, max_pos, width, height, top_buffer)

#calculating the interval borders
speedVector = np.arange(-max_vel, max_vel, (max_vel*2) / num_divs)
positionVector = np.arange(min_pos, max_pos, (max_pos-min_pos) / num_divs)

#Statistics for graphs
pos = []
vel = []

def q_learning(divs):
    #Initialise Q-mattrix randomly
    q_table = np.random.rand(divs,divs,3)
    t_steps = threshold_timesteps+1
    i = 0
    #Learning Iteratinos
    while t_steps > threshold_timesteps:
        i += 1
    #for i in range(iter_max):
        #reset
        environment = [init_pos, init_vel]
        tot_r = 0
        t_steps = 0
        done = False

        #Game loop
        while (not done) and (t_steps < max_timesteps):
            #Get Indices of current state of the car
            a, b = obs_to_state(environment)
            #Get best actions in this situation
            act = q_table[a][b][:]
            action = np.argmax(act)

            #Update Environment
            environment[1] += (action - 1) * acc + math.cos(3 * environment[0]) * (g_force)
            environment[1] = np.clip(environment[1], -max_vel, max_vel)
            environment[0] += environment[1]
            environment[0] = np.clip(environment[0], min_pos, max_pos)

            if (environment[0] == min_pos and environment[1] < 0): environment[1] = 0
            done = bool(environment[0] >= goal)

            tot_r += reward

            #Look ahead and update
            a_, b_ = obs_to_state(environment)
            q_table[a][b][action] += alpha*((reward + discount * np.max(q_table[a_][b_][:])) - q_table[a][b][action])
            t_steps += 1

        print("[Iteration " + str(i) + " finished after "+ str(t_steps)+" timesteps]")

    #Simulate the learned policy
    solution_policy = np.argmax(q_table, axis=2)
    print("SOLUTION POLICY")
    print(solution_policy)
    #for _ in range(num_tests):
    #    run_episode(solution_policy,True)
    return q_table

def run_episode(policy=None, render=False):
    #Run episode of with the learned policy
    screen = pygame.display.set_mode((width, height + top_buffer))
    lines = gen_points(500, min_pos, max_pos, width, height, top_buffer)
    environment = [init_pos, init_vel]
    observation = environment
    tot_r = 0
    step = 0
    done = False

    timesteps = 0

    while not done:
        if render:
            pygame.event.get()
            screen.fill((255, 255, 255))
            # Draw Flag
            #pygame.draw.line(screen, (0, 255, 0), flag_bottom, flag_top, 5)

            screen.blit(flag, (600,10))
            # Draw Mountain
            pygame.draw.lines(screen, (0, 0, 0), False, lines, 5)
            # Draw Car
            pygame.draw.circle(screen, (255, 0, 0), (int(width * ((environment[0] + abs(min_pos)) / (max_pos - min_pos))),top_buffer + int(height * (1 - math.sin(3 * environment[0])) / 2)), 15, 0)
            pygame.display.update()
            time.sleep(framerate)

        if policy is None:
            action = np.random.choice([-1,0,1])
        else:
            a, b = obs_to_state(observation)
            act = policy[a][b][:]
            action = np.argmax(act)

        environment[1] += (action - 1) * acc + math.cos(3 * environment[0]) * (g_force)
        environment[1] = np.clip(environment[1], -max_vel, max_vel)
        environment[0] += environment[1]
        environment[0] = np.clip(environment[0], min_pos, max_pos)
        timesteps += 1
        if min_pos == environment[0] and environment[1] < 0: environment[1] = 0
        done = bool(environment[0] >= goal)

        tot_r += reward
        step += 1
    print("finished in "+str(timesteps)+" timesteps")
    pygame.quit()
    return tot_r

def dyna_q(divs, dyna_q_iters):
    #Initialise Q-mattrix randomly
    q_table = np.random.rand(divs,divs,3)

    model = dict()
    model_list = []

    #dyna_q_model = np.random.rand(divs, divs, 3)
    #dyna_q_model_s1prime = np.random.rand(divs, divs, 3)
    #dyna_q_model_s2prime = np.random.rand(divs, divs, 3)
    #history = np.zeros((divs,divs,3))
    i = 0
    t_steps = threshold_timesteps + 1
    # Learning Iteratinos
    while t_steps > threshold_timesteps:
        i += 1
        #reset
        environment = [init_pos, init_vel]
        tot_r = 0
        t_steps = 0
        done = False



        #Game loop
        while (not done) and (t_steps < max_timesteps):
            #Get Indices of current state of the car
            a, b = obs_to_state(environment)
            #Get best actions in this situation
            act = q_table[a][b][:]
            action = np.argmax(act)

            #history[a][b][action] = 1 #Add to history

            #Update Environment
            environment[1] += (action - 1) * acc + math.cos(3 * environment[0]) * (g_force)
            environment[1] = np.clip(environment[1], -max_vel, max_vel)
            environment[0] += environment[1]
            environment[0] = np.clip(environment[0], min_pos, max_pos)

            if (environment[0] == min_pos and environment[1] < 0): environment[1] = 0
            done = bool(environment[0] >= goal)

            tot_r += reward

            #Look ahead and update
            a_, b_ = obs_to_state(environment)
            q_table[a][b][action] += alpha*((reward + discount * np.max(q_table[a_][b_][:])) - q_table[a][b][action])
            t_steps += 1

            #Dyna Q
            model[a, b, action] = (reward, a_, b_)  # store in model, deterministic environment
            model_list = list(model.keys())
            #model_size = len(model_list)
            # Halucinate experience
            for _ in range(dyna_q_iters):
                #(da, db, dact), (dr, da_, db_) = random.choice(model_list)
                (da, db, dact) = random.choice(model_list)
                (dr, da_, db_) = model[da, db, dact]
                if dr != -1:
                    print(dr)
                q_table[da][db][dact] += alpha*((dr + discount * np.max(q_table[da_][db_][:])) - q_table[da][db][dact])

        print("[Iteration " + str(i) + " finished after "+ str(t_steps)+" timesteps]")

    #Simulate the learned policy
    solution_policy = np.argmax(q_table, axis=2)
    print("SOLUTION POLICY")
    print(solution_policy)
    #for _ in range(num_tests):
    #    run_episode(solution_policy,True)
    return q_table

def store_visit(model,a,b,act,r,a_,b_):
    model[(a,b,act)] = (r,a_,b_) #store in model

def obs_to_state(obs):
    indexPostion = sum([obs[0] >= x for x in positionVector]) - 1
    indexSpeed = sum([obs[1] >= x for x in speedVector]) - 1
    return indexPostion, indexSpeed

def plot_q_matrix(q):
    values = np.zeros((num_divs, num_divs))
    for i in range(0, num_divs):
        for j in range(0, num_divs):
            values[i][j] = np.max(q[i, j, :])

    # plot
    plt.imshow(values, aspect='auto', extent=[-1.2, 0.6, -0.07, 0.07])
    plt.colorbar()
    plt.xlabel("position")
    plt.ylabel("speed")
    plt.show()

#for _ in range(1,10):
#    num_divs = _*5
##    q = q_learning(num_divs)
#    np.save("qMatrix_"+str(iter_max)+"iters_"+str(num_divs)+"divs.npy", q)
#q = q_learning(num_divs)
#np.save("qMatrix_"+str(iter_max)+"iters_"+str(num_divs)+"divs.npy", q)
#q = dyna_q(num_divs,100)
#np.save("dynaq100Matrix_"+str(iter_max)+"iters_"+str(num_divs)+"divs.npy", q)
qmat = np.load("qMatrix_"+str(iter_max)+"iters_"+str(num_divs)+"divs.npy")
print("Standard Q-learning")
run_episode(qmat, True)
plot_q_matrix(qmat)
print("Dyna Q-learning")
qmat = np.load("dynaq100Matrix_"+str(iter_max)+"iters_"+str(num_divs)+"divs.npy")
run_episode(qmat, True)
plot_q_matrix(qmat)