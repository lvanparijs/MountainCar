import numpy as np
import math
import random
import pygame
import sys
import time

def gen_points(amount, min, max, w, h, tb):
    pts = []
    step_size = (abs(max)+abs(min))/amount
    for i in range(0,amount-1):
        pts = pts + [(int(w*i/amount),tb+int(h*(1-math.sin(3 * (step_size*i+min)))/2))]
    return pts

#CAR VARIABLES
init_pos = -0.5 #Initial position
min_pos = -1.2 #Minimum allowed position
max_pos = 0.6 #Maximum allowed position
init_vel = 0 #Initial velocity
acc = 0.001 #Acceleration
max_vel = 0.07 #Max velocity

#ENVIRONMENT VARIABLES
goal = 0.5  #Finish flag
g_force = -0.0025 #Gravity

#Q-LEARNING VARIABLES
reward = -1 #Reward for each timestep
discount = 0.999 #Discount factor
num_divs = 50 #Number of division for Q-matrix
iter_max = 500 #Maximum learning iterations
num_tests = 100 #Number of visual tests after learning

# DISPLAY VARIABLES
width = 640 #Screen width
height = 430 #screen height
top_buffer = 50 #Buffer for drawing purposes

fps = 60 #Frames per second
framerate = 1. / fps #fps in seconds

flag_length = 30
flag_bottom = (
int(width * ((goal + abs(min_pos)) / (max_pos - min_pos))), top_buffer + int(height * (1 - math.sin(3 * goal)) / 2))
flag_top = (int(width * ((goal + abs(min_pos)) / (max_pos - min_pos))),
            top_buffer + int(height * (1 - math.sin(3 * goal)) / 2) - flag_length)
lines = gen_points(500, min_pos, max_pos, width, height, top_buffer)

#calculating the interval borders
speedVector = np.arange(-max_vel, max_vel, (max_vel*2) / num_divs)
positionVector = np.arange(min_pos, max_pos, (max_pos-min_pos) / num_divs)

def q_learning():
    #Initialise Q-mattrix randomly
    q_table = np.random.rand(num_divs,num_divs,3)

    #Learning Iteratinos
    for i in range(iter_max):
        #reset
        environment = [init_pos, init_vel]
        tot_r = 0
        t_steps = 0
        done = False

        #Game loop
        while not done:
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
            q_table[a][b][action] = reward + discount * np.max(q_table[a_][b_][:])
            t_steps += 1
        print("[Iteration " + str(i) + " finished after "+ str(t_steps)+" timesteps]")

    #Simulate the learned policy
    solution_policy = np.argmax(q_table, axis=2)
    print("SOLUTION POLICY")
    print(solution_policy)
    for _ in range(num_tests):
        run_episode(solution_policy,True)
    return

def run_episode(policy=None, render=False):
    #Run episode of with the learned policy
    screen = pygame.display.set_mode((width, height + top_buffer))
    lines = gen_points(500, min_pos, max_pos, width, height, top_buffer)
    environment = [init_pos, init_vel]
    observation = environment
    tot_r = 0
    step = 0
    done = False

    while not done:
        if render:
            pygame.event.get()
            screen.fill((0, 0, 0))
            # Draw Flag
            pygame.draw.line(screen, (0, 255, 0), flag_bottom, flag_top, 5)
            # Draw Mountain
            pygame.draw.lines(screen, (255, 255, 255), False, lines, 5)
            # Draw Car
            pygame.draw.circle(screen, (255, 0, 0), (int(width * ((environment[0] + abs(min_pos)) / (max_pos - min_pos))),top_buffer + int(height * (1 - math.sin(3 * environment[0])) / 2)), 15, 0)
            pygame.display.update()
            time.sleep(framerate)

        if policy is None:
            action = np.random.choice(np.array([-1,0,1]))
        else:
            a, b = obs_to_state(observation)
            action = policy[a][b]

        environment[1] += (action - 1) * acc + math.cos(3 * environment[0]) * (g_force)
        environment[1] = np.clip(environment[1], -max_vel, max_vel)
        environment[0] += environment[1]
        environment[0] = np.clip(environment[0], min_pos, max_pos)
        if (environment[0] == min_pos and environment[1] < 0): environment[1] = 0
        done = bool(environment[0] >= goal)

        tot_r += reward
        step += 1
    pygame.quit()
    return tot_r

def obs_to_state(obs):
    indexPostion = sum([obs[0] >= x for x in positionVector]) - 1
    indexSpeed = sum([obs[1] >= x for x in speedVector]) - 1
    return indexPostion, indexSpeed

q_learning()