from gym_torcs import TorcsEnv
import numpy as np
import random
import argparse
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit
import sys

OU = OU()       #Ornstein-Uhlenbeck Process

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def playGame(train_indicator=1):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 2  #Steering/Acceleration/Brake
    state_dim = 25 -2 #of sensors input + wanted_speed

    np.random.seed(1337)

    vision = False

    EXPLORE = 100000.
    episode_count = 10000
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    sess = tf.Session(config=config)

    from tensorflow.keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True, gear_change=False)

    episodes_rewards = []
    #gear_p = 0.19

    #Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights("actormodel.h5")
        critic.model.load_weights("criticmodel.h5")
        actor.target_model.load_weights("actormodel.h5")
        critic.target_model.load_weights("criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")
    summary_writer = tf.summary.FileWriter("output", sess.graph)

    print("TORCS Experiment Start.")
    for i in range(episode_count):
        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()
        env.wanted_speed = random.randint(50,110)
        env.avg_speed = 0
        #gear = 1
        #gear_p += 0.001
        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, env.wanted_speed))
        
        total_reward = 0.
        loss = 0.
        for j in range(max_steps):
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            
            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.10)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.7 , 1.00, 0.10)
            #noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)

            #for x in range(3,action_dim):
            #    noise_t[0][x] = 0#train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.5, 0.60, 0.10)

            #The following code do the stochastic brake
            if random.random() <= 0.1 and train_indicator:
                print("********Now we apply the brake***********")
                noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  -0.1, 1.00, 0.10)
            
            for x in range(action_dim):
                a_t[0][x] = a_t_original[0][x] + noise_t[0][x]
            
            #a_t[0][2] = 0#max(0., a_t[0][2])
            #a_t[0][1] = -1
            #if random.random() >= gear_p and train_indicator == 1:
            #    print("Stochastic gear p:", 1-gear_p)
            #    gear_change = random.randint(-1,1)
            #    print("stochastic gear: %d"% gear_change)
            #    a_t = a_t.tolist()
            #    a_t[0] = a_t[0][:3] + [1 if g-1 == gear_change else 0 for g in range(3)]
            #    a_t = np.array(a_t)
            #else:
            #    gear_change = np.argmax(a_t[0][3:]) - 1
            #def sigmoid(x, derivative=False):
            #    x = np.array(x)
            #    sigm = 1. / (1. + np.exp(-x))
            #    if derivative:
            #        return sigm * (1. - sigm)
            #    return sigm.tolist()
            #a_t = a_t.tolistos.sys()
            #a_t[0] = a_t[0][:3] + sigmoid(a_t[0][3:])
            #a_t = np.array(a_t)
            #print("AVG Speed", env.avg_speed, "WANTED Speed", env.wanted_speed, "Speed", ob.speedX*300)
            steering = a_t[0][0]
            acceleration = a_t[0][1] if a_t[0][1] >= 0 else 0.
            brake = abs(a_t[0][1]) if a_t[0][1] < 0 else 0.
            #gear = gear + gear_change
            #if gear > 6: gear = 6
            #if gear < -1: gear = -1
            performed_action = [steering, acceleration, brake]
            ob, r_t, done, info = env.step(performed_action)

            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, env.wanted_speed))
            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            
            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])  
           
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
       
            if (train_indicator):
                loss += critic.model.train_on_batch([states,actions], y_t) 
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1
        
            #print("Episode", i, "Step", step, "Action", a_t[0], "Reward", r_t, "Loss", loss)
            step += 1
            if done:
                break
        
        episodes_rewards.append(total_reward)
        print("mean_reward", np.array(episodes_rewards).mean())

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)
        if np.mod(i, 200) == 0:
            if (train_indicator):
                print("Now we save the intermediate model")
                actor.model.save_weights("actormodel.%d.h5"%i, overwrite=True)
                with open("actormodel.%d.json"%i, "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.%d.h5"%i, overwrite=True)
                with open("criticmodel.%d.json"%i, "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)
        summary = tf.Summary()
        summary.value.add(tag='episode_reward', simple_value=total_reward)
        summary_writer.add_summary(summary, i)
        summary_writer.flush()
        summary = tf.Summary()
        summary.value.add(tag='loss', simple_value=loss)
        summary_writer.add_summary(summary, i)
        summary_writer.flush()
        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward) + " Steps:%d"%j)
        print("Total Step: " + str(step))
        print("")

    with open("rewards.json", "w") as outfile:
        json.dump(episodes_rewards, outfile)

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        train_indicator = int(sys.argv[1])
        sys.argv = [sys.argv[0]]
        playGame(train_indicator)
    else:
        playGame()
