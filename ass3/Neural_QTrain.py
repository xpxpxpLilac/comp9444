import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA = 0.99 # discount factor
INITIAL_EPSILON = 0.85 # starting value of epsilon
FINAL_EPSILON = 0.08 # final value of epsilon
EPSILON_DECAY_STEPS = 500 # decay period

HIDDEN_NODE = 180
TRAINING_STEP = 2
BATCH_SIZE = 80
# s_t, a_t, r_t, s_t+1
replay_experience = []
REPLAY_SIZE = 10000

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

# TODO: Define Network Graph
def DQNetwork(state_in, STATE_DIM, HIDDEN_NODE, ACTION_DIM):


    layer1 = tf.layers.dense(state_in, HIDDEN_NODE, tf.nn.tanh)
    # layer3 = tf.layers.dense(layer1,HIDDEN_NODE)
    layer2_output = tf.layers.dense(layer1, ACTION_DIM)

    return layer2_output

# TODO: Network outputs
q_values = DQNetwork(state_in,STATE_DIM,HIDDEN_NODE, ACTION_DIM)
q_action = tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)  #?????????

# TODO: Loss/Optimizer Definition
loss = tf.reduce_mean(tf.square(tf.subtract(target_in,q_action)))
optimizer = tf.train.AdamOptimizer().minimize(loss)

def updateReplay(replay_experience, state, action, reward, next_state, done):

    # append
    replay_experience.append([state, action, reward, next_state, done])
    
    # Ensure replay_experience is not larger than REPLAY_SIZE
    if len(replay_experience) > REPLAY_SIZE:
        replay_experience.pop(0)

    return None

def getTrainingBatch(replay_experience,BATCH_SIZE,q_values):
    minibatch = random.sample(replay_experience, BATCH_SIZE)
    # print("==================================="+ str(minibatch))
    state_batch = [data[0] for data in minibatch]
    # print("===================================="+ str(state_batch))
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]

    q_value_batch = q_values.eval(feed_dict={
        state_in: next_state_batch
    })
    target_batch = []
    for j in range (BATCH_SIZE):
        DONE = minibatch[j][4]
        if DONE:
            target_batch.append(reward_batch[j])
        else:    
            target = reward_batch[j] + GAMMA * np.max(q_value_batch[j])
            target_batch.append(target)
    return state_batch, action_batch, target_batch, next_state_batch


# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action


# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= epsilon / EPSILON_DECAY_STEPS
    
    # Move through env according to e-greedy policy
    for step in range(STEP):

        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))
        nextstate_q_values = q_values.eval(feed_dict={
            state_in: [next_state]
        })
        
        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated
        # if done:
        #     target = reward
        # else:
        #     target = reward + GAMMA * np.max(nextstate_q_values[len(nextstate_q_values)-1])
        # if(step % TRAINING_STEP == 0 and step != 0):
        
        updateReplay(replay_experience, state, action, reward, next_state, done)
        
        # if(step % TRAINING_STEP == 0 and step != 0 and len(replay_experience) > BATCH_SIZE):
        if(len(replay_experience) > BATCH_SIZE):
            # Do TRAINING_TIME training steps
            state_batch, action_batch, target_batch, _ = getTrainingBatch(replay_experience,BATCH_SIZE,q_values)

            session.run([optimizer], feed_dict={
                target_in: target_batch,
                action_in: action_batch,
                state_in: state_batch
            })

        # Update
        state = next_state
        if done:
            break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                # env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
