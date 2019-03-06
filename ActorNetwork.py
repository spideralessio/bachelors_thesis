from tensorflow.initializers import variance_scaling
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Lambda, concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras.backend as K

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)   
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size,action_dim):
        print("Now we build the model")
        with tf.name_scope('ActorNetwork') as scope:
            S = Input(shape=[state_size], name='state')   
            h0 = Dense(HIDDEN1_UNITS, activation='relu', name='DenseLayer0')(S)
            h1 = Dense(HIDDEN2_UNITS, activation='relu', name='DenseLayer1')(h0)
            Steering = Dense(1, name='Steering',activation='tanh',kernel_initializer=variance_scaling(scale=1e-4, distribution='normal'), bias_initializer=variance_scaling(scale=1e-4, distribution='normal'))(h1)#,init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)  
            Acceleration = Dense(1, name='Acceleration',activation='tanh', kernel_initializer=variance_scaling(scale=1e-4, distribution='normal'), bias_initializer=variance_scaling(scale=1e-4, distribution='normal'))(h1)#,init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)   
            #Brake = Dense(1,activation='sigmoid', kernel_initializer=variance_scaling(scale=1e-4, distribution='normal'), bias_initializer=variance_scaling(scale=1e-4, distribution='normal'))(h1) 
            #Gear = Dense(3,activation='softmax')(h1)
            V = concatenate([Steering,Acceleration], name='Action')#,Brake])          
            model = Model(inputs=S, outputs=V)
            return model, model.trainable_weights, S

