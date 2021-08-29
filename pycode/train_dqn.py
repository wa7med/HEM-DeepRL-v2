from algos.DQN import DQN
from pycode.smart_home import SmartHome
import tensorflow, numpy, glob

def train_dqn( epoch=200, save_model=True ):
	env = SmartHome()
	algo = DQN( env, verbose=1 )
	algo.loop( num_episodes=epoch )
	if save_model: algo.actor.save( f"trained_models/neural_network_dqn_ep{epoch}.h5" )

def _get_action_dqn( net, state, env ):
	return numpy.argmax(net(state.reshape((1, -1))))