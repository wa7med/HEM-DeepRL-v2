import sys, os; os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from pycode.smart_home import SmartHome
from pycode.plotter import LinePlotter, HistoPlotter, DoubleLinePlotter, DemandsPlotter, GenerationPlotter
from algos.mcPPO import mcPPO
from algos.DQN import DQN
import tensorflow, numpy, glob

import numpy as np

def train_ppo( epoch=200, save_model=True ):
	env = SmartHome()
	algo = mcPPO( env, verbose=1 )
	algo.loop( num_episodes=epoch )
	if save_model: algo.actor.save( f"trained_models/neural_network_trained.h5" )


def test_models( _get_action, network_name, debug=False ):
	env = SmartHome()
	neural_network = tensorflow.keras.models.load_model( network_name, compile=False )
	state = env.reset()
	# Set first day of the year
	env.month = 1
	env.day = 1
	env.time = 0
	env.ev_battery_state = 0
	env.home_battery_state = 0
	# Cycle for the full year
	current_total_spent = 0
	current_month = 1
	monthly_total_list = []
	hourly_action_list = [[] for _ in range(24)]
	hourly_home_battery_list = [[] for _ in range(24)]
	hourly_ev_battery_list = [[] for _ in range(24)]
	hourly_price_list = [[] for _ in range(24)]
	hourly_pv_list = [[] for _ in range(24)]

	hourly_demands = [[] for _ in range(24)]	# emarche

	use_ev_fail_total = 0
	charge_ev_fail_total = 0

	while True:
		action = _get_action( neural_network, state, env )
		state, _, _, info = env.step( action )
		use_ev_fail_total += info['use_ev_fail']
		charge_ev_fail_total += info['charge_ev_fail']

		hourly_action_list[ env.time ].append(action)
		hourly_home_battery_list[ env.time ].append(env.home_battery_state)
		hourly_ev_battery_list[ env.time ].append(env.ev_battery_state)
		hourly_price_list[ env.time ].append(env.processed_data.get_SG_prices(env.month, env.day, env.time))
		hourly_pv_list[ env.time ].append(env.processed_data.get_PV_entry(env.month, env.day, env.time))
		
		hourly_demands[ env.time ].append(env.processed_data.get_H4_consumes(env.month, env.day, env.time))	# emarche

		if env.month > current_month:
			monthly_total_list.append( env.total_euros_spent - current_total_spent )
			current_month = env.month			
			current_total_spent = env.total_euros_spent

		if ( env.month == 1 and env.day == 1 and env.time == 0 ): 
			monthly_total_list.append( env.total_euros_spent - current_total_spent )
			break

	if debug:
		print( f"This years spent: {env.total_euros_spent:6.2f} euros" )
		print( f"Total days with not enough charge on the EV: {use_ev_fail_total}" )
		print( f"Total try to charge while EV is away: {charge_ev_fail_total}\n" )

	hourly_home_battery = [numpy.mean( total ) for total in hourly_home_battery_list]
	hourly_ev_battery = [numpy.mean( total ) for total in hourly_ev_battery_list]
	hourly_price = [numpy.mean( total ) for total in hourly_price_list]
	hourly_pv = [numpy.mean( total ) for total in hourly_pv_list]
	hourly_mean_charge_battery = []
	hourly_mean_charge_pv = []

	for idx, time in enumerate(hourly_action_list):	
		hourly_mean_charge_battery.append( time.count(2) / len(time) * 11 )
		hourly_mean_charge_battery[-1] += ( time.count(4) / len(time) * 11 )

		hourly_mean_charge_pv.append( time.count(1) / len(time) * hourly_pv[idx] )
		hourly_mean_charge_pv[-1] += ( time.count(3) / len(time) * hourly_pv[idx] )
	
	hourly_demands = [np.mean(total) for total in hourly_demands]
	return monthly_total_list, use_ev_fail_total, hourly_home_battery, hourly_ev_battery, hourly_price, hourly_mean_charge_battery, hourly_mean_charge_pv, hourly_demands


def _get_action_ppo( net, state, env ):
	softmax_out = net(state.reshape((1, -1)))
	return numpy.random.choice(5, p=softmax_out.numpy()[0])


def _get_action_random( net, state, env ):
	return env.action_space.sample()


if __name__ == "__main__":
	print("\nHEM based on Deep RL Model  !\n")


	if(sys.argv[1] == "-train_model"):
		print("\nStarting the training  !\n")
		train_ppo( epoch=int(sys.argv[2]), save_model=True )

	elif(sys.argv[1] == "-test_model"):
		print("\nStarting the testing  !\n")
		test_models( _get_action_ppo, network_name="trained_models/neural_network_trained.h5", debug=True )

	elif(sys.argv[1] == "-plot_graph" and sys.argv[2] == "money_spent"): 
		histo_plotter = HistoPlotter()
		monthly_total_list_ppo, _, _, _, _, _, _, _ = test_models( _get_action_ppo, network_name="trained_models/neural_network_trained.h5" )
		monthly_total_list_random, _, _, _, _, _, _, _ = test_models( _get_action_random, network_name="trained_models/neural_network_trained.h5" )
		histo_plotter.trained_list = monthly_total_list_ppo
		histo_plotter.random_list = monthly_total_list_random
		histo_plotter.plot( )

	elif(sys.argv[1] == "-plot_graph" and sys.argv[2] == "battery_charge"): 
		_, _, hourly_home_battery__ppo, hourly_ev_battery__ppo, hourly_price_ppo, hourly_charge_ppo_sg, hourly_charge_ppo_pv, _ = test_models( _get_action_ppo, network_name="trained_models/neural_network_trained.h5" )
		line_plotter = DoubleLinePlotter( hourly_home_battery__ppo, hourly_ev_battery__ppo, hourly_price_ppo, hourly_charge_ppo_sg, hourly_charge_ppo_pv )
		line_plotter.plot( )

	elif(sys.argv[1] == "-plot_graph" and sys.argv[2] == "reward_function"): 
		plotter = LinePlotter(x_label="Episode", y_label="Reward", title="Reward Function")
		plotter.load_array([ glob.glob("generated/reward_PPO_*.txt") ])
		plotter.process_data( rolling_window=100, starting_pointer=0 )
		plotter.render( labels=["PPO"], colors=["r"] )

	elif(sys.argv[1] == "-plot_graph" and sys.argv[2] == "demands"): 

		_, _, hourly_home_battery__ppo, hourly_ev_battery__ppo, hourly_price_ppo, hourly_charge_ppo_sg, hourly_charge_ppo_pv, house_demands = test_models( _get_action_ppo, network_name="trained_models/neural_network_trained.h5" )
		
		print(np.array(hourly_charge_ppo_pv).shape)
		print(np.array(hourly_charge_ppo_sg).shape)

		line_plotter = DemandsPlotter(house_demands, hourly_charge_ppo_pv)
		line_plotter.plot( )

	# PV – Battery discharge- grid
	elif(sys.argv[1] == "-plot_graph" and sys.argv[2] == "generation"): 

		_, _, hourly_home_battery__ppo, hourly_ev_battery__ppo, hourly_price_ppo, hourly_charge_ppo_sg, hourly_charge_ppo_pv, house_demands = test_models( _get_action_ppo, network_name="trained_models/neural_network_trained.h5" )
		
		for i in range(len(hourly_home_battery__ppo)):
			hourly_home_battery__ppo[i] -= 50
		print(hourly_home_battery__ppo)
		print(np.array(hourly_home_battery__ppo).shape)
		print(np.array(hourly_charge_ppo_pv).shape)
		print(np.array(hourly_charge_ppo_sg).shape)
		line_plotter = GenerationPlotter(hourly_charge_ppo_pv, hourly_charge_ppo_sg, hourly_home_battery__ppo)
		line_plotter.plot( )

	else:
		raise ValueError(f"Invalid argument: '{sys.argv[1]}' (options: [-train_model *n*, -test_model, -plot_graph *type*])")

