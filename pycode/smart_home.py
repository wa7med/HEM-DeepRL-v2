from pycode.constant import *
from pycode.processed_data import ProcessedData
from gym import spaces, logger
from gym.utils import seeding
import pandas as pd
import numpy as np

class SmartHome:

	def __init__(self):
		# Initial battery state
		self.home_battery_state = 0
		self.ev_battery_state = 0
		# Initial date and time
		self.month = 1
		self.day = 1
		self.time = 0
		# Initial episode length
		self.max_step_hours = 0
		# Load the data tables and data processing in the ProcessedData class
		self.processed_data = ProcessedData()
		self.__load_data_tables()
		# Define gym actions and env
		self.action_space = spaces.Discrete(5)
		low = np.array([0 for _ in range(5)], dtype=np.float64)
		high = np.array([1 for _ in range(5)], dtype=np.float64)
		self.observation_space = spaces.Box(low, high, dtype=np.float64)


	#####################
	#### GYM METHODS ####
	#####################


	def reset(self): 
		# Generate random home battery and ev battery level
		self.home_battery_state = np.random.rand() * Specifications.HOME_CAPACITY
		self.ev_battery_state = np.random.rand() * Specifications.EV_CAPACITY
		# Generate random datetime
		self.month = np.random.random_integers(1, 12)
		self.day = np.random.random_integers(1, Constant.MONTHS_LEN[self.month-1])
		self.time = np.random.random_integers(0, 23)
		# Reset environment parameters
		self.total_euros_spent = 0
		# Reset learning parameters
		self.max_step_hours = Constant.MAX_STEP_HOURS

		return self.__get_normalized_state()


	def step(self, action): 
		# Define local variables
		reward = 0
		done = False
		current_spent = 0
		info = {'use_ev_fail':0, 'charge_ev_fail':0}

		# Get processed data of the current datetime
		users_demand = self.processed_data.get_H4_consumes(self.month, self.day, self.time)
		pv_potential = self.processed_data.get_PV_entry(self.month, self.day, self.time)
		sg_cost = self.processed_data.get_SG_prices(self.month, self.day, self.time)

		# Case 1: eough energy on the home battery, use it to meet the request
		if self.home_battery_state > users_demand: self.home_battery_state -= users_demand
		# Case 2: not enough energy on the home battery but enough from the PV
		elif pv_potential > users_demand: pv_potential -= users_demand
		# Case 3: take the enrgy from the smart greed
		else: current_spent += (sg_cost * users_demand)

		# Perform Action
		if action == Actions.CHARGE_HOME_wPV: 
			self.home_battery_state += pv_potential
	
		if action == Actions.CHARGE_HOME_wSG: 
			self.home_battery_state += Specifications.MAX_CHARGE_FOR_HOUR
			current_spent += (Specifications.MAX_CHARGE_FOR_HOUR * sg_cost)
			
		if action == Actions.CHARGE_EV_wPV: 
			self.ev_battery_state += pv_potential

		if action == Actions.CHARGE_EV_wSG: 
			self.ev_battery_state += Specifications.MAX_CHARGE_FOR_HOUR
			current_spent += (Specifications.MAX_CHARGE_FOR_HOUR * sg_cost)

		# Fix Values with the max battery capacity
		self.home_battery_state = min(self.home_battery_state, Specifications.HOME_CAPACITY)
		self.ev_battery_state = min(self.ev_battery_state, Specifications.EV_CAPACITY)

		# Negative reward for euros spent
		reward = -current_spent
		# Negative reward if the EV is not ready in the morning
		if self.time == 6 and self.ev_battery_state < Specifications.EV_DAILY_CONSUME: 
			reward -= 10
			info['use_ev_fail'] = 1
		# Negative reward if try to charge the car in the working times
		if self.time >= 6 and self.time <= 18 and (action == Actions.CHARGE_EV_wPV or action == Actions.CHARGE_EV_wSG):
			reward -= 10
			info['charge_ev_fail'] = 1
		
		# Use the daily requirement of the EV at 6am
		if self.time == 6 and self.ev_battery_state >= Specifications.EV_DAILY_CONSUME: self.ev_battery_state -= Specifications.EV_DAILY_CONSUME
		# Compute the total cost of the episode
		self.total_euros_spent += current_spent

		# Increase datetime by 1 hour and fix day and month
		self.max_step_hours -= 1
		self.time += 1
		if self.time > 23: 
			self.time = 0
			self.day += 1
		if self.day > Constant.MONTHS_LEN[self.month-1]:
			self.day = 1
			self.month += 1
		if self.month > 12:
			self.month = 1
			self.day = 1
			self.time = 0

		# Check Done state
		if self.max_step_hours <= 0: done = True

		return self.__get_normalized_state(), reward, done, info


	def seed(self, seed=None): 
		self.np_random, seed = seeding.np_random(seed)
		return [seed]


	#####################
	## PRIVATE METHODS ##
	#####################


	def __load_data_tables(self):
		PV_production_df = pd.read_csv('data/PV.csv', sep=';')
		SG_prices_df = pd.read_csv('data/Prices.csv', sep=';')
		H4_consumes_df = pd.read_csv('data/H4.csv', sep=';')

		PV_production = PV_production_df['P_PV_'].apply(lambda x:  x.replace(',','.')  ).to_numpy( dtype=float )
		SG_prices = SG_prices_df['Price'].apply(lambda x:  x.replace(',','.')  ).to_numpy( dtype=float )
		H4_consumes = H4_consumes_df['Power'].apply(lambda x:  x.replace(',', '.')).to_numpy( dtype=float )

		for el in PV_production: 
			self.processed_data.add_PV_entry( el )
		for el in SG_prices: 
			self.processed_data.add_SG_prices( el / 1000 )
		for i in range(0, H4_consumes.shape[0], 60): 
			el = H4_consumes[i:i+60]
			self.processed_data.add_H4_consumes( sum(el) )


	def __get_normalized_state(self):
		norm_month = (self.month-1) / 11
		norm_day = (self.day-1) / 31
		norm_time = self.time / 23
		norm_home_battery_state = self.home_battery_state / Specifications.HOME_CAPACITY
		norm_ev_battery_state = self.ev_battery_state / Specifications.EV_CAPACITY
		return np.array([norm_month, norm_day, norm_time, norm_home_battery_state, norm_ev_battery_state])

