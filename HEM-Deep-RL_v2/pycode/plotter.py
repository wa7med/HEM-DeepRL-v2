import matplotlib.pyplot as plt
import numpy as np
from collections import deque

class LinePlotter:

	def __init__(self, x_label="X Label", y_label="Y Label", title="No Title"):
		self.fig, self.ax = plt.subplots(1)
		self.ax.spines["top"].set_visible(False)    
		self.ax.spines["bottom"].set_visible(False)    
		self.ax.spines["right"].set_visible(False)    
		self.ax.spines["left"].set_visible(False)  
		self.ax.set_facecolor('#eaeaf2')
		plt.grid(color='#ffffff', linestyle='-', linewidth=1)
		plt.xticks(fontsize=12)
		plt.yticks(fontsize=12)
		plt.xlabel(x_label, fontsize=12)
		plt.ylabel(y_label, fontsize=12)
		plt.title(title, fontsize=16)
		self.data_arrays = []
		self.array_len = -1
		self.mean_array = []
		self.var_array =  []
		self.max_array = []
		self.min_array = []

	def load_array(self, file_name_arrays, early_stop=None):
		data_arrays = [[np.loadtxt(name, delimiter='\n', unpack=True) for name in array_set] for array_set in file_name_arrays]
		if(early_stop == None): self.array_len = min([min([len(el) for el in array_set]) for array_set in data_arrays])
		else: self.array_len = early_stop
		self.data_arrays = np.array([[el[:self.array_len] for el in array_set] for array_set in data_arrays], dtype=object)

	def render(self, labels, colors):
		err_msg = "load some data before the render!"
		assert self.array_len > 0, err_msg
		for mean_values, max_values, min_values, label, color in zip(self.mean_array, self.max_array, self.min_array, labels, colors):
			self.ax.plot(self.x_axes, mean_values, label=label, color=color, linestyle='-', linewidth=1.2 )
			self.ax.fill_between(self.x_axes, max_values, min_values, facecolor=color, alpha=0.3)
		self.ax.legend(loc='lower right', bbox_to_anchor=(1, 0), fontsize=14)
		plt.show()

	def process_data(self, rolling_window=1, starting_pointer=0, early_stop=None):		
		rolling_queue = deque(maxlen=rolling_window)
		self.x_axes = [i for i in range(self.array_len-starting_pointer)]
		for array_set in self.data_arrays:
			for array in array_set:
				for i in range(self.array_len):
					rolling_queue.append(array[i])
					array[i] = np.mean(rolling_queue)
				rolling_queue.clear()
		# Fix for different array size
		self.data_arrays = np.array([np.array(el) for el in self.data_arrays], dtype=object)
		self.mean_array = np.array([[np.mean(array_set[:, i]) for i in range(self.array_len)][starting_pointer:] for array_set in self.data_arrays])
		self.var_array =  np.array([[np.std(array_set[:, i]) for i in range(self.array_len)][starting_pointer:] for array_set in self.data_arrays])
		self.max_array = [[np.max(array_set[:, i]) for i in range(self.array_len)][starting_pointer:] for array_set in self.data_arrays]
		self.min_array = [[np.min(array_set[:, i]) for i in range(self.array_len)][starting_pointer:] for array_set in self.data_arrays]


class HistoPlotter():

	def __init__(self):
		self.trained_list = []
		self.random_list = []

	def plot( self ):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		N = len(self.trained_list)
		ind = np.arange(N)              
		width = 0.35           
		rects1 = ax.bar(ind, self.trained_list, width,
						color='black',
						error_kw=dict(elinewidth=2,ecolor='Red'))
		rects2 = ax.bar(ind+width, self.random_list, width,
							color='red',
							error_kw=dict(elinewidth=2,ecolor='Blue'))
		ax.set_xlim(-width,len(ind)+width)
		ax.set_ylabel('Money Spent (euros)', fontsize=12)
		ax.set_title('Smart Home')
		xTickMarks = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'] 
		ax.set_xticks(ind+width)
		xtickNames = ax.set_xticklabels(xTickMarks)
		plt.setp(xtickNames, rotation=45, fontsize=12)
		ax.legend( (rects1[0], rects2[0]), (f"Trained Network (total={sum(self.trained_list):6.2f})", f"Random Actions (total={sum(self.random_list):6.2f})"), fontsize=14 )
		plt.show()


class DoubleLinePlotter():

	def __init__(self, battery_level=[10, 20, 30], ev_level= [0, 0, 0], price=[30, 20, 10], hourly_charge_sg=[0, 0, 0], hourly_charge_pv=[0, 0, 0] ):
		self.battery_level = battery_level
		self.ev_level = ev_level
		self.charge_sg = hourly_charge_sg
		self.charge_pv = hourly_charge_pv
		self.price = price

	def plot( self ):
		fig, ax = plt.subplots()
		axes = [ax, ax.twinx()]
		fig.subplots_adjust(right=0.86)
		ax.set_xlabel('Time (h)')

		# Plot on Axis 1: Battery consupmtion
		X = [str(i) for i in range(24)]
		axes[0].bar(X, self.charge_sg, color='green', label="from SG")
		axes[0].bar(X, self.charge_pv, color='blue', label="from PV")
		axes[0].set_ylabel("Charge (kWh)", color='black')
		axes[0].tick_params(axis='y', colors='black')
		axes[0].legend(loc='upper right', fontsize=14)

		# Plot on Axis 2: Price
		axes[1].plot(self.price, linestyle='-', color='red', linewidth=1.2)
		axes[1].set_ylabel('Price (kWh)', color='red')
		axes[1].tick_params(axis='y', colors='red')

		# Final Plot
		plt.show()
		