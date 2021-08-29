from pycode.constant import *

class ProcessedData:
	def __init__(self):
		self.PV_production = []
		self.SG_prices = []
		self.H4_consumes = []

	# Methods to add the elements in the specific arrays
	def add_PV_entry(self, el): self.PV_production.append( el )
	def add_SG_prices(self, el): self.SG_prices.append( el )
	def add_H4_consumes(self, el): self.H4_consumes.append( el )

	# Get Methods (given month and time return the value)
	def get_PV_entry(self, month, day, day_time): return self.PV_production[ (sum(Constant.MONTHS_LEN[:month-1]) + day - 1) * 24  + day_time ]
	def get_SG_prices(self, month, day, day_time): return self.SG_prices[ (sum(Constant.MONTHS_LEN[:month-1]) + day - 1) * 24  + day_time ]
	def get_H4_consumes(self, month, day, day_time): return self.H4_consumes[ (sum(Constant.MONTHS_LEN[:month-1]) + day - 1) * 24  + day_time ]