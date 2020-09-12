import time


def print_header(my_str, char='*'):
	"""Print header to seperate output
	str - text to print
	char - the charater repeated on next line for pizzaz"""
	print(f"{my_str}\n{char * len(my_str)}\n")


class MyTimer:
	"""Class to calculate passage of time.
	This class does not take into consideration background CPU use or any nuances.
	start timer with call to tic, print elapsed time with call to toc"""

	def __init__(self, memo=''):
		self.start_time = time.time()
		self.memo = memo

	def tic(self):
		self.start_time = time.time()

	def toc(self):
		self.end_time = time.time()
		print(f"Elapsed number of seconds: *{self.end_time-self.start_time}* while in *{self.memo}*")
