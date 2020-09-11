import time as time

class Timer():
	def __init__(self):
		self.times = {}
		self.depth = 0
		self.depths = {}
	def start_time(self,name):
		try: self.times[name].append(time.process_time())
		except:
			self.times[name] = [time.process_time()]
			self.depths[name] = self.depth
		self.depth += 1
	def end_time(self,name):
		self.times[name][-1] = time.process_time() - self.times[name][-1]
		self.depth -= 1
	def print_summary(self):
		for n, l in self.times.items():
			spacer = "".join([" "]*self.depths[n])
			if len(l) == 1:
				print ("{0}{1}:{2:.2f}s".format(spacer, n, sum(l)))
			else:
				print ("{0}{1}:{2:.2f}s, averaging {3:.2f} per iteration".format(spacer, n, sum(l), sum(l)/len(l)))
	def summary(self):
		sum_str = ""
		for n, l in self.times.items():
			spacer = "".join([" "]*self.depths[n])
			if len(l) == 1:
				line = "{0}{1}:{2:.2f}s\n".format(spacer, n, sum(l))
			else:
				line = "{0}{1}:{2:.2f}s, averaging {3:.2f} per iteration\n".format(spacer, n, sum(l), sum(l)/len(l))
			sum_str += line
		return sum_str
