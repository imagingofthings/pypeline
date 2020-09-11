import time as time

class Timer():
	def __init__(self):
		self.times = {}
		self.depth = 0
		self.depths = {}
		self.ops = {}

	def set_Nops(self,name,n):
		self.ops[name] = n

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
				print ("{0}{1}: {2:.2f}s".format(spacer, n, sum(l)))
			else:
				print ("{0}{1}: {2:.2f}s, averaging {3:.2f} per iteration for {4} iterations".format(spacer, n, sum(l), sum(l)/len(l),len(l)))
				if n in self.ops:
					print("{0}{1} operations per iteration for {2:.2f} ops/s".format("".join([" "]*len(line_title)),self.ops[n],len(l)*self.ops[n]/sum(l)))
	def summary(self):
		sum_str = ""
		for n, l in self.times.items():
			spacer = "".join([" "]*self.depths[n])
			if len(l) == 1:
				line = "{0}{1}: {2:.2f}s\n".format(spacer, n, sum(l))
			else:
				line_title = "{0}{1}".format(spacer, n)
				line = "{0}: {1:.2f}s, averaging {2:.2f} per iteration for {3} iterations\n".format(line_title, sum(l), sum(l)/len(l),len(l))
				if n in self.ops:
					line += "{0}{1} operations per iteration for {2:.2f} ops/s\n".format("".join([" "]*len(line_title)),self.ops[n],len(l)*self.ops[n]/sum(l))
			sum_str += line
		return sum_str
