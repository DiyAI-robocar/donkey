import requests
import json
import time


class MapMatching:
	"""
	Class to update the map file 
	"""
	def __init__(self):
		self.github_file = "https:/github.com/path/to/the/file"
		self.on = True
		self.map = None

	def run(self):
		pass

	def update(self):
		while self.on:
			self.response = requests.get(self.github_file).content()
			self.map = json.load(self.response)

			time.sleep(10)

	def run_threaded(self):
		return self.map

	def shutdown(self):
		self.on = False
