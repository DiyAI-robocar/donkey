import requests
import json


class MapMatching:
	"""
	Class to uppdate the map file 
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

			sleep(10)

	der run_threaded(self):
		return self.map

	def shutdown(self):
		self.on = False
