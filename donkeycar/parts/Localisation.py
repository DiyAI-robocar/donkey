import socket


class BaseLocalisation:
    def run_threaded(self):
    	# ToDo parse the values!!!
        return self.data


class Localisation(BaseLocalisation):
	"""
	Calss to recieve the longitude and latitude.
	"""
	def __init__(self):
		self.udp_ip = '127.0.0.1'
		self.udp_socket = 5005
		sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
   		sock.bind((self.udp_ip, self.udp_socket))

		self.on = True

	def run(self):
		pass

	def update(self):
		while self.on:
			self.data, self.addr = self.sock.recvfrom(1024)

	def shutdown(self):
		self.on = False
		self.sock.close()