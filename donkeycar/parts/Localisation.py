import socket


class Localisation:
	"""
	Calss to recieve the longitude and latitude.
	"""
	def __init__(self):
		self.udp_ip = '127.0.0.1'
		self.udp_socket = 5005
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
   		self.sock.bind((self.udp_ip, self.udp_socket))

		self.on = True

		self.data = None
		self.addr = None

	def update(self):
		while self.on:
			self.data, self.addr = self.sock.recvmsg(1024)

    def run_threaded(self):
    	# ToDo parse the values!!!
        return self.data

	def shutdown(self):
		self.on = False
		self.sock.close()