from requests import get, RequestException

# shelly pro 1pm switching the light, controlled over its local rpc api
# a failing light never stops the capture, it is only logged
class Light:
	timeout = 2 # s

	def __init__(self, address):
		self.address = address

	def set(self, on):
		try:
			response = get('http://' + self.address + '/rpc/Switch.Set', params={ 'id': 0, 'on': 'true' if on else 'false' }, timeout=self.timeout)
			response.raise_for_status()

			print('[light] ' + ('on' if on else 'off'))
		except RequestException as error:
			print('[light] switching ' + ('on' if on else 'off') + ' failed: ' + str(error))
