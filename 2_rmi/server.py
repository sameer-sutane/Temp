import Pyro4

@Pyro4.expose
class StringService:
    def concatenate(self, s1, s2):
        return s1 + s2

# Start the Pyro daemon and register the object
daemon = Pyro4.Daemon()
uri = daemon.register(StringService)
print("Server is running. URI:", uri)
daemon.requestLoop()
