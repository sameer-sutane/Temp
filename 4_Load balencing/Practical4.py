import random, time

class Server:
    def __init__(self, name):
        self.name = name
        self.conn = 0

def simulate(servers, use_least_conn=False):
    for i in range(5):
        s = min(servers, key=lambda x: x.conn) if use_least_conn else servers[i % len(servers)]
        print(f"Client {i+1} routed to {s.name}")
        time.sleep(2)  # Simulate handling the request

def main():
    servers = [Server(f"Server {i+1}") for i in range(3)]
    print("\n--- Round Robin Load Balancing ---")
    simulate(servers)
    print("\n--- Least Connections Load Balancing ---")
    simulate(servers, True)

if __name__ == "__main__":
    main()
