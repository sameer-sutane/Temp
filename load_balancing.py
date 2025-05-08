# Define servers and their load
server1 = 0
server2 = 0
server3 = 0

# List of client requests (10 clients)
clients = ["Client1", "Client2", "Client3", "Client4", "Client5",
           "Client6", "Client7", "Client8", "Client9", "Client10"]

print("=== Round Robin Load Balancing ===")
count = 1
for client in clients:
    if count == 1:
        print(client, "→ Server1")
    elif count == 2:
        print(client, "→ Server2")
    elif count == 3:
        print(client, "→ Server3")
    count += 1
    if count > 3:
        count = 1

print("\n=== Least Connections Load Balancing ===")
# Reset server loads
server1 = 0
server2 = 0
server3 = 0

for client in clients:
    # Choose server with least connections
    if server1 <= server2 and server1 <= server3:
        print(client, "→ Server1")
        server1 += 1
    elif server2 <= server1 and server2 <= server3:
        print(client, "→ Server2")
        server2 += 1
    else:
        print(client, "→ Server3")
        server3 += 1
