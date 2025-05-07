"""Design a distributed application using RPC for remote computation where client submits an
integer value to the server and server calculates factorial and returns the result to the client
program """

import xmlrpc.client

# Connect to server
proxy = xmlrpc.client.ServerProxy("http://localhost:8000/")

# Input from user
n = int(input("Enter an integer to compute factorial: "))

# Remote call
result = proxy.compute_factorial(n)

print(f"Factorial of {n} is: {result}")
