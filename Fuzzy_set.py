from tabulate import tabulate

# Fuzzy Sets (as simple dictionaries with elements and their membership values)
A = {'x': 0.6, 'y': 0.8}
B = {'x': 0.4, 'y': 0.5}

# 1. Union of two fuzzy sets (max value for each element)
union = {key: max(A[key], B[key]) for key in A}

# 2. Intersection of two fuzzy sets (min value for each element)
intersection = {key: min(A[key], B[key]) for key in A}

# 3. Complement of a fuzzy set (1 - value for each element)
complement_A = {key: 1 - A[key] for key in A}

# 4. Difference of two fuzzy sets (min of A and (1 - B) for each element)
difference = {key: min(A[key], 1 - B[key]) for key in A}

# --- Fuzzy Relations ---

# 5. Cartesian product of fuzzy sets A and B (combining each pair of elements from A and B)
R1 = {(a, b): min(A[a], B[b]) for a in A for b in B}

# 6. Cartesian product of B and A for the second relation
R2 = {(b, a): min(B[b], A[a]) for b in B for a in A}

# Max-Min composition of R1 and R2
composition = {}
for (a, b1) in R1:
    for (b2, c) in R2:
        if b1 == b2:  # Match the middle element
            key = (a, c)
            composition[key] = max(composition.get(key, 0), min(R1[(a, b1)], R2[(b2, c)]))

# ---- Output using tabulate ----

print("\nUnion of A and B:")
print(tabulate(union.items(), headers=["Element", "Membership"]))

print("\nIntersection of A and B:")
print(tabulate(intersection.items(), headers=["Element", "Membership"]))

print("\nComplement of A:")
print(tabulate(complement_A.items(), headers=["Element", "Membership"]))

print("\nDifference of A and B:")
print(tabulate(difference.items(), headers=["Element", "Membership"]))

print("\nCartesian Product of A and B (R1):")
print(tabulate(R1.items(), headers=["(A,B)", "Min(A,B)"]))

print("\nCartesian Product of B and A (R2):")
print(tabulate(R2.items(), headers=["(B,A)", "Min(B,A)"]))

print("\nMax-Min Composition of R1 and R2:")
print(tabulate(composition.items(), headers=["(A,C)", "Value"]))
