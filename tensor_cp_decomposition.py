import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac

tl.set_backend("numpy")

# 1. Setup our Dimensions
n_members = 100
n_metrics = 3  # [Hard Braking, Speeding, Sharp Turns]
n_times   = 3  # [Morning Commute, Mid-day, Late Night]

# 2. Create Synthetic Data (The "Insurance Cube")
# Imagine this is populated from your telematics API
np.random.seed(42)
data_cube = tl.tensor(np.random.rand(n_members, n_metrics, n_times), dtype=tl.float32)

# 3. Apply CP Decomposition 
# We want to find 2 distinct "Underwriting Cohorts" (Rank=2)
weights, factors = parafac(data_cube, rank=2, init='random', normalize_factors=True)

member_factors = factors[0]  # Member weights for each cohort
metric_factors = factors[1]  # Which driving behaviors define the cohort
time_factors   = factors[2]  # Which times of day define the cohort

# 4. Interpret the Results for Underwriting
print("--- Cohort 1: Driving Behavior Signature ---")
print(f"Metrics (Braking/Speeding/Turns): {metric_factors[:, 0]}")
print(f"Time of Day (Morning/Day/Night): {time_factors[:, 0]}")

print("\n--- Cohort 2: Driving Behavior Signature ---")
print(f"Metrics (Braking/Speeding/Turns): {metric_factors[:, 1]}")
print(f"Time of Day (Morning/Day/Night): {time_factors[:, 1]}")

dims = {
    "members": 100,
    "metrics": 3,  # Braking, Speeding, Turning
    "times": 3,    # Morning, Afternoon, Night
    "cities": 5,   # New York, Austin, Chicago, etc.
    "vehicles": 4  # Sedan, SUV, Sports, Truck
}

# 2. Create the 5D Hyper-cube
shape = tuple(dims.values())
data_hypercube = np.random.rand(*shape)

# 3. Perform CP Decomposition
# We look for 3 multi-way "Underwriting Themes"
rank = 3
weights, factors = parafac(data_hypercube, rank=rank, init='random')

# 4. Extract the new dimensions
city_factors = factors[3]
vehicle_factors = factors[4]

# Let's look at Theme #1
theme_idx = 0
print(f"--- Underwriting Theme #{theme_idx + 1} ---")
print(f"Top City Impact: {np.argmax(city_factors[:, theme_idx])}")
print(f"Top Vehicle Type Impact: {np.argmax(vehicle_factors[:, theme_idx])}")
