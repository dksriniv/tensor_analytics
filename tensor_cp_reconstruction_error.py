import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac

# 1. Setup: Historic Model
# 50 members, 3 metrics, 3 time slots
normal_data = np.random.rand(50, 3, 3) 
rank = 2
weights, factors = parafac(normal_data, rank=rank)

# 2. New Observation (e.g., a new trip) 
# Shape must be (metrics, times) -> (3, 3)
new_trip = np.random.rand(3, 3) 

# 3. The "Why" & "How": Projection
# We project the new trip onto the Metric and Time factors
# This tells us: "How does this trip fit into our known patterns?"
# We use the Pseudo-Inverse to find the best fit
m_factor = factors[1] # (3, rank)
t_factor = factors[2] # (3, rank)

# Solve for the 'latent signature' of this specific trip
# New_Trip ≈ M_factor * Signature * T_factor.T
# We use kronecker products or simple pinv for small 2D slices:
k_prod = np.kron(m_factor, t_factor) # Combining known patterns
flat_trip = new_trip.flatten()
signature = np.linalg.pinv(k_prod) @ flat_trip # This is the "Identity" of this trip

# 4. Reconstruct and Calculate Error
reconstructed_flat = k_prod @ signature
reconstructed_trip = reconstructed_flat.reshape(3, 3)

error = np.linalg.norm(new_trip - reconstructed_trip)
print(f"Reconstruction Error: {error:.4f}")

# Real-time prediction with the Tensor Contraction
# 1. Dimensions: Metrics=3, TimeSteps=3, Cities=5
# Weights represent the "Risk Intensity" of each combination
risk_weights = np.random.rand(3, 3, 5) 

# 2. Live Stream: (Metrics x Time)
# We need to make sure we have 3 metrics to match our weights
live_stream = np.array([
    [0.8, 0.2, 0.1], # Metric 1: Braking over 3 seconds
    [0.1, 0.1, 0.9], # Metric 2: Speeding over 3 seconds
    [0.0, 0.4, 0.2]  # Metric 3: Turning over 3 seconds
])

# 3. Corrected Contraction
# 'mt' = Metric, Time (Live Stream)
# 'mtc' = Metric, Time, City (Risk Weights)
# Result 'c' = Risk score per City
risk_scores = np.einsum('mt,mtc->c', live_stream, risk_weights)

print("--- Real-time Risk Assessment ---")
cities = ["New York", "Austin", "Chicago", "Miami", "Seattle"]
for i, city in enumerate(cities):
    print(f"{city}: {risk_scores[i]:.2f}")