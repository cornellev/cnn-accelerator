import os
import random

SEED = 1
NUM_RANDOM_VECTORS = 100000

KERNEL_SIZE = 3
TILE_SIZE   = 4

WEIGHT_BITS = 8
U_BITS      = 12

# Same G' = 2G as winograd_3x3_vectors.py and quantize_custom.py
G2 = [[2,  0, 0],
      [1,  1, 1],
      [1, -1, 1],
      [0,  0, 2]]

def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]

def transpose(a):
    return [list(row) for row in zip(*a)]

# Convert int to two's complement bit sequence
def int_to_twos_complement(value, bits):
    max_value =  pow(2, bits - 1) - 1
    min_value = -pow(2, bits - 1)
    assert min_value <= value <= max_value, f"{value} does not fit in {bits} bits"

    masked_value = value & ((1 << bits) - 1)
    return f"{masked_value:0{bits}b}"

# Transformed kernel U' = G' g G'^T, flattened row-major (16 values)
def transform_kernel(weights):
    g = [weights[r*KERNEL_SIZE:(r+1)*KERNEL_SIZE] for r in range(KERNEL_SIZE)]
    u = matmul(matmul(G2, g), transpose(G2))
    return [value for row in u for value in row]

# Extremes: largest |U'| and every sign pattern of +/-1 combinations
def corner_vectors():
    signed = [127 if (r + c) % 2 == 0 else -128 for r in range(KERNEL_SIZE) for c in range(KERNEL_SIZE)]
    return [
        [0]    * 9,
        [127]  * 9,
        [-128] * 9,
        signed,
        [-value - 1 for value in signed],
    ] + [[-128 if i == j else 0 for i in range(9)] for j in range(9)]

def generate_vectors(rng, n):
    vectors = corner_vectors()
    for i in range(0, n):
        vectors.append([rng.randint(-128, 127) for j in range(KERNEL_SIZE * KERNEL_SIZE)])
    return vectors

def write_weight_transform_vectors(path, rng, n):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for weights in generate_vectors(rng, n):
            u = transform_kernel(weights)

            # Fields: 9 weights, 16 expected U' values, each concatenated row-major
            weight_bits = "".join(int_to_twos_complement(value, WEIGHT_BITS) for value in weights)
            u_bits      = "".join(int_to_twos_complement(value, U_BITS) for value in u)
            f.write(f"{weight_bits} {u_bits}\n")

MNIST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
write_weight_transform_vectors(os.path.join(MNIST_DIR, "tb", "vectors", "weight_transform_vectors.txt"),
                               random.Random(SEED), NUM_RANDOM_VECTORS)
