import random

SEED = 1
NUM_RANDOM_VECTORS = 100000

TILE_SIZE   = 4
KERNEL_SIZE = 3
OUTPUT_SIZE = 2

PIXEL_BITS  = 8
U_BITS      = 12
RESULT_BITS = 20

# G' = 2G, so U' = G' g G'^T = 4 * (G g G^T) stays an integer
G2 = [[2,  0, 0],
      [1,  1, 1],
      [1, -1, 1],
      [0,  0, 2]]

# Winograd F(2x2, 3x3) input and output transforms
BT = [[1,  0, -1,  0],
      [0,  1,  1,  0],
      [0, -1,  1,  0],
      [0,  1,  0, -1]]

AT = [[1, 1,  1,  0],
      [0, 1, -1, -1]]

def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]

def transpose(a):
    return [list(row) for row in zip(*a)]

# Convert int to two's complement bit sequence, erroring if it doesn't fit
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

# Reference: direct 3x3 convolution (same pairing as convolve_3x3) over a 4x4 tile
def convolve_tile(tile, weights):
    outputs = []
    for r in range(OUTPUT_SIZE):
        for c in range(OUTPUT_SIZE):
            sum = 0
            for i in range(KERNEL_SIZE):
                for j in range(KERNEL_SIZE):
                    sum = sum + tile[(r+i)*TILE_SIZE + (c+j)] * weights[i*KERNEL_SIZE + j]
            outputs.append(sum)
    return outputs

# Winograd model of the RTL, used to check the math before simulating
def winograd_tile(tile, u):
    d = [tile[r*TILE_SIZE:(r+1)*TILE_SIZE] for r in range(TILE_SIZE)]
    v = matmul(matmul(BT, d), transpose(BT))
    m = [[u[r*TILE_SIZE + c] * v[r][c] for c in range(TILE_SIZE)] for r in range(TILE_SIZE)]
    y = matmul(matmul(AT, m), transpose(AT))
    return [value >> 2 for row in y for value in row]

# Extremes that stress the widest intermediate values
def corner_vectors():
    checker = [255 if (r + c) % 2 == 0 else 0 for r in range(TILE_SIZE) for c in range(TILE_SIZE)]
    signed  = [127 if (r + c) % 2 == 0 else -128 for r in range(KERNEL_SIZE) for c in range(KERNEL_SIZE)]
    return [
        ([0]   * 16, [0]    * 9),
        ([255] * 16, [127]  * 9),
        ([255] * 16, [-128] * 9),
        ([255] * 16, signed),
        (checker,    signed),
        (checker,    [-128] * 9),
    ]

def generate_vectors(rng, n):
    vectors = corner_vectors()
    for i in range(0, n):
        tile    = [rng.randint(0, 255)    for j in range(TILE_SIZE * TILE_SIZE)]
        weights = [rng.randint(-128, 127) for j in range(KERNEL_SIZE * KERNEL_SIZE)]
        vectors.append((tile, weights))
    return vectors

def write_winograd_vectors(path, rng, n):
    with open(path, "w") as f:
        for tile, weights in generate_vectors(rng, n):
            u        = transform_kernel(weights)
            expected = convolve_tile(tile, weights)
            assert winograd_tile(tile, u) == expected, f"Winograd mismatch: {tile} {weights}"

            # Fields: 16 pixels, 16 U' values, 4 expected outputs, each concatenated row-major
            tile_bits   = "".join(f"{pixel:0{PIXEL_BITS}b}" for pixel in tile)
            u_bits      = "".join(int_to_twos_complement(value, U_BITS) for value in u)
            result_bits = "".join(int_to_twos_complement(value, RESULT_BITS) for value in expected)
            f.write(f"{tile_bits} {u_bits} {result_bits}\n")

write_winograd_vectors("tb/vectors/winograd_vectors.txt", random.Random(SEED), NUM_RANDOM_VECTORS)
