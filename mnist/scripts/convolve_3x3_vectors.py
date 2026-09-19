import math 
import random

SEED = 1
CONVOLUTION_NUM_ELEMENTS = 9

# Convert an int that was interpeted as unsigned
# binary to an int as interpreted as two's complement
def unsigned_to_twos_complement(value, bits):
    # Check if number fits into specified bit width, 
    # return if it doesn't fit
    if ((value >> bits) != 0):
        return

    # Means MSB isn't set, so two's magnitude value
    # equals unsigned binary interpretation
    if ((value >> (bits - 1)) == 0):
        return value
    # MSB is set, so subtract two times the MSB value
    else:
        return value - 2 * pow(2, bits - 1)

# Convert int to two's complement bit sequence
def int_to_twos_complement(value, bits):
    # Check if number fits into specified bit width, 
    # return if it doesn't fit
    max_value =  pow(2, bits - 1) - 1
    min_value = -pow(2, bits - 1)

    if ((value > max_value) or (value < min_value)):
        return

    # Mask with bitwise AND (Python already uses two's magnitude for ints)
    masked_value = value & ((1 << bits) - 1)
    return f"{masked_value:0{bits}b}"

# Pixel values are unsigned binary, weight values are generated as bare ints
# and are converted to the signed/two's complement number system interpretation
def convolve(pixels_list, weights_list):

    sum_list = []
    for i in range(0, len(pixels_list)):
        sum = 0
        for j in range(0, CONVOLUTION_NUM_ELEMENTS):
            # Weight input is 8-bit two's complement so has to be converted
            sum = sum + pixels_list[i][j] * unsigned_to_twos_complement(weights_list[i][j], 8)

        sum_list.append(sum)
    return sum_list

# Generate pixel and weight values
def generate_vectors(rng, n):
    pixels_list  = []
    weights_list = []

    for i in range(0, n):
        pixels  = []
        weights = []

        for j in range(0, CONVOLUTION_NUM_ELEMENTS):
            pixels.append(rng.randint(0,255))
            weights.append(rng.randint(0,255))

        pixels_list.append(pixels)
        weights_list.append(weights)

    return pixels_list, weights_list

def write_convolve_vectors(path, rng, n):
    pixels_list, weights_list = generate_vectors(rng, n)
    sum_list = convolve(pixels_list, weights_list)

    with open(path, "w") as f:
        for pixels, weights, result in zip(pixels_list, weights_list, sum_list):
            # Joins together all pixel bits and then the weight bits so tb can read easily
            pixel_bits = "".join(f"{pixel:08b}" for pixel in pixels)
            weight_bits = "".join(f"{weight:08b}" for weight in weights)
            result_bits = int_to_twos_complement(result, 20)
            f.write(f"{pixel_bits} {weight_bits} {result_bits}\n")

write_convolve_vectors("tb/vectors/convolve_vectors.txt", random.Random(SEED), 100000)