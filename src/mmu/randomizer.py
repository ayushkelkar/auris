import struct
import random
import math

num_vectors = 10000
dims = 32

def normalize(v):
    mag = math.sqrt(sum(x*x for x in v))
    return [x/mag for x in v]

def quantize(v):
    return [max(-127, min(127, int(round(x * 127)))) for x in v]

with open("mockdb.bin", "wb") as f:
    f.write(struct.pack("ii", num_vectors, dims))
    for _ in range(num_vectors):
        vec = [random.gauss(0, 1) for _ in range(dims)]
        vec = normalize(vec)
        vec = quantize(vec)
        f.write(struct.pack(f"{dims}b", *vec))

print(f"Written {num_vectors} vectors of {dims} dims to mockdb.bin")