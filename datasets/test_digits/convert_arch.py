import numpy as np
from pyperclip import copy
import sys

if len(sys.argv) != 2:
    print("have not specified the path of the architecture to convert")
    exit()

with open(sys.argv[1], "rb") as file:
    input_node_count, layer_count = np.frombuffer(file.read(8), dtype=np.uint32)
    layers = np.frombuffer(file.read(layer_count * 8), dtype=np.uint32).reshape((layer_count, 2))
    layer_sizes = layers[:, 0]
    biases_flat = np.frombuffer(file.read(sum(layer_sizes) * 4), dtype=np.float32)
    weights_flat = np.frombuffer(file.read(), dtype=np.float32)

biases = []
for layer_size in layer_sizes:
    bias_vector, biases_flat = biases_flat[:layer_size], biases_flat[layer_size:]
    biases.append(bias_vector)

weights = []
prev_layer_size = input_node_count
for this_layer_size in layer_sizes:
    weight_count = prev_layer_size * this_layer_size
    weight_matrix, weights_flat = weights_flat[:weight_count], weights_flat[weight_count:]
    weights.append(weight_matrix.reshape((this_layer_size, prev_layer_size)))
    prev_layer_size = this_layer_size

class label:
    def __init__(self, label):
        self.label = label
    def __repr__(self):
        return self.label
activations = [label("sigmoid"), label("softmax")]

arch = []
for bias_vector, weight_matrix, activation_id in zip(biases, weights, layers[:, 1]):
    arch.append([
        [float(b) for b in bias_vector],
        [[float(w) for w in row] for row in weight_matrix],
        activations[activation_id]
    ])

copy(f"const layers = {arch};")