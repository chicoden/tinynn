import numpy as np
from pyperclip import copy

with open("../../digit_classifier.bin", "rb") as file:
    input_node_count, layer_count = np.frombuffer(file.read(8), dtype=np.uint32)
    layer_sizes = np.frombuffer(file.read(layer_count * 4), dtype=np.uint32)
    biases_flat = np.frombuffer(file.read(sum(layer_sizes) * 4), dtype=np.float32)
    weights_flat = np.frombuffer(file.read(), dtype=np.float32)

biases = []
for layer_size in layer_sizes:
    bias_vector, biases_flat = biases_flat[:layer_size], biases_flat[layer_size:]
    biases.append(bias_vector)

prev_layer_size = input_node_count
weights = []
for this_layer_size in layer_sizes:
    weight_count = prev_layer_size * this_layer_size
    weight_matrix, weights_flat = weights_flat[:weight_count], weights_flat[weight_count:]
    weights.append(weight_matrix.reshape((this_layer_size, prev_layer_size)))
    prev_layer_size = this_layer_size

biases = [[float(b) for b in bias_vector] for bias_vector in biases]
weights = [[[float(w) for w in row] for row in weight_matrix] for weight_matrix in weights]

copy(f"const weights = {weights};\nconst biases = {biases};\n")
