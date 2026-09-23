#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include "../include/network.h"
#include "../include/activation.h"
#include "../include/cost.h"
#include "../include/initializers.h"
#include "../include/training.h"
#include "../include/evaluation.h"

enum idx_data_type_t {
    IDX_UNSIGNED_BYTE = 0x08,
    IDX_SIGNED_BYTE = 0x09,
    IDX_SHORT = 0x0B,
    IDX_INT = 0x0C,
    IDX_FLOAT = 0x0D,
    IDX_DOUBLE = 0x0E
};

struct idx_dataset_t {
    uint8_t data_type;
    uint8_t dimension_count;
    uint32_t* dimensions;
    void* data;
};

size_t idx_data_type_size(enum idx_data_type_t data_type) {
    switch (data_type) {
        case IDX_UNSIGNED_BYTE:
        case IDX_SIGNED_BYTE:
            return 1;
        case IDX_SHORT:
            return 2;
        case IDX_INT:
        case IDX_FLOAT:
            return 4;
        case IDX_DOUBLE:
            return 8;
        default:
            return 0;
    }
}

void convert_big_endian_to_native(void* data, size_t item_size, size_t item_count) {
    for (size_t i = 0; i < item_count; i++) {
        uint8_t* item = (uint8_t*)data + i * item_size;
        for (size_t a = 0, b = item_size - 1; a < b; a++, b--) {
            uint8_t temp = item[a];
            item[a] = item[b];
            item[b] = temp;
        }
    }
}

int idx_read_file(const char* path, struct idx_dataset_t* dataset) {
    FILE* file = fopen(path, "rb");
    if (file == NULL) return 0;

    fseek(file, 2, SEEK_SET);
    fread(&dataset->data_type, 1, 1, file);
    fread(&dataset->dimension_count, 1, 1, file);
    dataset->dimensions = (uint32_t*)malloc(dataset->dimension_count * sizeof(uint32_t));
    fread(dataset->dimensions, sizeof(uint32_t), dataset->dimension_count, file);
    convert_big_endian_to_native(dataset->dimensions, sizeof(uint32_t), dataset->dimension_count);

    size_t item_size = idx_data_type_size(dataset->data_type);
    size_t item_count = dataset->dimensions[0];
    for (uint8_t i = 1; i < dataset->dimension_count; i++) {
        item_count *= dataset->dimensions[i];
    }

    dataset->data = malloc(item_count * item_size);
    fread(dataset->data, item_size, item_count, file);
    convert_big_endian_to_native(dataset->data, item_size, item_count);

    fclose(file);
    return 1;
}

void idx_destroy_dataset(struct idx_dataset_t* dataset) {
    free(dataset->dimensions);
    free(dataset->data);
}

float measure_accuracy(struct tinynn_evaluation_ctx_t* evaluation_ctx, const struct idx_dataset_t* testing_images, const struct idx_dataset_t* testing_labels) {
    uint32_t pixel_count = evaluation_ctx->network->layout.input_node_count;
    float* test_input = (float*)malloc(pixel_count * sizeof(float));
    float test_output[10];

    uint8_t* test_image = (uint8_t*)testing_images->data;
    uint32_t number_correct = 0;
    for (uint32_t test_index = 0; test_index < testing_images->dimensions[0]; test_index++) {
        for (uint32_t i = 0; i < pixel_count; i++) {
            test_input[i] = (float)(*(test_image++)) / 255.0f;
        }
        tinynn_evaluate(evaluation_ctx, test_input, test_output);

        uint8_t prediction = 0;
        for (uint8_t i = 1; i < 10; i++) {
            if (test_output[i] > test_output[prediction]) {
                prediction = i;
            }
        }

        if (prediction == *((uint8_t*)testing_labels->data + test_index)) {
            number_correct++;
        }
    }

    free(test_input);
    return (float)number_correct / (float)testing_images->dimensions[0];
}

int save_neural_network(const struct tinynn_network_t* network, const char* path) {
    FILE* save_file = fopen(path, "wb");
    if (save_file == NULL) return 0;

    fwrite(&network->layout.input_node_count, sizeof(network->layout.input_node_count), 1, save_file);
    fwrite(&network->layout.layer_count, sizeof(network->layout.layer_count), 1, save_file);
    for (uint32_t i = 0; i < network->layout.layer_count; i++) {
        uint32_t layer_size = network->layout.layers[i].node_count;
        fwrite(&layer_size, sizeof(layer_size), 1, save_file);
    }

    fwrite(network->biases, sizeof(float), network->bias_count, save_file);
    fwrite(network->weights, sizeof(float), network->weight_count, save_file);

    fclose(save_file);
    return 1;
}

int main() {
    int status = 0;

    struct idx_dataset_t training_images, training_labels, testing_images, testing_labels;
    if (!idx_read_file("../datasets/mnist_digits/training_images", &training_images)) {
        printf("failed to read training images\n");
        status = -1;
        goto done;
    }
    if (!idx_read_file("../datasets/mnist_digits/training_labels", &training_labels)) {
        printf("failed to read training labels\n");
        status = -1;
        goto destroy_training_images;
    }
    if (!idx_read_file("../datasets/mnist_digits/testing_images", &testing_images)) {
        printf("failed to read testing images\n");
        status = -1;
        goto destroy_training_labels;
    }
    if (!idx_read_file("../datasets/mnist_digits/testing_labels", &testing_labels)) {
        printf("failed to read testing labels\n");
        status = -1;
        goto destroy_testing_images;
    }
    if (
        training_images.data_type != IDX_UNSIGNED_BYTE ||
        training_images.dimension_count != 3 ||
        training_labels.data_type != IDX_UNSIGNED_BYTE ||
        training_labels.dimension_count != 1 ||
        training_labels.dimensions[0] != training_images.dimensions[0] ||
        testing_images.data_type != IDX_UNSIGNED_BYTE ||
        testing_images.dimension_count != 3 ||
        testing_images.dimensions[1] != training_images.dimensions[1] ||
        testing_images.dimensions[2] != training_images.dimensions[2] ||
        testing_labels.data_type != IDX_UNSIGNED_BYTE ||
        testing_labels.dimension_count != 1 ||
        testing_labels.dimensions[0] != testing_images.dimensions[0]
    ) {
        printf("invalid dataset format\n");
        status = -1;
        goto destroy_testing_labels;
    }

    struct tinynn_network_t network;
    tinynn_create_network(&network, (struct tinynn_network_layout_t){
        .input_node_count = training_images.dimensions[1] * training_images.dimensions[2],
        .layer_count = 3,
        .layers = (struct tinynn_layer_t[]){
            {
                .activation = &TINYNN_ACTIVATION_SIGMOID,
                .node_count = 64
            },
            {
                .activation = &TINYNN_ACTIVATION_SIGMOID,
                .node_count = 32
            },
            {
                .activation = &TINYNN_ACTIVATION_SIGMOID,
                .node_count = 10
            }
        }
    });
    //tinynn_init_params_random_normalized(&network, time(NULL));
    FILE* save_file = fopen("digit_recognition.bin", "rb");
    if (save_file == NULL) {
        printf("failed to open nn save file\n");
        status = -1;
        goto destroy_network;
    }
    fseek(save_file, sizeof(network.layout.input_node_count) + sizeof(network.layout.layer_count) + sizeof(uint32_t) * network.layout.layer_count, SEEK_SET);
    fread(network.biases, sizeof(float), network.bias_count, save_file);
    fread(network.weights, sizeof(float), network.weight_count, save_file);
    fclose(save_file);

    struct tinynn_training_ctx_t training_ctx;
    tinynn_create_training_ctx(&training_ctx, &network);

    size_t element_count = training_images.dimensions[0] * network.layout.input_node_count;
    float* training_inputs = (float*)malloc(element_count * sizeof(float));
    for (size_t i = 0; i < element_count; i++) {
        uint8_t element = *((uint8_t*)training_images.data + i);
        training_inputs[i] = (float)element / 255.0f;
    }

    float* training_outputs = (float*)calloc(training_labels.dimensions[0] * 10, sizeof(float));
    for (size_t i = 0; i < training_labels.dimensions[0]; i++) {
        uint8_t label = *((uint8_t*)training_labels.data + i);
        training_outputs[i * 10 + label] = 1.0f;
    }

    struct tinynn_training_params_t training_params;
    training_params.cost = TINYNN_COST_QUADRATIC;
    training_params.learning_rate = 5.0f;

    //tinynn_train(&training_ctx, training_params, training_images.dimensions[0], training_inputs, training_outputs, 100, 1);

    float max_weight = 0.0f;
    float max_bias = 0.0f;
    for (uint32_t i = 0; i < network.weight_count; i++) {
        float value = network.weights[i];
        if (value < 0.0f) value = -value;
        if (value > max_weight) max_weight = value;
    }
    for (uint32_t i = 0; i < network.bias_count; i++) {
        float value = network.biases[i];
        if (value < 0.0f) value = -value;
        if (value > max_bias) max_bias = value;
    }
    printf("maximum weight = %f, maximum bias = %f\n", max_weight, max_bias);

    printf("accuracy on training data: %.2f%%\n", measure_accuracy(&training_ctx.evaluation_ctx, &training_images, &training_labels) * 100.0f);
    printf("accuracy on test data: %.2f%%\n", measure_accuracy(&training_ctx.evaluation_ctx, &testing_images, &testing_labels) * 100.0f);
    //save_neural_network(&network, "digit_recognition.bin");

    //free_training_outputs:
        free(training_outputs);
    //free_training_inputs:
        free(training_inputs);
    //destroy_training_ctx:
        tinynn_destroy_training_ctx(&training_ctx);
    destroy_network:
        tinynn_destroy_network(&network);
    destroy_testing_labels:
        idx_destroy_dataset(&testing_labels);
    destroy_testing_images:
        idx_destroy_dataset(&testing_images);
    destroy_training_labels:
        idx_destroy_dataset(&training_labels);
    destroy_training_images:
        idx_destroy_dataset(&training_images);
    done:
        return status;
}