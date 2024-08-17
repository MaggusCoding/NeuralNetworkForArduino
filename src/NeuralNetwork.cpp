#include "NeuralNetwork.h"

void NeuralNetwork::add_layer(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
    std::vector<double> current = input;
    for (auto& layer : layers) {
        current = layer->forward(current);
    }
    return current;
}

void NeuralNetwork::backward(const std::vector<double>& input, const std::vector<double>& target, double learning_rate) {
    std::vector<double> output = forward(input);
    std::vector<double> gradient = output;

    for (size_t i = 0; i < gradient.size(); ++i) {
        gradient[i] -= target[i];
    }

    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        gradient = (*it)->backward(gradient, learning_rate);
    }
}
