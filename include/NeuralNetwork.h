#pragma once

#include "Layer.h"
#include <vector>
#include <memory>

class NeuralNetwork {
private:
    std::vector<std::unique_ptr<Layer>> layers;

public:
    void add_layer(std::unique_ptr<Layer> layer);
    std::vector<double> forward(const std::vector<double>& input);
    void backward(const std::vector<double>& input, const std::vector<double>& target, double learning_rate);
};
