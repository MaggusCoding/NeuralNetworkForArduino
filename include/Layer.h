#pragma once

#include <vector>

class Layer {
public:
    virtual std::vector<double> forward(const std::vector<double>& input) = 0;
    virtual std::vector<double> backward(const std::vector<double>& gradient, double learning_rate) = 0;
    virtual ~Layer() = default;
};
