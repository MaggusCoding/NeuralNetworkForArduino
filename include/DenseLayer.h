#pragma once

#include "Layer.h"
#include <vector>

enum class Activation {
    Sigmoid,
    ReLU,
    Softmax
};

class DenseLayer : public Layer {
private:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> input;
    std::vector<double> output;
    Activation activation;

    static double sigmoid(double x);
    static double sigmoid_derivative(double x);
    static double relu(double x);
    static double relu_derivative(double x);
    std::vector<double> softmax(const std::vector<double>& x);

public:
    DenseLayer(int input_size, int output_size, Activation act);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& gradient, double learning_rate) override;
};