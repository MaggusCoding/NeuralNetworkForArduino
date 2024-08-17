#include "DenseLayer.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

DenseLayer::DenseLayer(int input_size, int output_size, Activation act) : activation(act) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, std::sqrt(2.0 / input_size));

    weights.resize(output_size, std::vector<double>(input_size));
    biases.resize(output_size);

    for (auto& neuron_weights : weights) {
        for (auto& weight : neuron_weights) {
            weight = d(gen);
        }
    }

    for (auto& bias : biases) {
        bias = 0.0;
    }
}

std::vector<double> DenseLayer::forward(const std::vector<double>& input) {
    this->input = input;
    output.resize(weights.size());

    for (size_t i = 0; i < weights.size(); ++i) {
        double sum = biases[i];
        for (size_t j = 0; j < weights[i].size(); ++j) {
            sum += weights[i][j] * input[j];
        }
        output[i] = sum;
    }

    switch (activation) {
        case Activation::Sigmoid:
            for (auto& val : output) val = sigmoid(val);
            break;
        case Activation::ReLU:
            for (auto& val : output) val = relu(val);
            break;
        case Activation::Softmax:
            output = softmax(output);
            break;
    }

    return output;
}

std::vector<double> DenseLayer::backward(const std::vector<double>& gradient, double learning_rate) {
    std::vector<double> input_gradient(input.size(), 0.0);

    for (size_t i = 0; i < weights.size(); ++i) {
        double delta;
        switch (activation) {
            case Activation::Sigmoid:
                delta = gradient[i] * sigmoid_derivative(output[i]);
                break;
            case Activation::ReLU:
                delta = gradient[i] * relu_derivative(output[i]);
                break;
            case Activation::Softmax:
                delta = gradient[i];  // For softmax, we assume the gradient is already correct
                break;
        }

        for (size_t j = 0; j < weights[i].size(); ++j) {
            input_gradient[j] += weights[i][j] * delta;
            weights[i][j] -= learning_rate * delta * input[j];
        }
        biases[i] -= learning_rate * delta;
    }

    return input_gradient;
}

double DenseLayer::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double DenseLayer::sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

double DenseLayer::relu(double x) {
    return std::max(0.0, x);
}

double DenseLayer::relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

std::vector<double> DenseLayer::softmax(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    double max = *std::max_element(x.begin(), x.end());
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::exp(x[i] - max);
        sum += result[i];
    }
    for (double& val : result) {
        val /= sum;
    }
    return result;
}
