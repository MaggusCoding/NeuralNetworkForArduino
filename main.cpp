#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <chrono>
#include <memory>
#include <filesystem>
#include <cstring>

// MNIST Data Structure
struct MNISTData {
    std::vector<std::vector<double>> images;
    std::vector<std::vector<double>> labels;
};

// Data Preparation Functions
MNISTData load_mnist_data(const std::string& filename, int num_samples = -1) {
    std::ifstream file(filename);
    std::string line;
    MNISTData data;

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        std::cerr << "Current working directory: " << std::filesystem::current_path() << std::endl;
        std::cerr << "Error details: " << std::strerror(errno) << std::endl;
        return data;
    }

    int line_count = 0;
    while (std::getline(file, line) && (num_samples == -1 || data.images.size() < num_samples)) {

        std::vector<double> image(784);
        std::vector<double> label(10, 0.0);

        std::istringstream iss(line);
        std::string token;
        int idx = 0;

        while (std::getline(iss, token, ',')) {
            if (idx == 0) {
                int digit = std::stoi(token);
                label[digit] = 1.0;
                if (line_count < 5) {
                    std::cout << "Label: " << digit << std::endl;
                }
            } else if (idx <= 784) {
                double pixel_value = std::stod(token);
                image[idx - 1] = pixel_value / 255.0;  // Normalize to [0, 1]
                if (pixel_value != 0 && line_count < 5) {
                    std::cout << "Non-zero pixel at index " << idx-1 << ": " << pixel_value << std::endl;
                }
            }
            idx++;
        }

        if (line_count < 5) {
            std::cout << "Number of values read: " << idx << std::endl;
            std::cout << std::endl;
        }

        data.images.push_back(image);
        data.labels.push_back(label);
        line_count++;
    }

    std::cout << "Total lines read: " << line_count << std::endl;

    return data;
}

std::pair<MNISTData, MNISTData> split_data(const MNISTData& data, double train_ratio = 0.8) {
    MNISTData train_data, test_data;
    size_t train_size = static_cast<size_t>(data.images.size() * train_ratio);

    train_data.images = std::vector<std::vector<double>>(data.images.begin(), data.images.begin() + train_size);
    train_data.labels = std::vector<std::vector<double>>(data.labels.begin(), data.labels.begin() + train_size);

    test_data.images = std::vector<std::vector<double>>(data.images.begin() + train_size, data.images.end());
    test_data.labels = std::vector<std::vector<double>>(data.labels.begin() + train_size, data.labels.end());

    return {train_data, test_data};
}

void shuffle_data(MNISTData& data) {
    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<size_t> indices(data.images.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), g);

    std::vector<std::vector<double>> shuffled_images(data.images.size());
    std::vector<std::vector<double>> shuffled_labels(data.labels.size());

    for (size_t i = 0; i < indices.size(); ++i) {
        shuffled_images[i] = data.images[indices[i]];
        shuffled_labels[i] = data.labels[indices[i]];
    }

    data.images = shuffled_images;
    data.labels = shuffled_labels;
}

// Activation functions
enum class Activation {
    Sigmoid,
    ReLU,
    Softmax
};

class Layer {
public:
    virtual std::vector<double> forward(const std::vector<double>& input) = 0;
    virtual std::vector<double> backward(const std::vector<double>& gradient, double learning_rate) = 0;
    virtual ~Layer() = default;
};

class DenseLayer : public Layer {
private:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> input;
    std::vector<double> output;
    Activation activation;

    static double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    static double sigmoid_derivative(double x) {
        double s = sigmoid(x);
        return s * (1 - s);
    }

    static double relu(double x) {
        return std::max(0.0, x);
    }

    static double relu_derivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }

    std::vector<double> softmax(const std::vector<double>& x) {
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

public:
    DenseLayer(int input_size, int output_size, Activation act) : activation(act) {
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

    std::vector<double> forward(const std::vector<double>& input) override {
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

    std::vector<double> backward(const std::vector<double>& gradient, double learning_rate) override {
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
};

class NeuralNetwork {
private:
    std::vector<std::unique_ptr<Layer>> layers;

public:
    void add_layer(std::unique_ptr<Layer> layer) {
        layers.push_back(std::move(layer));
    }

    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> current = input;
        for (auto& layer : layers) {
            current = layer->forward(current);
        }
        return current;
    }

    void backward(const std::vector<double>& input, const std::vector<double>& target, double learning_rate) {
        std::vector<double> output = forward(input);
        std::vector<double> gradient = output;

        for (size_t i = 0; i < gradient.size(); ++i) {
            gradient[i] -= target[i];
        }

        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            gradient = (*it)->backward(gradient, learning_rate);
        }
    }
};

int main() {
    std::string filename = "example.csv";

    MNISTData all_data = load_mnist_data(filename);
    if (all_data.images.empty()) {
        std::cerr << "Failed to load data. Exiting." << std::endl;
        return 1;
    }
    std::cout << "Loaded " << all_data.images.size() << " samples." << std::endl;

    auto [train_data, test_data] = split_data(all_data);
    std::cout << "Training samples: " << train_data.images.size() << std::endl;
    std::cout << "Testing samples: " << test_data.images.size() << std::endl;

    // Check for empty images
    int empty_images = 0;
    for (const auto& image : all_data.images) {
        if (std::all_of(image.begin(), image.end(), [](double pixel) { return pixel == 0; })) {
            empty_images++;
        }
    }
    std::cout << "Number of empty images: " << empty_images << std::endl;

    // Check label distribution
    std::vector<int> label_counts(10, 0);
    for (const auto& label : all_data.labels) {
        int digit = std::distance(label.begin(), std::max_element(label.begin(), label.end()));
        label_counts[digit]++;
    }
    std::cout << "Label distribution:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "Digit " << i << ": " << label_counts[i] << std::endl;
    }

    shuffle_data(train_data);

    NeuralNetwork nn;
    nn.add_layer(std::make_unique<DenseLayer>(784, 128, Activation::ReLU));
    nn.add_layer(std::make_unique<DenseLayer>(128, 64, Activation::ReLU));
    nn.add_layer(std::make_unique<DenseLayer>(64, 10, Activation::Softmax));

    int epochs = 10;
    double learning_rate = 0.001;  // Reduced learning rate
    int batch_size = 32;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto start = std::chrono::high_resolution_clock::now();

        double total_loss = 0.0;
        for (int i = 0; i < train_data.images.size(); i += batch_size) {
            for (int j = i; j < std::min(i + batch_size, static_cast<int>(train_data.images.size())); ++j) {
                nn.backward(train_data.images[j], train_data.labels[j], learning_rate);

                // Calculate loss
                auto output = nn.forward(train_data.images[j]);
                double sample_loss = 0.0;
                for (size_t k = 0; k < output.size(); ++k) {
                    sample_loss -= train_data.labels[j][k] * std::log(output[k] + 1e-10);
                }
                total_loss += sample_loss;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        int correct = 0;
        for (size_t i = 0; i < test_data.images.size(); ++i) {
            auto output = nn.forward(test_data.images[i]);
            if (std::distance(output.begin(), std::max_element(output.begin(), output.end())) ==
                std::distance(test_data.labels[i].begin(), std::max_element(test_data.labels[i].begin(), test_data.labels[i].end()))) {
                ++correct;
            }
        }

        double avg_loss = total_loss / train_data.images.size();
        double accuracy = static_cast<double>(correct) / test_data.images.size() * 100.0;

        std::cout << "Epoch " << epoch + 1 << "/" << epochs
                  << ", Time: " << diff.count() << "s"
                  << ", Avg Loss: " << avg_loss
                  << ", Test accuracy: " << accuracy << "%"
                  << std::endl;
    }

    return 0;
}
