#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <memory>
#include <filesystem>
#include "MNISTData.h"
#include "NeuralNetwork.h"
#include "DenseLayer.h"


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
    nn.add_layer(std::make_unique<DenseLayer>(784, 30, Activation::ReLU));
    nn.add_layer(std::make_unique<DenseLayer>(30, 25, Activation::ReLU));
    nn.add_layer(std::make_unique<DenseLayer>(25, 10, Activation::Softmax));

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
