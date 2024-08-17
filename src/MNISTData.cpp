#include "MNISTData.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <numeric>

MNISTData load_mnist_data(const std::string& filename, int num_samples) {
    std::ifstream file(filename);
    std::string line;
    MNISTData data;


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
            } else if (idx <= 784) {
                double pixel_value = std::stod(token);
                image[idx - 1] = pixel_value / 255.0;  // Normalize to [0, 1]
            }
            idx++;
        }

        data.images.push_back(image);
        data.labels.push_back(label);
        line_count++;
    }

    std::cout << "Total lines read: " << line_count << std::endl;

    return data;
}

std::pair<MNISTData, MNISTData> split_data(const MNISTData& data, double train_ratio) {
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