#pragma once

#include <vector>
#include <string>

struct MNISTData {
    std::vector<std::vector<double>> images;
    std::vector<std::vector<double>> labels;
};

MNISTData load_mnist_data(const std::string& filename, int num_samples = -1);
std::pair<MNISTData, MNISTData> split_data(const MNISTData& data, double train_ratio = 0.8);
void shuffle_data(MNISTData& data);
