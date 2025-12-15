#include "nn/linear.h"
#include "nn/sigmoid.h"
#include "nn/relu.h"
#include "nn/softmax.h"
#include "nn/mse.h"
#include "nn/crossentropy.h"
#include "nn/sgd.h"
#include "nn/network.h"

#include <cstdlib>
#include <algorithm>
#include <iterator>
#include <stdexcept>

uint32_t readBigEndian32(std::ifstream &in)
{
    uint32_t result = 0;
    unsigned char bytes[4];
    in.read(reinterpret_cast<char*>(bytes), 4);
    result = (bytes[0] << 24) |
             (bytes[1] << 16) |
             (bytes[2] << 8)  |
             (bytes[3]);
    return result;
}

std::vector<std::vector<float>> loadMNISTImages(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
    {
        throw std::runtime_error("Failed to open MNIST image file: " + path);
    }

    uint32_t magicNum = readBigEndian32(in);
    if (magicNum != 2051)
    {
        throw std::runtime_error("Invalid MNIST image file magic number");
    }

    uint32_t numImages = readBigEndian32(in);
    uint32_t numRows = readBigEndian32(in);
    uint32_t numCols = readBigEndian32(in);

    size_t imageSize = numRows * numCols;
    std::vector<std::vector<float>> images(numImages, std::vector<float>(imageSize));

    for (uint32_t i = 0; i < numImages; i++)
    {
        std::vector<unsigned char> buffer(imageSize);
        in.read(reinterpret_cast<char*>(buffer.data()), imageSize);

        for (size_t j = 0; j < imageSize; j++)
        {
            images[i][j] = buffer[j] / 255.0f;
        }
    }

    return images;
}

std::vector<std::vector<float>> loadMNISTLabels(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
    {
        throw std::runtime_error("Failed to open MNIST label file: " + path);
    }

    uint32_t magicNum = readBigEndian32(in);
    if (magicNum != 2049)
    {
        throw std::runtime_error("Invalid MNIST label file magic number");
    }

    uint32_t numLabels = readBigEndian32(in);

    std::vector<std::vector<float>> labels(numLabels, std::vector<float>(10, 0.0f));

    for (uint32_t i = 0; i < numLabels; i++)
    {
        unsigned char label;
        in.read(reinterpret_cast<char*>(&label), 1);
        labels[i][static_cast<size_t>(label)] = 1.0f;
    }

    return labels;
}

void trainSimple()
{
    nn::Network net;

    std::cout << "Creating network...\n";
    net.addLayer(new nn::Linear(2, 2));
    net.addLayer(new nn::Sigmoid(2));
    net.addLayer(new nn::Linear(2, 2));
    net.addLayer(new nn::Sigmoid(2));
    net.addLayer(new nn::Linear(2, 1));
    net.addLayer(new nn::Sigmoid(1));

    nn::MSELoss lossFn;
    nn::SGDOptimizer optimizer(net.getParams(), 0.1f);

    // Create and initialize input and output
    nn::Tensor input({2});
    input.allocHost();
    input.h_data[0] = 1.0f;
    input.h_data[1] = 2.0f;

    nn::Tensor output({1});
    output.allocHost();
    output.h_data[0] = 1.0f;

    // Run training loop
    std::cout << "Training simple model\n";
    for (int i = 0; i < 50; i++)
    {
        std::cout << "Iteration " << (i + 1) << "/50\n";

        nn::Tensor pred = net.forward(input);
        float loss = lossFn.lossFn(pred, output);
        std::cout << " Loss: " << loss << "\n";

        nn::Tensor gradLoss = lossFn.backward();
        net.backward(gradLoss);
        optimizer.step();
    }

    std::cout << "Saving network model...\n";
    net.save("simple_model.txt");
}

void trainMNIST()
{
    // Load MNIST data
    std::cout << "Loading MNIST data...\n";

    auto trainImages = loadMNISTImages("../data/train-images.idx3-ubyte");
    auto trainLabels = loadMNISTLabels("../data/train-labels.idx1-ubyte");
    int trainSubset = 10000;

    auto testImages = loadMNISTImages("../data/t10k-images.idx3-ubyte");
    auto testLabels = loadMNISTLabels("../data/t10k-labels.idx1-ubyte");
    int testSubset = 5000;

    // Initialize network
    nn::Network net;

    std::cout << "Creating network...\n";
    net.addLayer(new nn::Linear(784, 128));
    net.addLayer(new nn::ReLU(128));
    net.addLayer(new nn::Linear(128, 64));
    net.addLayer(new nn::ReLU(64));
    net.addLayer(new nn::Linear(64, 10));
    net.addLayer(new nn::Softmax(10));

    nn::CrossEntropyLoss lossFn;
    nn::SGDOptimizer optimizer(net.getParams(), 0.001f);

    // Run training loop
    int epochs = 8;
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs << "\n";
        float epochLoss = 0.0f;

        for (int i = 0; i < trainSubset; i++)
        {
            std::clog << " Training sample " << (i + 1) << "/" << trainSubset << "\r";

            nn::Tensor in = nn::Tensor::fromVector(trainImages[i]);
            nn::Tensor target = nn::Tensor::fromVector(trainLabels[i]);

            nn::Tensor pred = net.forward(in);
            float loss = lossFn.lossFn(pred, target);
            epochLoss += loss;

            nn::Tensor gradLoss = lossFn.backward();
            net.backward(gradLoss);
            optimizer.step();
        }

        std::cout << "\nAverage Loss: " << (epochLoss / trainSubset) << "\n";
    }

    // Evaluate on test data
    float testLoss = 0.0f;
    int correct = 0;

    net.eval();
    for (int i = 0; i < testSubset; i++)
    {
        std::clog << " Testing sample " << (i + 1) << "/" << testSubset << "\r";

        nn::Tensor in = nn::Tensor::fromVector(testImages[i]);
        nn::Tensor target = nn::Tensor::fromVector(testLabels[i]);

        nn::Tensor pred = net.forward(in);
        float loss = lossFn.lossFn(pred, target);
        testLoss += loss;

        // Determine predicted class
        pred.toHost();

        auto maxIt = std::max_element(pred.h_data.begin(), pred.h_data.end());
        int predClass = static_cast<int>(std::distance(pred.h_data.begin(), maxIt));
        auto targetIt = std::max_element(target.h_data.begin(), target.h_data.end());
        int targetClass = static_cast<int>(std::distance(target.h_data.begin(), targetIt));

        if (predClass == targetClass)
        {
            correct++;
        }
    }

    std::cout << "\nTest Average Loss: " << (testLoss / testSubset) << "\n";
    std::cout << "Test Accuracy: " << (static_cast<float>(correct) / testSubset) * 100.0f << "%\n";

    std::cout << "Saving network model...\n";
    net.save("mnist_model.txt");
}

int main(int argc, char *argv[])
{
    // trainSimple();
    trainMNIST();

    return 0;
}
