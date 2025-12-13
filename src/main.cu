#include "linear.h"
#include "sigmoid.h"
#include "relu.h"
#include "mse.h"
#include "network.h"

int main(int argc, char *argv[])
{
    // Initialize network
    nn::Network net;
    
    net.addLayer(new nn::Linear(2, 2));
    net.addLayer(new nn::Sigmoid(2));
    net.addLayer(new nn::Linear(2, 2));
    net.addLayer(new nn::Sigmoid(2));
    net.addLayer(new nn::Linear(2, 1));
    net.addLayer(new nn::Sigmoid(1));

    nn::MSELoss lossFn;

    // Create dummy input and output
    nn::Tensor input({2});
    input.allocHost();
    input.h_data[0] = 1.0f;
    input.h_data[1] = 2.0f;

    nn::Tensor output({1});
    output.allocHost();
    output.h_data[0] = 1.0f;
    
    // Run network
    nn::Tensor pred = net.forward(input);
    float loss = lossFn.lossFn(pred, output);
    std::cout << "Loss: " << loss << "\n";

    // Save the network architecture and parameters
    std::cout << "Saving network to ../network_model.txt\n";
    net.save("../network_model.txt");

    return 0;
}