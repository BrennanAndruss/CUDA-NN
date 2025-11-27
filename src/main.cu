#include "linear.h"
#include "sigmoid.h"
#include "relu.h"
#include "network.h"

int main(int argc, char *argv[])
{
    // Initialize network
    Network net;
    
    net.addLayer(new Linear(2, 2));
    net.addLayer(new Sigmoid(2));
    net.addLayer(new Linear(2, 2));
    net.addLayer(new Sigmoid(2));
    net.addLayer(new Linear(2, 1));
    net.addLayer(new Sigmoid(1));

    // Save the network architecture and parameters
    std::cout << "Saving network to ../network_model.txt\n";
    net.save("../network_model.txt");

    return 0;
}