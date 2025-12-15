#include "linear.h"
#include "sigmoid.h"
#include "relu.h"
#include "mse.h"
#include "sgd.h"
#include "network.h"

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

int main(int argc, char *argv[])
{
    trainSimple();

    return 0;
}