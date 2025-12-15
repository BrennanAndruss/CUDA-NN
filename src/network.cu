#include "network.h"

namespace nn {

void Network::addLayer(Layer *layer)
{
    layers.push_back(layer);
    std::vector<Tensor*> layerParams = layer->getParams();
    params.insert(params.end(), layerParams.begin(), layerParams.end());
}

Tensor Network::forward(const Tensor &in) const
{
    Tensor curIn = in;
    curIn.toDevice();
    Tensor curOut;

    for (Layer *layer : layers)
    {
        curOut = layer->forward(curIn);
        curIn = curOut;
    }

    return curOut;
}

void Network::backward(const Tensor &gradLoss) const
{
    Tensor gradOut = gradLoss;
    Tensor gradIn;

    for (auto it = layers.rbegin(); it != layers.rend(); it++)
    {
        Layer *layer = *it;
        gradIn = layer->backward(gradOut);
        gradOut = gradIn;
    }
}

std::vector<Tensor*> Network::getParams() const { return params; }

void Network::save(const std::string &filepath) const
{
    std::ofstream out(filepath);
    if (!out.is_open())
    {
        std::cerr << "Error: Could not open file " << filepath << ".\n";
        return;
    }

    out << layers.size() << "\n";
    for (Layer *layer : layers)
    {
        layer->save(out);
    }
    
    out.close();
}

} // namespace nn
