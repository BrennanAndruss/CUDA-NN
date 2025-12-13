#pragma once

#include <vector>
#include "tensor.h"
#include "layer.h"

namespace nn 
{
    class Network
    {
    public:
        Network() = default;

        void addLayer(Layer *layer);

        Tensor forward(const Tensor &in) const;
        void backward(const Tensor &gradLoss) const;

        std::vector<Tensor*> getParams() const;

        void save(const std::string &filepath) const;

    private:
        std::vector<Layer*> layers;
        std::vector<Tensor*> params;
    };
} // namespace nn
