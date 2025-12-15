#pragma once

#include "tensor.h"
#include "layer.h"
#include "common.h"

namespace nn
{
    class Network
    {
    public:
        Network() = default;

        void train();
        void eval();

        void addLayer(Layer *layer);

        Tensor forward(const Tensor &in) const;
        void backward(const Tensor &gradLoss) const;

        std::vector<Tensor*> getParams() const;

        void save(const std::string &filepath) const;

    private:
        Mode mode;
        std::vector<Layer*> layers;
        std::vector<Tensor*> params;
    };
} // namespace nn
