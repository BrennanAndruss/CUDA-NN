#pragma once

#include "layer.h"

namespace nn
{
    class Softmax : public Layer
    {
    public:
        Softmax(int size);

        Tensor forward(Tensor &in) override;
        Tensor backward(Tensor &gradOut) override;

        std::vector<Tensor*> getParams() override;

        void save(std::ostream &out) const override;

    private:
        int gridSize;
    };
} // namespace nn
