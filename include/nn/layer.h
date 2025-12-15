#pragma once

#include "tensor.h"
#include "common.h"

namespace nn 
{
    class Layer
    {
    public:
        int inSize;
        int outSize;

        Layer(int inSize, int outSize) : inSize(inSize), outSize(outSize), mode(Mode::Train) {}
        virtual ~Layer() = default;

        virtual Tensor forward(Tensor &in) = 0;
        virtual Tensor backward(Tensor &gradOut) = 0;

        virtual void setMode(Mode m) { mode = m; }

        virtual std::vector<Tensor*> getParams() = 0;

        virtual void save(std::ostream &out) const = 0;
    
    protected:
        Mode mode;
    };
} // namespace nn
