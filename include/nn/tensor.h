#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include <fstream>
#include "shape.h"

namespace nn 
{
    class Tensor
    {
    public:
        Tensor() = default;
        Tensor(const Shape &s);
        Tensor(std::initializer_list<int> dims);

        static Tensor fromVector(const std::vector<float> &data);

        int numel() const;
        const Shape& getShape() const;

        void allocDevice();
        void allocGrad();
        void allocHost() const;

        void generateRand(float factor = 1.0f);
        void generateKaiming(int fanIn, float factor = 1.0f);

        void toDevice();
        void toHost() const;

        float* data();
        float* grad();

        void save(std::ostream &out) const;

        void printData() const;
        void printGrad() const;

        mutable thrust::host_vector<float> h_data;

    private:
        Shape shape;

        thrust::device_vector<float> d_data;
        thrust::device_vector<float> d_grad;
    };
} // namespace nn
