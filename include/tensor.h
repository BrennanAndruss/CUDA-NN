#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "shape.h"

class Tensor
{
public:
    Tensor() = default;
    Tensor(const Shape &s);
    Tensor(std::initializer_list<int> dims);

    int numel() const;

    void allocDevice();
    void allocGrad();
    void allocHost() const;

    void generateRand();

    void toDevice();
    void toHost() const;

    const float* data() const;
    const float* grad() const;

    void printData() const;
    void printGrad() const;

    mutable thrust::host_vector<float> h_data;

private:
    Shape shape;

    thrust::device_vector<float> d_data;
    thrust::device_vector<float> d_grad;
};
