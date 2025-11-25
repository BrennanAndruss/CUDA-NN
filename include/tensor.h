#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "shape.h"

struct Tensor
{
    Shape shape;

    mutable thrust::host_vector<float> h_data;
    thrust::device_vector<float> d_data;
    thrust::device_vector<float> d_grad;

    Tensor() = default;
    Tensor(const Shape &s);
    Tensor(std::initializer_list<int> dims);

    int numel() const;

    void allocDevice();
    void allocGrad();
    void allocHost() const;

    void generateRand();

    void copyToDevice();
    void copyToHost() const;

    void printData() const;
    void printGrad() const;
};
