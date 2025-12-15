#include "nn/tensor.h"

#include <thrust/generate.h>
#include <iostream>
#include <random>

namespace nn {

Tensor::Tensor(const Shape& shape) : 
    shape(shape), h_data(), d_data(), d_grad() {}

Tensor::Tensor(std::initializer_list<int> dims) : 
    shape(dims), h_data(), d_data(), d_grad() {}

Tensor Tensor::fromVector(const std::vector<float> &data)
{
    Tensor t({static_cast<int>(data.size())});
    t.allocHost();
    t.h_data = data;

    return t;
}

int Tensor::numel() const { return shape.numel(); }

const Shape& Tensor::getShape() const { return shape; }

void Tensor::allocDevice() { d_data.resize(numel()); }
void Tensor::allocGrad() { d_grad.resize(numel()); }
void Tensor::allocHost() const { h_data.resize(numel()); }

void Tensor::generateRand(float factor)
{
    allocHost();
    thrust::generate(h_data.begin(), h_data.end(), [factor]() {
        return factor * static_cast<float>(std::rand()) / RAND_MAX;
    });
}

void Tensor::generateKaiming(int fanIn, float factor)
{
    allocHost();
    float stddev = factor * std::sqrt(2.0f / fanIn);
    thrust::generate(h_data.begin(), h_data.end(), [stddev]() {
        static std::default_random_engine generator;
        static std::normal_distribution<float> dist(0.0f, stddev);
        return dist(generator);
    });
}

void Tensor::toDevice()
{
    if (d_data.size() != h_data.size())
        allocDevice();
    d_data = h_data;
}

void Tensor::toHost() const
{
    if (h_data.size() != d_data.size())
        allocHost();
    h_data = d_data;
}

float* Tensor::data()
{
    return thrust::raw_pointer_cast(d_data.data());
}

float* Tensor::grad()
{
    return thrust::raw_pointer_cast(d_grad.data());
}

void Tensor::save(std::ostream &out) const
{
    thrust::host_vector<float> h_temp = d_data;
    for (size_t i = 0; i < shape.size(); i++)
    {
        out << shape[i] << " ";
    }
    out << "\n";

    for (size_t i = 0; i < h_temp.size(); i++)
    {
        out << h_temp[i] << " ";
    }
    out << "\n";
}

void Tensor::printData() const
{
    thrust::host_vector<float> h_temp = d_data;
    for (size_t i = 0; i < h_temp.size(); i++)
    {
        std::cout << h_temp[i] << " ";
    }
    std::cout << "\n";
}

void Tensor::printGrad() const
{
    thrust::host_vector<float> h_temp = d_grad;
    for (size_t i = 0; i < h_temp.size(); i++)
    {
        std::cout << h_temp[i] << " ";
    }
    std::cout << "\n";
}

} // namespace nn
