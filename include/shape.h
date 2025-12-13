#pragma once

#include <vector>

namespace nn 
{
    struct Shape
    {
        std::vector<int> dims;

        Shape() = default;
        Shape(const std::vector<int> &dimensions);
        Shape(std::initializer_list<int> dimensions);

        int operator[](size_t i) const;
        size_t size() const;
        int numel() const;
    };
} // namespace nn
