#include "shape.h"

Shape::Shape(const std::vector<int> &dimensions) : dims(dimensions) {}

Shape::Shape(std::initializer_list<int> dimensions) : dims(dimensions) {}

int Shape::operator[](size_t i) const { return dims[i]; }

size_t Shape::size() const { return dims.size(); }

int Shape::numel() const
{
    int n = 1;
    for (int d : dims)
    {
        n *= d;
    }
    return n;
}