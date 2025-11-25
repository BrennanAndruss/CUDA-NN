#include "tensor.h"

int main(int argc, char *argv[])
{
    Tensor input({4, 4});
    input.generateRand();
    input.copyToDevice();
    input.printData();

    return 0;
}