#ifndef ATTRIBUTE_HPP
#define ATTRIBUTE_HPP
#include <vector>
#include <cassert>
#include <cstring>
#include <stdint.h>

namespace MINI_MLsys
{
  class Attribute
  {
    std::vector<char> data;
    std::vector<int> shape;
    std::vector<float> get(bool clear_origin_data);
    Attribute(std::vector<float> &data, std::vector<int> &shape) : data(data.begin(), data.end()), shape(shape.begin(), shape.end()){};
  };
}
#endif