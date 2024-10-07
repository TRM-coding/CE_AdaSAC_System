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
    std::vector<float> get(bool clear_origin_data)
    {
      assert(!data.empty());
      std::vector<float> res;
      const uint32_t float_size = sizeof(float);
      assert(data.size() % float_size == 0);
      for (uint32_t i = 0; i < data.size(); i += float_size)
      {
        float value;
        memcpy(&value, &data[i], float_size);
        res.push_back(value);
      }
      if (clear_origin_data)
      {
        data.clear();
      }
      return res;
    }
  };
}
#endif