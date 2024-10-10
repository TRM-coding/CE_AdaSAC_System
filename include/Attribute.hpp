#ifndef ATTRIBUTE_HPP
#define ATTRIBUTE_HPP
#include <cassert>
#include <cstring>
#include <iostream>
#include <stdint.h>
#include <vector>
#include <memory>

namespace MINI_MLsys {
class Attribute {
  public:
  std::vector<int> shape;
  std::vector<float> data_;
  // std::vector<float> get(bool clear_origin_data);
  Attribute(std::vector<char> &data, std::vector<int> &shape);
  friend std::ostream &operator<<(std::ostream &os, const std::shared_ptr<Attribute> attr);

private:
  std::vector<char> data;
  
};
} // namespace MINI_MLsys
#endif