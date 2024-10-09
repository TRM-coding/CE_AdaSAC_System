#include <Attribute.hpp>

namespace MINI_MLsys {
std::vector<float> Attribute::get(bool clear_origin_data) {
  assert(!data.empty());
  std::vector<float> res;
  const uint32_t float_size = sizeof(float);
  assert(data.size() % float_size == 0);
  for (uint32_t i = 0; i < data.size(); i += float_size) {
    float value;
    memcpy(&value, &data[i], float_size);
    res.push_back(value);
  }
  if (clear_origin_data) {
    data.clear();
  }
  return res;
}
} // namespace MINI_MLsys