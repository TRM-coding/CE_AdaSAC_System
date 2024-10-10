#include <Attribute.hpp>

namespace MINI_MLsys {
// std::vector<float> Attribute::get(bool clear_origin_data) {
//   assert(!data.empty());
//   std::vector<float> res;
//   const uint32_t float_size = sizeof(float);
//   assert(data.size() % float_size == 0);
//   for (uint32_t i = 0; i < data.size(); i += float_size) {
//     float value;
//     memcpy(&value, &data[i], float_size);
//     res.push_back(value);
//   }
//   if (clear_origin_data) {
//     data.clear();
//   }
//   this->data_=res;
//   return res;
// }

Attribute::Attribute(std::vector<char> &data, std::vector<int> &shape)
    : data(data), shape(shape) {
      const uint32_t float_size = sizeof(float);
      assert(data.size() % float_size == 0);
      for (uint32_t i = 0; i < data.size(); i += float_size) {
        float value;
        memcpy(&value, &data[i], float_size);
        data_.push_back(value);
      }
    }

std::ostream& operator<<(std::ostream& os,const std::shared_ptr<Attribute> attr)
{
  std::cout<<"<<RUNED"<<std::endl;
  int idx=0;
  for(auto x:attr->data_)
  {
    os<<idx<<":"<<(float)x<<std::endl;
    idx++;
  }
  os<<std::endl;
  return os;
}

} // namespace MINI_MLsys
