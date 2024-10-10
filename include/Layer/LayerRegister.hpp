#ifndef MINI_ML_LAYER_REGISTER_HPP
#define MINI_ML_LAYER_REGISTER_HPP
#include <Operator.hpp>
#include <map>
#include <string>
namespace MINI_MLsys {
class LayerRegister {

public:
  typedef bool (*LayerCreator)(const std::shared_ptr<Operator> &op);
  

  static void Register(const std::string &type, LayerCreator creator);

  static const std::map<std::string, LayerCreator>* get_registry();
  // private:
  static std::map<std::string, LayerCreator>* registry;
};

class LayerRegisterAssistant {
public:
  LayerRegisterAssistant(const std::string &type,
                         LayerRegister::LayerCreator creator) {
    LayerRegister::Register(type, creator);
  }
};
} // namespace MINI_MLsys
#endif // MINI_ML_LAYER_REGISTER_HPP