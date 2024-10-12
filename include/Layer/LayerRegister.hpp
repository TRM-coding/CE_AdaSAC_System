#pragma once
// #include <Layer/Linear.hpp>
// #include <Layer/Relu.hpp>
// #include <Layer/sigmoid.hpp>
#include <Operator.hpp>
#include <map>
#include <string>
namespace MINI_MLsys {
class Operator;
class LayerRegister {

public:
  typedef bool (*LayerCreator)(std::shared_ptr<Operator> &op);

  static void Register(const std::string &type, LayerCreator creator);

  static const std::map<std::string, LayerCreator> *get_registry();
  // private:
  static std::map<std::string, LayerCreator> *registry;
};

class LayerRegisterAssistant {
public:
std::string type;
  LayerRegisterAssistant(const std::string &type,
                         LayerRegister::LayerCreator creator) {
    LayerRegister::Register(type, creator);
    
      this->type=type;
    
  }
};
} // namespace MINI_MLsys
 // MINI_ML_LAYER_REGISTER_HPP