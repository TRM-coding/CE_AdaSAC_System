#ifndef MINI_ML_LAYER_REGISTER_HPP
#define MINI_ML_LAYER_REGISTER_HPP
#include <map>
#include <string>
#include <Operator.hpp>
namespace MINI_MLsys
{
  class LayerRegister
  {

  public:
    typedef bool (*LayerCreator)(const std::shared_ptr<Operator> &op, std::shared_ptr<Layer> &layer);
    static std::map<std::string, LayerCreator> registry;

    static void Register(const std::string &type, LayerCreator creator);

    static const std::map<std::string, LayerCreator> get_registry();
  };

  class LayerRegisterAssistant
  {
  public:
    LayerRegisterAssistant(const std::string &type, LayerRegister::LayerCreator creator)
    {
      LayerRegister::Register(type, creator);
    }
  };
}
#endif // MINI_ML_LAYER_REGISTER_HPP