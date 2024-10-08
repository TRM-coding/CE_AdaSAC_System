#include <Layer/LayerRegister.hpp>
namespace MINI_MLsys {
void LayerRegister::Register(const std::string &type, LayerCreator creator) {

  if (!creator) {
    std::cout << "LayerRegister: Layer creator is nullptr." << std::endl;
    return;
  }
  if (registry.find(type) != registry.end()) {
    std::cout << "LayerRegister: Layer type " << type << " already registered."
              << std::endl;
    return;
  }
  registry[type] = creator;
}

const std::map<std::string, LayerRegister::LayerCreator>
LayerRegister::get_registry() {
  return LayerRegister::registry;
}
} // namespace MINI_MLsys