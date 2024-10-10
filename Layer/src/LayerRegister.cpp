#include <Layer/LayerRegister.hpp>
namespace MINI_MLsys {

std::map<std::string, LayerRegister::LayerCreator> *LayerRegister::registry;

void LayerRegister::Register(const std::string &type, LayerCreator creator) {

  if (!creator) {
    std::cout << "LayerRegister: Layer creator is nullptr." << std::endl;
    return;
  }
  if (registry == nullptr) {
    registry = new std::map<std::string, LayerCreator>();
  }

  if (registry->find(type) != registry->end()) {
    std::cout << "LayerRegister: Layer type " << type << " already registered."
              << std::endl;
    return;
  }
  registry->insert(std::make_pair(type, creator));
  return;
}

const std::map<std::string, LayerRegister::LayerCreator> *
LayerRegister::get_registry() {
  return LayerRegister::registry;
}
} // namespace MINI_MLsys