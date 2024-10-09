#include <Layer/LayerRegister.hpp>
namespace MINI_MLsys {

std::map<std::string, LayerRegister::LayerCreator>*LayerRegister::registry;

void LayerRegister::Register(const std::string &type, LayerCreator creator) {

  if (!creator) {
    std::cout << "LayerRegister: Layer creator is nullptr." << std::endl;
    return;
  }
  if(registry==nullptr)
  {
    registry=new std::map<std::string, LayerCreator>();
  }
  std::cout<<(registry->begin()==registry->end())<<std::endl;
  int cnt=0;
  for(auto x=registry->begin();x!=registry->end();x++)
  {
    cnt++;
  }
  std::cout<<cnt<<std::endl;

  // if(registry==nullptr)
  // {
  //   registry=new std::map<std::string, LayerCreator>();
  // }
  if (registry->find(type) != registry->end()) {
    std::cout << "LayerRegister: Layer type " << type << " already registered."
              << std::endl;
    return;
  }
  std::cout<<type<<" "<<(creator==nullptr)<<std::endl;
  registry->insert(std::make_pair(type, creator));
  return ;
}

const std::map<std::string, LayerRegister::LayerCreator>*
LayerRegister::get_registry() {
  return LayerRegister::registry;
}
} // namespace MINI_MLsys