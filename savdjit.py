import torch
import torch.jit.mobile
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
from splited_model import Splited_Model

# edge_A=Splited_Model()
edge_A=torch.load('./p_model/edge_A.pth').to('cpu')
# edge_B=Splited_Model()
edge_B=torch.load('./p_model/edge_B.pth').to('cpu')
# cloud=Splited_Model()
cloud=torch.load('./p_model/cloud.pth').to('cpu')

edge_A.eval()
cloud.eval()
edge_B.eval()

example=torch.rand(1,3,32,32).to('cpu')


traced_script_module = torch.jit.trace(edge_A, example)
#traced_script_module_optimized = optimize_for_mobile(traced_script_module)
traced_script_module._save_for_lite_interpreter("./p_model/edge_A.ptl")

example=cloud(edge_A(example))

traced_script_module = torch.jit.trace(edge_B, example)
#traced_script_module_optimized = optimize_for_mobile(traced_script_module)
traced_script_module._save_for_lite_interpreter("./p_model/edge_B.ptl")
