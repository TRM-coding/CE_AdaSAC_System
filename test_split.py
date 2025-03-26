from detection.splited_model import Splited_Model

import torch

model_A = Splited_Model()
model_A = torch.load("./clientA.pth")
model_B=Splited_Model()
model_B=torch.load("./clientB.pth")
model_C=Splited_Model()
model_C=torch.load("./clientC.pth")
model_A.to('cpu')
model_B.to('cpu')
model_C.to('cpu')
model_A.eval()
model_B.eval()
model_C.eval()

input=torch.randn(1,3,224,224)
output_A=model_A(input)
output_B=model_B(output_A)
output_C=model_C(output_B)
print(output_C)