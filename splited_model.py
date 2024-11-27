from torch import nn
class Splited_Model(nn.Module):
    def __init__(self):
        super(Splited_Model,self).__init__()
        self.model_list=nn.ModuleList()

    def forward(self,x):
        for model in self.model_list:
            x=model(x)
        return x
    