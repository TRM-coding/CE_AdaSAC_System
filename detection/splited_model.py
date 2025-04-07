from torch import nn
class Splited_Model(nn.Module):
    def __init__(self):
        super(Splited_Model,self).__init__()
        self.model_list=nn.ModuleList()

    def forward(self,x):
        time=0
        for model in self.model_list:
            if(len(x.shape)==3):
                x=x.unsqueeze(0)
            x=model(x)
        return x
    