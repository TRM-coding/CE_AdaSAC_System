from torch import nn
class Splited_Model(nn.Module):
    def __init__(self):
        super(Splited_Model,self).__init__()
        self.model_list=nn.ModuleList()

    def forward(self,x):
        for model in self.model_list:
            if(len(x.shape)==5):
                x=x.view(-1,x.shape[2],x.shape[3],x.shape[4])
            if(len(x.shape)==3):
                x=x.view(1,x.shape[0],x.shape[1],x.shape[2])
            x=model(x)
        return x
    