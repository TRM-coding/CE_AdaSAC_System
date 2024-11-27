class Matrix_dp:
    def __init__(self,matrix_list):
        self.matrix_size_list=[]
        self.compute_order={}
        self.matrix_list=matrix_list
        self.dp_list={}
        for matrix_i in matrix_list:
            self.matrix_size_list.append((matrix_i.shape[0],matrix_i.shape[1]))
        print(self.matrix_size_list)
        self.cnt=0
        return
    
    def Recrusively_DP(self,left,right):
        if((left,right)in self.dp_list):
            return self.dp_list[(left,right)][0],self.dp_list[(left,right)][1]
        if(left==right):
            self.dp_list[(left,right)]=(self.matrix_size_list[left],0)
            return self.matrix_size_list[left],0
        compute_cost=0x3f3f3f3f
        split_location=0
        
        for new_right in range(left,right):
            matrix_l,compute_cost_l = self.Recrusively_DP(left,new_right)
            matrix_r,compute_cost_r = self.Recrusively_DP(new_right+1,right)
            if(matrix_l[0]*matrix_l[1]*matrix_r[1]+compute_cost_l+compute_cost_r<compute_cost):
                compute_cost   = matrix_l[0]*matrix_l[1]*matrix_r[1]+compute_cost_l+compute_cost_r
                split_location = new_right
        self.compute_order[(left,right)]=split_location
        
        self.dp_list[(left,right)]=((self.matrix_size_list[left][0],self.matrix_size_list[right][1]),compute_cost)
        return (self.matrix_size_list[left][0],self.matrix_size_list[right][1]),compute_cost

    def Recrusively_compute(self,left,right):
        if(left==right):
            return self.matrix_list[left]
        print(left,right)
        split_location=self.compute_order[(left,right)]
        matrix_l=self.Recrusively_compute(left,split_location)
        matrix_r=self.Recrusively_compute(split_location+1,right)
        return matrix_l@matrix_r

import torch

if __name__=='__main__':
    tensor_list=[]
    tensor_list.append(torch.randn(3,1))
    tensor_list.append(torch.randn(1,55))
    tensor_list.append(torch.randn(55,6))
    tensor_list.append(torch.randn(6,777))
    tensor_list.append(torch.randn(777,8))
    dp=Matrix_dp(tensor_list)
    _,cost=dp.Recrusively_DP(0,len(tensor_list)-1)
    print("minimum cost:",cost)
    print(dp.Recrusively_compute(0,len(tensor_list)-1))
    print(dp.compute_order)
    print(dp.cnt)
    # for i in range(3,4):
    #     print(i)