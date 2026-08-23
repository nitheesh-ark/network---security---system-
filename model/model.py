'''import torch 
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_dim, inter_dim, inter_dim2, inter_dim3, out_dim):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, inter_dim)
        self.fc2 = nn.Linear(inter_dim, inter_dim2)
        self.fc3 = nn.Linear(inter_dim2, inter_dim3)
        self.fc4 = nn.Linear(inter_dim3, out_dim)

        self.relu = nn.ReLU()
    
    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return x


def model_pipe():

    branch_1 = Model(10,64,32,16,16)
    b1_x = torch.tensor([1,2,3,4,5,6,7,8,9,1], dtype = torch.float)
    b1_out = branch_1.forward(b1_x)

    branch_2 = Model(10,64,32,16,16)
    b2_x = torch.tensor([1,2,3,4,5,6,7,8,,9,1], dtype = torch.float)
    b2_out = branch_1.forward(b2_x)

    branch_3 = Model(10,64,32,16,16)
    b3_x = torch.tensor([1,2,3,4,5,6,7,8,,9,1], dtype = torch.float)
    b3_out = branch_1.forward(b3_x)

    resuduial = torch.cat((b1_out, b2_out, b3_out), dim = 0)

    final_model = Model(48, 64, 32, 16, 2)
    final_out = final_model.forward(resuduial)
'''



    


