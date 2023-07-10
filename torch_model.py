import torch
import torch.nn as nn
import torch.nn.functional as F
    
class torch_organization_model(nn.Module):
    def __init__(self, input_dim=89, hidden_units=[128], out_dim = 64):
        super(torch_organization_model, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_units[0])
        hidden_layers = []
        for i in range(1,len(hidden_units)):
            hidden_layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(hidden_units[-1], out_dim)

    def forward(self, x):

        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)

        return x

class torch_top_model(nn.Module):
    def __init__(self, input_dim=89, hidden_units=[128], num_classes=2):
        super(torch_top_model, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_units[0])
        hidden_layers = []
        for i in range(1,len(hidden_units)):
            hidden_layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(hidden_units[-1], num_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        # temp_tensor = torch.zeros(len(x), 2)
        # x = torch.sigmoid(x).squeeze()
        # for i in range(len(x)):
        #     if x[i] > 0.5:
        #         temp_tensor[i][1] = 1
        #     else:
        #         temp_tensor[i][0] = 1
        # x = torch.reshape(x, (len(x), 1))
        # x = self.softmax(x)

        # return temp_tensor
        # return x
        x = torch.sigmoid(x).squeeze()
        zeros_tensor = torch.zeros_like(x)
        x = torch.cat((zeros_tensor.view(-1,1), x.view(-1,1)), dim=1)
        # x = torch.reshape(x, (len(x), 1))
        # x = self.softmax(x)

        return x

class MlpModel(nn.Module):
    def __init__(self, input_dim=89, hidden_units=[128], num_classes=2):
        super(MlpModel, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_units[0])
        hidden_layers = []
        for i in range(1,len(hidden_units)):
            hidden_layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(hidden_units[-1], num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        # x = self.softmax(x)
        # x = torch.sigmoid(x)

        return x
