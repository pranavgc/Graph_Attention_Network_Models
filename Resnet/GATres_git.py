import torch
import numpy as np
import pandas as pd
from torch import nn
#from torch.utils.data import DataLoader
#from torchvision import datasets
#from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Lambda
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv,BatchNorm
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import add_self_loops as sloops


columns=["1"."2","3","4","5"]
s=pd.read_csv('data..txt',sep=" ",usecols=columns)
cell= np.array([s['1'],s['2'],s['3'],s['4']])
cell=np.transpose(cell)


class CustomDataset(Dataset):
    def __init__(self, labels, nodes,mark, edges, edge_a, transform=None, target_transform=None,add_self_loops: bool = True,prominent:bool=True,add_xy:bool=False):
    #def __init__(self, data_file, transform=None, target_transform=None):
        #self.labels = pd.read_csv(labels_file)
        #self.labels=labels.astype(int)
        self.labels=labels
        self.nodes=nodes
        self.mark=mark
        self.ei=edges
        self.ea=edge_a
        self.transform = transform
        self.ud=T.ToUndirected(reduce='max')
        self.target_transform = target_transform
        self.add_self_loops=add_self_loops
        self.prominent=prominent
        self.add_xy=add_xy
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data_n = self.nodes[idx][:,1:3]
        
        
        
        
        data_ei=self.ei[idx]
        data_ea=self.ea[idx]
        label=self.labels[idx]
        
        
        x1=torch.tensor(data_n, dtype=torch.float32)
        
        if self.add_xy:
            d_x,d_y=[],[]
            for i in self.nodes[idx][:,0]:
                d_x.append(cell[int(i),0])
                d_y.append(cell[int(i),1])
            d_x=torch.transpose(torch.tensor([d_x], dtype=torch.float32),0,1)
            d_y=torch.transpose(torch.tensor([d_y], dtype=torch.float32),0,1)
            x1=torch.cat((x1,d_x,d_y),dim=1)
        #print("x1 shape after xy",x1.shape)
        
        "Maximal or not"
        if self.prominent:
            data_mark=list(self.mark[idx].values())
            d_mark=torch.reshape(torch.tensor(data_mark, dtype=torch.float32),(len(data_mark),1))
        #print(d_mark.shape)
        #print(torch.tensor(data_n, dtype=torch.float32))
        #x1=torch.tensor(data_n, dtype=torch.float32)
            x1= torch.cat((x1,d_mark),dim=1)
            
                             
                             
        #x1=F.normalize(x1,p=2.0,dim=0)
        edge_ind1=torch.tensor(data_ei,dtype=torch.long)
        ei=edge_ind1.t().contiguous()
        #print("ei shape before self looping",ei.shape)
        ea=torch.tensor(data_ea,dtype=torch.float32)
        #ed1=F.normalize(ed1,p=2.0,dim=0)
        
        '''adding self loops'''
        if self.add_self_loops:
            a2,b2=sloops(ei,edge_attr=ea,num_nodes=len(x1[:,0]))
            b2[len(ei[0,:]):]=torch.reshape(x1[:,0],shape=(len(x1[:,0]),1))
        
            adj=torch.cat((a2,torch.transpose(b2,0,1)),dim=0)
            #print("Adj shape",adj.shape)
            #rint("ei/ea shape after self looping",a2.shape,b2.shape)
            #print(adj[2,:])
            
            adj=adj[:,adj[0, :].sort()[1]]
            #print(adj[2,:])
            #data = Data(x=x1, edge_index=adj[:2,:].type(torch.long), edge_attr=adj[2,:])
            ei=adj[:2,:].type(torch.long)
            ea=adj[2,:]
        
        label=torch.tensor(label,dtype=torch.long)
        #data = Data(x=x1, edge_index=edge_ind1.t().contiguous(), edge_attr=ed1)
        data = Data(x=x1, edge_index=ei, edge_attr=ea)
        
        #data=self.ud(data)
        if label== 3 or label==4 or label==5:
            #print("in2")
            data.y=torch.tensor(2,dtype=torch.long)
        elif label==6 or label==7:
            #print("in3")
            data.y=torch.tensor(3,dtype=torch.long)
        elif label==1 or label==2:
            #print("in1")
            data.y=torch.tensor(1,dtype=torch.long)
        elif label==0:
            data.y=torch.tensor(0,dtype=torch.long)
        else:
            raise Exception("Label incorrect")
            #data.y= label
            
        

        if self.transform is not None:
                data = self.pre_transform(data)
        
        return data
    
    

class GNN(torch.nn.Module):
    def __init__(self,input_channels, hidden_channels,num_classes,num_conv_layers,dropout, pool):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        
        self.num_conv_layers = num_conv_layers
        #self.bns0=nn.BatchNorm1d(input_channels)
    
        self.convs = nn.ModuleList([GATConv(input_channels, hidden_channels,add_self_loops=False,edge_dim=1,heads=4,concat=False)])
        self.convs.extend([GATConv(hidden_channels+input_channels,
                                                          hidden_channels,add_self_loops=False,edge_dim=1,heads=4,concat=False)])
        
        self.convs.extend([GATConv(hidden_channels*(i+1)+input_channels, hidden_channels,add_self_loops=False,edge_dim=1,heads=4,concat=False) for i in range(1,num_conv_layers)])
            
        #self.convs.extend([GCNConv(hidden_channels, hidden_channels)])
        
        #self.conv3 = GCNConv(hidden_channels, 32)
        self.linear_stack = nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            
            torch.nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            
            torch.nn.Linear(hidden_channels, num_classes),
            nn.LogSoftmax(dim=1),)
        self.pool = pool
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for i in range(num_conv_layers+1)])
        
        self.dropout = dropout

    def forward(self, a):
        # 1. Obtain node embeddings 
        #x=torch.reshape(x,(len(x),1))
        
        #x=self.bns0(x)
        #edge_attr=nn.BatchNorm1d(len(edge_attr))
        x_in, edge_index, edge_attr, batch = a.x,a.edge_index,a.edge_attr,a.batch
        x = self.convs[0](x_in, edge_index,edge_attr)
        x = self.bns[0](x)
                #edge_attr=nn.BatchNorm1d(len(edge_attr))
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(1,self.num_conv_layers):
            if i!=self.num_conv_layers-1:
                x=torch.cat((x,x_in),dim=1)
                x_in=x
                x = self.convs[i](x, edge_index,edge_attr)
                x = self.bns[i](x)
                #edge_attr=nn.BatchNorm1d(len(edge_attr))
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x=torch.cat((x,x_in),dim=1)
                x = self.convs[i](x, edge_index)
        # 2. Readout layer
        x = self.pool(x, batch)
        #print('pool batch',batch)# [batch_size, hidden_channels]
        #print('pool' , x.shape)

        # 3. Apply a final classifier
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x=self.bns[self.num_conv_layers](x)
        
        
        out = self.linear_stack(x)
        #print('prediction',out.shape)
        
        return out
    
def train_loop(dataloader, model, loss_fn, optimizer,dev):
    model.train()
    size = len(dataloader.dataset)
    num_batches=len(dataloader)
    i_train=0
    tr_loss=0
    for data in dataloader:# Iterate in batches over the training dataset.
        print("\n\n==================================================================\n\n")
        print('train loop counter',i_train)
        print("\n\n-------------------------------------------------------------------\n\n")
        #data=data.to(dev)
        out = model(data)# Perform a single forward pass.
        #print('In train before loss out shape',out.shape, 'y shape',data.y.shape)
        y = torch.cat([torch.reshape(d1.y,(1,)) for d1 in data]).to(out.device)
        loss = loss_fn(out, y)  # Compute the loss.

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i_train += 1


        '''if step % 100 == 0:
            loss, current = loss.item(), step* len(data)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")'''
        loss = loss.item()
        tr_loss += loss
        print(f"loss: {loss:>7f}")
    tr_loss /= num_batches
    return tr_loss
def test_loop(dataloader, model, loss_fn,dev):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    c=np.zeros(1)
    c1=np.zeros(1)
    tst_loss=[]
    with torch.no_grad():
        for data in dataloader:
            #data=data.to(dev)# Iterate in batches over the training/test dataset.
            pred = model(data)
            
            
            y = torch.cat([torch.reshape(d1.y,(1,)) for d1 in data]).to(pred.device)
            #print('test pred shape',torch.argmax(pred2, dim#=1).shape)
            #time.sleep(5)
            #y=data.y
            
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            c=np.append(c,torch.argmax(pred, dim=1).cpu())
            y1=y.detach().clone()
            y1=y1.cpu()
            c1=np.append(c1,y1)

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")
    #return c,c1,correct,jac1_te
    return c[1:],c1[1:],correct,test_loss
