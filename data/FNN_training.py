import torch
import torch.nn as nn
from torch.optim import SGD
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import os
from sklearn.preprocessing import StandardScaler

root_path=os.path.dirname(__file__)
mydata=np.loadtxt(root_path+'\\training_data\\total_s1_s2_s3_sm_se_tao5_currentporos.txt')
# T=np.reshape(T,[len(T),1])
T=mydata[:,3]/mydata[:,4]
L=(2.*mydata[:,1]-mydata[:,0]-mydata[:,2])/(mydata[:,0]-mydata[:,2])
s_ave=np.average(mydata[:,0:3],axis=1)
J3_se3=(mydata[:,0]-s_ave)*(mydata[:,1]-s_ave)*(mydata[:,2]-s_ave)/(mydata[:,4]**3)
J3=np.cbrt((mydata[:,0]*mydata[:,1]*mydata[:,2]))

# (np.sum(mydata[:,-6:-1]**20,axis=1))**(1/20)
traintotal=np.transpose(np.vstack([mydata[:,3],mydata[:,4],mydata[:,5],mydata[:,-1]]))
input_num=3
save_result=True

traintotal=traintotal.astype(np.float32)
a=np.random.rand(int(len(traintotal)*0.2))*len(traintotal)
val_index=[]
for i in a:
    val_index.append(int(i))
val_index=np.unique(val_index).tolist()

train_index=[]
for i in range(0,len(traintotal)):
    if(i not in val_index):
        train_index.append(i)

indata=traintotal[train_index]
valdata=traintotal[val_index]


indata=indata.astype(np.float32)
data_mean=np.mean(indata,axis=0)
data_std=np.std(indata,axis=0)
ss = StandardScaler(copy=True, with_mean=True, with_std=True)
in_data = ss.fit_transform(indata)
# in_data=indata

# validation dataset
valdata=valdata.astype(np.float32)
valdata=(valdata-data_mean)/data_std


# # calculating with GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

train_xy=torch.tensor(in_data[:,:-1]).cuda(device)
zs=torch.tensor(in_data[:,-1])
zs=torch.reshape(zs,[len(zs),1]).cuda(device)

val_xy=torch.tensor(valdata[:,:-1]).cuda(device)
val_zs=torch.tensor(valdata[:,-1])
val_zs=torch.reshape(val_zs,[len(val_zs),1]).cuda(device)

train_data = Data.TensorDataset(train_xy,zs)
train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=256,
                               shuffle=True,
                               num_workers=0)


class myf(nn.Module):
    def __init__(self):
        super(myf, self).__init__()
        n1=200
        n2=100
        n3=40
        n4=20
        pdrop=0.1

        self.pre1=nn.Sequential(
            nn.Linear(in_features=input_num, out_features=20, bias=True),
            nn.Dropout(pdrop),
            nn.Tanh(),
            nn.Linear(in_features=20, out_features=40, bias=True),
            nn.Dropout(pdrop),
            nn.Tanh(),
            nn.Linear(in_features=40, out_features=20, bias=True),
            nn.Dropout(pdrop),
            nn.Tanh(),
        )
        self.pre2=nn.Sequential(
            nn.Linear(in_features=2, out_features=20, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=20, out_features=10, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=10, out_features=10, bias=True),
            nn.Tanh(),
        )
        self.hidden=nn.Sequential(
            nn.Linear(in_features=20, out_features=n1, bias=True),
            nn.Dropout(pdrop),
            nn.Tanh(),
            # nn.Sigmoid(),
            # nn.LeakyReLU(0.2),
            nn.Linear(in_features=n1, out_features=n2, bias=True),
            nn.Dropout(pdrop),
            nn.Tanh(),
            # nn.Sigmoid(),
            # nn.LeakyReLU(0.2),
            # nn.Linear(in_features=n2, out_features=n3, bias=True),
            # nn.Dropout(pdrop),
            # nn.Tanh(),
            # # nn.Sigmoid(),
            # # nn.LeakyReLU(0.2),
            # nn.Linear(in_features=n3, out_features=n4, bias=True),
            # nn.Dropout(pdrop),
            # nn.Tanh(),
            # # # nn.Sigmoid(),
            # # # nn.LeakyReLU(0.2),
        )
        self.regression=nn.Linear(in_features=n2,out_features=1,)

    def forward(self,x1):
        x11=self.pre1(x1)
        # x22=self.pre2(x2)
        # x = self.hidden(torch.hstack([x11,x22]))
        x=self.hidden(x11)
        output = self.regression(x)
        return output

my=myf().cuda(device)

# Calculating the number of trainable parameters for the neural network
# sum(p.numel() for p in my.parameters() if p.requires_grad)

# # # Load a previously trained model
# log_path= 'D:\program\PythonProject/GTN-ANN/void_crystal/log/smsetao3-20-10x2-4-200-150-tanh.pt'
# if os.path.exists(log_path):
#     print('load successfully')
#     my.load_state_dict(torch.load(log_path)['netpara'])


opt = SGD(my.parameters(),lr=0.005)
# opt = torch.optim.Adam(my.parameters(),lr=0.1)

# -----------Self-defined loss functions------------------
def myloss(xy,z,dz_x,dz_y):
    xy=Variable(xy,requires_grad=True)
    zpr=my(xy)
    dzpr=torch.autograd.grad(zpr,xy,grad_outputs=z,create_graph=True)[0]
    dzx_y=dz_x/dz_y
    err3=dzpr[:,0]/dzpr[:,1]-dzx_y
    sqr_err3=torch.square(err3)
    mean_sqr_err3 = torch.mean(sqr_err3)
    err_dz=dzpr-torch.hstack([dz_x,dz_y])
    err_z = z-zpr
    sqr_err1 = torch.square(err_z)
    sqr_err2=torch.square(err_dz)
    mean_sqr_err1 = torch.mean(sqr_err1)
    mean_sqr_err2 = torch.mean(sqr_err2)
    sqrt_mean_sqr_err = torch.sqrt(mean_sqr_err1)+5*torch.sqrt(mean_sqr_err3)
    return sqrt_mean_sqr_err

def mymseloss(xy,z):
    zpr = my(xy[:, 0:input_num])
    # zpr = my(xy[:, 0:3],xy[:, 3:5])
    # zpr=my(xy[:,0:3],torch.reshape(xy[:,-2],[len(xy),1]))
    err_z = z - zpr
    sqr_err1 = torch.square(err_z)
    mean_sqr_err1 = torch.mean(sqr_err1)
    return mean_sqr_err1

def my_weight_mseloss(xy,z):
    zpr=my(xy)
    # a,b=0.18,0.38
    # weight=1.5-0.25*z
    m=10
    b=2*(m+1)/(m-1)
    k=2+b
    weight=k/(z+b)
    err_z = (z - zpr)*weight
    sqr_err1 = torch.square(err_z)
    mean_sqr_err1 = torch.mean(sqr_err1)
    return mean_sqr_err1


def generate_use_data_han2013(fpros, sige, taomax):
    alp, q1, q2 = 6.456, 1.471, 1.325
    A = 2 * alp * fpros / 45
    B = 2 * q1 * fpros
    C = q2 * 0.15 ** 0.5
    D = 1 + q1 ** 2 * fpros ** 2
    sigm = np.arccosh((D - taomax ** 2 - A * sige ** 2) / B) / C
    return sigm

loss_function = nn.MSELoss().cuda(device)

train_loss_log=[]
epoch_log=[]
for epoch in range(0,200):

    for xy, z in train_loader:
        z_pr = my(xy[:, 0:input_num])
        # z_pr =my(xy[:, 0:3], xy[:, 3:5])
        # z_pr = my(xy[:,0:3],torch.reshape(xy[:,-2],[len(xy),1]))
        # dzxpr=torch.autograd.grad()
        loss = loss_function(z_pr,z)
        # loss=my_weight_mseloss(xy,z)
        if(str(loss.item())=='nan'):
            print('error')
        if(abs(loss.item())>10000):
            break

        opt.zero_grad()
        loss.backward()
        opt.step()
        # for p in my.parameters():
        #     p.data.clamp_(_min, _max)
    if (abs(loss.item()) > 10000):
        print('Training suspension')
        break
    if epoch%10==0:
        # calculate validation loss
        with torch.no_grad():
            my=my.eval()
            # valz = my(val_xy)
            # dzxpr=torch.autograd.grad()
            valloss = mymseloss(val_xy, val_zs)
            # trainzs=my(train_xy)
            trainloss=mymseloss(train_xy,zs)
        print(epoch,'train loss:',trainloss.item(),'val loss:',valloss.item())
        train_loss_log.append(np.array([trainloss.item(),valloss.item()]))
        epoch_log.append(epoch)
        my=my.train()

save_training_result=False
if(save_training_result==True):
    dir_path=root_path+'\\FNN_result'
    np.savetxt(dir_path+'/training_index.txt',train_index)
    np.savetxt(dir_path+'/val_index.txt',val_index)
    # np.savetxt(dir_path+'/FNN_sm_se_taomax_train_val_mse_loss.txt',train_MSE_loss_log)
    save_file = dir_path+'/FNN_sm_se_taomax3-20-40-20-200-100-1tanh_drop0_1.pt'
    torch.save({'netpara': my.state_dict()}, save_file)
    np.savetxt(dir_path+'/FNN_sm_se_taomax_datamean_std.txt',np.vstack([data_mean,data_std]))


