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
mydata=np.loadtxt(root_path+'\\training_data\\total_s1_s2_s3_sm_se_tao5_currentporos.txt')[0:8316*10]
T=mydata[:,3]/mydata[:,4]
L=(2.*mydata[:,1]-mydata[:,0]-mydata[:,2])/(mydata[:,0]-mydata[:,2])
s_ave=np.average(mydata[:,0:3],axis=1)
J3_se3=(mydata[:,0]-s_ave)*(mydata[:,1]-s_ave)*(mydata[:,2]-s_ave)/(mydata[:,4]**3)
J3=np.cbrt((mydata[:,0]*mydata[:,1]*mydata[:,2]))**3
# (np.sum(mydata[:,-6:-1]**20,axis=1))**(1/20)
traintotal=np.transpose(np.vstack([mydata[:,3],(np.sum(mydata[:,-6:-1]**20,axis=1))**(1/20),T,mydata[:,5],mydata[:,-1]]))
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

# train_index=np.loadtxt(root_path+'\\PINN_result\\training_index.txt',dtype=int)
# val_index=np.loadtxt(root_path+'\\PINN_result\\val_index.txt',dtype=int)

indata=traintotal[train_index]
valdata=traintotal[val_index]
dstrain=np.loadtxt(root_path+'\\training_data\\total_dstrain.txt')[0:8316*10]
dstrain[:,3:6]=dstrain[:,3:6]/2
dstrain=dstrain.astype(np.float32)
indstrain=dstrain[train_index]
valdstrain=dstrain[val_index]
inpsumtaops=np.loadtxt(root_path+'\\training_data\\total_p(sum(tao^20))^0_05ps.txt')[train_index].astype(np.float32)
inpJ3se3ps=np.loadtxt(root_path+'\\training_data\\total_p(J3_se^3)ps.txt')[train_index].astype(np.float32)
inptaomaxps=np.loadtxt(root_path+'\\training_data\\total_p(abs(taomax))ps.txt')[train_index].astype(np.float32)
valpsumtaops=np.loadtxt(root_path+'\\training_data\\total_p(sum(tao^20))^0_05ps.txt')[val_index].astype(np.float32)
valpJ3se3ps=np.loadtxt(root_path+'\\training_data\\total_p(J3_se^3)ps.txt')[val_index].astype(np.float32)
valptaomaxps=np.loadtxt(root_path+'\\training_data\\total_p(abs(taomax))ps.txt')[val_index].astype(np.float32)

indata=indata.astype(np.float32)
data_mean=np.mean(indata,axis=0)
data_std=np.std(indata,axis=0)
ss = StandardScaler(copy=True, with_mean=True, with_std=True)
in_data = ss.fit_transform(indata)

valdata=valdata.astype(np.float32)
valdata=(valdata-data_mean)/data_std


# calculating with GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

train_xy=torch.from_numpy(in_data[:,:-1]).cuda(device)
train_dstrain=torch.from_numpy(indstrain).cuda(device)
train_psumtaops=torch.from_numpy(inpsumtaops).cuda(device)
train_pJ3se3ps=torch.from_numpy(inpJ3se3ps).cuda(device)
train_ptaomaxps=torch.from_numpy(inptaomaxps).cuda(device)
zs=torch.from_numpy(in_data[:,-1])
zs=torch.reshape(zs,[len(zs),1]).cuda(device)

val_xy=torch.from_numpy(valdata[:,:-1]).cuda(device)
val_dstrain=torch.from_numpy(valdstrain).cuda(device)
val_psumtaops=torch.from_numpy(valpsumtaops).cuda(device)
val_pJ3se3ps=torch.from_numpy(valpJ3se3ps).cuda(device)
val_ptaomaxps=torch.from_numpy(valptaomaxps).cuda(device)
val_zs=torch.from_numpy(valdata[:,-1])
val_zs=torch.reshape(val_zs,[len(val_zs),1]).cuda(device)

train_data = Data.TensorDataset(train_xy,train_dstrain,train_psumtaops,train_pJ3se3ps,train_ptaomaxps,zs)
train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=1024,
                               shuffle=True,
                               num_workers=0)


class myf(nn.Module):
    def __init__(self):
        super(myf, self).__init__()
        n1=200
        n2=100
        n3=50
        n4=20
        pdrop=0.1

        self.pre1=nn.Sequential(
            nn.Linear(in_features=4, out_features=20, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=20, out_features=40, bias=True),
            nn.Tanh(),
            # nn.Linear(in_features=20, out_features=10, bias=True),
            # nn.Tanh(),
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
            nn.Linear(in_features=40, out_features=n1, bias=True),
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
            # nn.Sigmoid(),
            # nn.LeakyReLU(0.2),
            # nn.Linear(in_features=n3, out_features=n4, bias=True),
            # nn.Dropout(pdrop),
            # nn.Tanh(),
            # # nn.Sigmoid(),
            # # nn.LeakyReLU(0.2),
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


# # # Load a previously trained model
# log_path= 'D:\program\PythonProject\GTN-ANN/void_crystal/log/FNN_PINN_training_result/FNN_sm_sumtao^20_J3se3_taomax4-20-40-200-100-1tanh.pt'
# if os.path.exists(log_path):
#     print('load successfully')
#     my.load_state_dict(torch.load(log_path)['netpara'])

opt = SGD(my.parameters(),lr=0.02)
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
    zpr = my(xy[:, 0:4])
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

def PINNloss(xy,dstrain,px1ps,px2ps,px3ps,z):
    xy=Variable(xy,requires_grad=True)
    zpr=my(xy)
    weight = torch.ones(zpr.size()).cuda(device) # Here weight means the derivative multiplied by the weight as the output
    dzpr=torch.autograd.grad(zpr,xy,grad_outputs=weight,create_graph=True)[0]
    pfpx0 = dzpr[:, 0] * data_std[-1] / data_std[0] # Partial derivative of porosity to sm
    pfpx1 = dzpr[:, 1] * data_std[-1] / data_std[1] # Partial derivative of porosity to sum(tao^20)^0.05
    pfpx2 = dzpr[:, 2] * data_std[-1] / data_std[2] # Partial derivative of porosity to J3/se^3
    pfpx3 = dzpr[:, 3] * data_std[-1] / data_std[3] # Partial derivative of porosity to taomax

    px0ps=torch.zeros([len(xy),6],dtype=torch.float32).cuda(device)
    px0ps[:,0:3]=1./3.
    dstrain_pre=torch.zeros([len(xy),6]).cuda(device)
    for i in range(len(xy)):
        dstrain_pre[i,:]=-(pfpx0[i]*px0ps[i,:]+pfpx1[i]*px1ps[i,:]+pfpx2[i]*px2ps[i,:]+pfpx3[i]*px3ps[i,:])

    dstrain_n=torch.zeros([len(xy),6]).cuda(device) # n denotes a vector with absolute value 1
    dstrain_n_pre=torch.zeros([len(xy),6]).cuda(device)
    # dstrain_maxindex=torch.zeros([len(xy)],dtype=torch.int).cuda(device)
    for i in range(len(xy)):
        dstrain_n[i,:]=dstrain[i,:]/(torch.sum(dstrain[i,0:3]**2)+2*torch.sum(dstrain[i,3:6]**2))**0.5
        dstrain_n_pre[i, :] = dstrain_pre[i, :] / (torch.sum(dstrain_pre[i, 0:3] ** 2)+2*torch.sum(dstrain_pre[i, 3:6] ** 2)) ** 0.5
        # dstrain_maxindex[i]=torch.argmax(abs(dstrain[i,:]))
        # dstrain_n[i,:]=dstrain[i,:]/abs(dstrain[i,dstrain_maxindex[i]])
        # dstrain_n_pre[i, :] = dstrain_pre[i, :] / abs(dstrain_pre[i,dstrain_maxindex[i]])
    a=dstrain_n-dstrain_n_pre
    delta_dstrain_n_frobinius2=torch.sum(a[:,0:3]**2,axis=1)+2*torch.sum(a[:,3:6]**2,axis=1)
    err1=torch.mean(delta_dstrain_n_frobinius2)
    err2=torch.mean((z-zpr)**2) # The returned loss must be a scalar
    # print(err1)
    # if(epoch==0):
    #     k1=0.1
    # elif(np.sqrt(train_pinntotal.item())>0.7):
    #     k1=0.1
    # else:
    #     k1=0.3*(1+epoch/30)
    # k1=+0.01*(1+int(epoch/10.))
    k1=1
    # print(k1)
    return err2, err1, err2 + k1* err1

def PINNcheck(xy,dstrain,px1ps,px2ps,px3ps,z):
    xy=Variable(xy,requires_grad=True)
    zpr=my(xy)
    weight = torch.ones(zpr.size()).cuda(device) # Here weight means the derivative multiplied by the weight as the output
    dzpr=torch.autograd.grad(zpr,xy,grad_outputs=weight,create_graph=True)[0]
    pfpx0 = dzpr[:, 0] * data_std[-1] / data_std[0] # Partial derivative of porosity to sm
    pfpx1 = dzpr[:, 1] * data_std[-1] / data_std[1] # Partial derivative of porosity to sum(tao^20)^0.05
    pfpx2 = dzpr[:, 2] * data_std[-1] / data_std[2] # Partial derivative of porosity to J3/se^3
    pfpx3 = dzpr[:, 3] * data_std[-1] / data_std[3] # Partial derivative of porosity to taomax

    px0ps=torch.zeros([len(xy),6],dtype=torch.float32).cuda(device)
    px0ps[:,0:3]=1./3.
    dstrain_pre=torch.zeros([len(xy),6]).cuda(device)
    for i in range(len(xy)):
        dstrain_pre[i,:]=-(pfpx0[i]*px0ps[i,:]+pfpx1[i]*px1ps[i,:]+pfpx2[i]*px2ps[i,:]+pfpx3[i]*px3ps[i,:])

    dstrain_n=torch.zeros([len(xy),6]).cuda(device) # n denotes a vector with absolute value 1
    dstrain_n_pre=torch.zeros([len(xy),6]).cuda(device)
    dstrain_maxindex=torch.zeros([len(xy)],dtype=torch.int).cuda(device)
    for i in range(len(xy)):
        dstrain_n[i,:]=dstrain[i,:]/(torch.sum(dstrain[i,0:3]**2)+2*torch.sum(dstrain[i,3:6]**2))**0.5
        dstrain_n_pre[i, :] = dstrain_pre[i, :] / (torch.sum(dstrain_pre[i, 0:3] ** 2)+2*torch.sum(dstrain_pre[i, 3:6] ** 2)) ** 0.5
        # dstrain_maxindex[i]=torch.argmax(abs(dstrain[i,:]))
        # dstrain_n[i,:]=dstrain[i,:]/abs(dstrain[i,dstrain_maxindex[i]])
        # dstrain_n_pre[i, :] = dstrain_pre[i, :] / abs(dstrain_pre[i,dstrain_maxindex[i]])
    a=dstrain_n-dstrain_n_pre
    delta_dstrain_n_frobinius2=torch.sum(a[:,0:3]**2,axis=1)+2*torch.sum(a[:,3:6]**2,axis=1)
    err1=torch.mean(delta_dstrain_n_frobinius2)
    # err2=torch.mean((z-zpr)**2) # The returned loss must be a scalar
    return torch.max(abs(dstrain_n-dstrain_n_pre)),err1

def generate_use_data_han2013(fpros, sige, taomax):
    alp, q1, q2 = 6.456, 1.471, 1.325
    A = 2 * alp * fpros / 45
    B = 2 * q1 * fpros
    C = q2 * 0.15 ** 0.5
    D = 1 + q1 ** 2 * fpros ** 2
    sigm = np.arccosh((D - taomax ** 2 - A * sige ** 2) / B) / C
    return sigm

loss_function = nn.MSELoss().cuda(device)

train_MSE_loss_log=[]
epoch_log=[]
epoch=0
my = my.eval()
valloss = mymseloss(val_xy, val_zs)
trainloss = mymseloss(train_xy, zs)
train_maxerr, train_pinntotal = PINNcheck(train_xy, train_dstrain, train_psumtaops, train_pJ3se3ps, train_ptaomaxps, zs)
val_maxerr, val_pinntotal = PINNcheck(val_xy, val_dstrain, val_psumtaops, val_pJ3se3ps, val_ptaomaxps, val_zs)
print( 'train loss:', np.sqrt(trainloss.item()), 'val loss:', np.sqrt(valloss.item()))
print('train PINN loss:', np.sqrt(train_pinntotal.item()), 'val PINN loss:', np.sqrt(val_pinntotal.item()))
print('maxerr_train:', train_maxerr, 'maxerr_val:', val_maxerr)
train_MSE_loss_log.append(np.array([trainloss.item(),valloss.item(),train_pinntotal.item(),val_pinntotal.item()]))
epoch_log.append(epoch)
my = my.train()


for epoch in range(500):

    for xy,dstrain,px1ps,px2ps,px3ps, z in train_loader:
        z_pr = my(xy[:, 0:4])
        # z_pr =my(xy[:, 0:3], xy[:, 3:5])
        # z_pr = my(xy[:,0:3],torch.reshape(xy[:,-2],[len(xy),1]))
        # dzxpr=torch.autograd.grad()

        # loss = loss_function(z_pr,z)

        loss=PINNloss(xy, dstrain, px1ps, px2ps, px3ps, z)[2]

        # loss=my_weight_mseloss(xy,z)
        if(str(loss.item())=='nan'):
            print('error')
        if(abs(loss.item())>10000):
            break

        opt.zero_grad()
        loss.backward()
        opt.step()
    # if (abs(loss.item()) > 10000):
    #     print('Training suspension')
    #     break
    if (epoch+1)%5==0:

        # calculate validation loss
        my=my.eval()
        valloss = mymseloss(val_xy, val_zs)
        trainloss=mymseloss(train_xy,zs)
        train_maxerr,train_pinntotal=PINNcheck(train_xy,train_dstrain,train_psumtaops,train_pJ3se3ps,train_ptaomaxps,zs)
        val_maxerr,val_pinntotal = PINNcheck(val_xy, val_dstrain, val_psumtaops, val_pJ3se3ps, val_ptaomaxps, val_zs)
        print(epoch,'train loss:',np.sqrt(trainloss.item()),'val loss:',np.sqrt(valloss.item()))
        # print(epoch, 'train PINN loss:', np.sqrt(train_pinntotal.item()), 'val PINN loss:', np.sqrt(val_pinntotal.item()))
        # print(epoch,'maxerr_train:',train_maxerr,'maxerr_val:',val_maxerr)
        # train_MSE_loss_log.append(np.array([trainloss.item(),valloss.item(),train_pinntotal.item(),val_pinntotal.item()]))
        # epoch_log.append(epoch)
        my = my.train()

save_training_result=False
if(save_training_result==True):
    dir_path=root_path+'\\PINN_result'
    np.savetxt(dir_path+'/training_index.txt',train_index)
    np.savetxt(dir_path+'/val_index.txt',val_index)
    # np.savetxt(dir_path+'/PINN_sm_se_J3se3_taomax_train_val_mse_loss.txt',train_MSE_loss_log)
    save_file = dir_path+'/PINN_sm_sumtao^20_J3se3_taomax4-20-40-200-100-1tanh.pt'
    torch.save({'netpara': my.state_dict()}, save_file)
    np.savetxt(dir_path+'/PINN_sm_sumtao^20_J3se3_taomax_datamean_std.txt',np.vstack([data_mean,data_std]))




