
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import random
from torch.utils.data import Dataset, DataLoader
import time

from datetime import datetime
import os




device = 'cuda:4'
# Naming when saving the model
postfix = 'MODEL_NAME'
os.makedirs(f"./checkpoint/pretrained/{postfix}/", exist_ok = True)
os.makedirs(f"./checkpoint/noisy_finetuned/{postfix}/", exist_ok = True)

vars_in = [0,1,2]
vars_out = [0,1,2]

scale_factor = 2.

cnn_channels = 63
cnn_channels_u = 63
num_cnn_layers = 8

groups = 3
device_ids = [4, 3]


####################################################################
##   Dataset      ##################################################
####################################################################

def random_crop():
    r_seed = np.random.randint(1,1e9)
    c_size = np.random.randint(100,101)
    random.seed(r_seed)
    t = transforms.RandomCrop(size=(c_size, c_size)) # can be different for H and W
    return r_seed, t


def prep_data_transform(data, scale_factor = 2.):
    if len(data.shape) == 3:
        lr_father = F.interpolate(data.unsqueeze(0), scale_factor = (1./scale_factor), mode= 'bicubic') #downsize original: this is supposed to be the original input
        lr_father = F.interpolate(lr_father, (data.shape[1], data.shape[2]), mode= 'bicubic').squeeze(0)

        means = [torch.mean(lr_father[x,:,:]).item() for x in range(3)]
        stds = [torch.std(lr_father[x,:,:]).item() for x in range(3)]
        means_inv = [(-means[x]/stds[x]) for x in range(3)]


        return transforms.Compose([
                transforms.Normalize(mean=means, std=stds),
            ]), means, stds, means_inv

    elif len(data.shape) == 4:
        assert data.shape[1] <= 3
        lr_father = F.interpolate(data, scale_factor = (1./scale_factor), mode= 'bicubic') #downsize original: this is supposed to be the original input
        lr_father = F.interpolate(lr_father, (data.shape[-2], data.shape[-1]), mode= 'bicubic')

        means = [torch.mean(lr_father[:, x,:,:]).item() for x in range(3)]
        stds = [torch.std(lr_father[:, x,:,:]).item() for x in range(3)]
        means_inv = [(-means[x]/stds[x]) for x in range(3)]


        return transforms.Compose([
                transforms.Normalize(mean=means, std=stds),
            ]), means, stds, means_inv
    else:
        print("data shape is invalid.", data.shape)


# For Step1. Self-supervised pretraining
class PretrainDataset(Dataset):
    def __init__(self, hr_dir=None, transform=None, scale_factor = 2., data_num=None):
        """Large number of externel paired dataset"""
        self.hr_data = np.load(hr_dir)
        self.hr_data = self.hr_data.transpose((0, 3, 1, 2))
        self.hr_data = torch.from_numpy(self.hr_data)
        self.scale_factor = scale_factor

        assert self.hr_data.shape[-3] == 3
        if transform:
            data_transform, _, _, _ = prep_data_transform(self.hr_data.to(device))
            self.transform = data_transform
        else:
            self.transform = transform

        if data_num:
            self.data_num = min(len(self.hr_data), data_num)
        else:
            self.data_num = len(self.hr_data)

        self.hr_data = self.hr_data[:self.data_num]


    def __len__(self):
        return len(self.hr_data)

    def __getitem__(self, idx):
        hr_sample = self.hr_data[idx]
        hr_sample = hr_sample.to(torch.float32)

        return hr_sample


class NoisyDataGenerator(object):
    def __init__(self, hr_dir=None, transform=None, scale_factor = 2., uniform_noise=(-0.05, 0.05),
                 batch_num=5, instance_num=8):
        self.hr_data = np.load(hr_dir)
        self.transform = transform

        self.scale_factor = scale_factor
        self.uniform_noise = uniform_noise
        self.batch_num = batch_num
        self.instance_num = instance_num

    def __len__(self):
        return len(self.hr_data)

    def make_data_tensor(self):
#         hr_sample = self.hr_data[idx]
        hr_data = self.hr_data.transpose((0, 3, 1, 2))
        hr_data = torch.from_numpy(hr_data).to(torch.float32)
        _, chn, w, h = hr_data.shape

        if self.transform:
            data_transform, _, _, _ = prep_data_transform(hr_data)
            self.transform = data_transform
            hr_data = self.transform(hr_data)

        lr_father = hr_data[:self.batch_num*self.instance_num*2, :, :, :].to(device)

        lr_father = F.interpolate(lr_father, scale_factor = (1./self.scale_factor), mode= 'bicubic')
        lr_father = F.interpolate(lr_father, (hr_data.shape[2], hr_data.shape[3]), mode= 'bicubic')

        lr_son = F.interpolate(lr_father, scale_factor = (1./self.scale_factor), mode= 'bicubic')
        lr_noise = torch.cuda.FloatTensor(lr_son.shape).uniform_(*self.uniform_noise).to(device)
        lr_son = lr_noise*lr_son + lr_son
        lr_son = F.interpolate(lr_son, (hr_data.shape[2], hr_data.shape[3]), mode= 'bicubic')

        lr_son = torch.reshape(lr_son, (self.batch_num, self.instance_num*2, chn, w, h))
        lr_father = torch.reshape(lr_father, (self.batch_num, self.instance_num*2, chn, w, h))

        lr_son_train = lr_son[:, :self.instance_num, :, :, :]
        lr_son_test = lr_son[:, self.instance_num:, :, :, :]
        lr_father_train = lr_father[:, :self.instance_num, :, :, :]
        lr_father_test = lr_father[:, self.instance_num:, :, :, :]

        return lr_son_train, lr_father_train, lr_son_test, lr_father_test



class HRClimateDataset(Dataset):
    """Use it for downscaling for ZSSR type low resolution dataset."""
    def __init__(self, lr_dir=None, hr_dir=None, transform=None, season_dir=None):
        self.hr_data = np.load(hr_dir)
#         self.hr_data = hr_dir
        self.transform = transform

    def __len__(self):
        return len(self.hr_data)

    def __getitem__(self, idx):
        hr_sample  = self.hr_data[idx]
        hr_sample = hr_sample.transpose((2, 0, 1))
        hr_sample = torch.from_numpy(hr_sample).to(torch.float32)

        if self.transform is not None:
            hr_sample = self.transform(hr_sample)

        return hr_sample


class SingleDataset(Dataset):
    """Use it for finetuning on single-instance."""
    def __init__(self, dataset, scale_son = 2., scale_target = 2., scale_father=1., transform=None, crop=False, data_aug=None, uni_noise=False):

        self.hr_data = dataset
        self.transform = transform
        self.data_aug = data_aug
        self.scale_son = scale_son
        self.scale_target = scale_target
        self.scale_father = scale_father
        self.crop = crop
        self.uni_noise = uni_noise

    def __len__(self):
        return len(self.hr_data)

    def __getitem__(self, idx):
        hr_sample  = self.hr_data

        hr_sample = hr_sample.unsqueeze(0)


        if self.uni_noise:
            noise = torch.FloatTensor(hr_sample.shape).uniform_(self.uni_noise[0], self.uni_noise[1])
            hr_sample = noise*hr_sample + hr_sample
            hr_sample[:,:2, :, :] = torch.clamp(hr_sample[:,:2, :, :], min = 1e-11) # clip values

        #downsize original HR this is supposed to be the "original" LR input, factor same as target
        lr_father = F.interpolate(hr_sample, scale_factor = (1./self.scale_target), mode= 'bicubic')
        orig_hr = lr_father
        lr_father = F.interpolate(lr_father, (hr_sample.shape[2], hr_sample.shape[3]), mode= 'bicubic')

        # this step is like augmentation, if multiple LR-HR pairs are required from the "original" LR
        if self.scale_father != 1.:
            lr_father = F.interpolate(lr_father, scale_factor = (1./self.scale_father), mode= 'bicubic')
            lr_father = F.interpolate(lr_father, (hr_sample.shape[2], hr_sample.shape[3]), mode= 'bicubic')

        lr_son = F.interpolate(lr_father, scale_factor = (1./self.scale_son), mode= 'bicubic') #downsize upsized father
        orig_lr = lr_son
        lr_sample = F.interpolate(lr_son, (lr_father.shape[2], lr_father.shape[3]), mode= 'bicubic') #.squeeze(0) #upsize son; interpolate LR son to LR father, whihc now becomes a LR sample to be used
        hr_sample = lr_father.to(torch.float32) #.squeeze(0) #LR father now becomes the HR sample to be used


        if self.transform:
            lr_sample = self.transform(lr_sample)
            orig_lr = self.transform(orig_lr)

            hr_sample = self.transform(hr_sample)
            orig_hr = self.transform(orig_hr)

        if self.data_aug: #see if possible to return multiple augmentations
            hr_sample = self.data_aug(hr_sample)
            lr_sample = self.data_aug(hr_sample)

        if self.crop:
            r_seed, t = random_crop()

            torch.random.manual_seed(r_seed)
            lr_sample = t(lr_sample)
            torch.random.manual_seed(r_seed)
            hr_sample = t(hr_sample)

        return lr_sample, hr_sample, orig_lr, orig_hr



def inv(channel, data, means, stds, log=False):
    mean=(-means[channel]/stds[channel])
    std=(1.0/stds[channel])
    data = (data - mean)/std
    if log and channel!=2:
        data = torch.exp(data)
    return data


def lr_hr(vars_in, vars_loss, lr, hr):

    assert len(vars_in)>0 and len(vars_loss)>0, 'ko'
    if len(lr.shape) == 4:
        if len(vars_in)>1:
            lr_vars = torch.stack([lr[:,x,:,:] for x in vars_in], 1)
            hr_vars = torch.stack([hr[:,x,:,:] for x in vars_in], 1)
        elif len(vars_in) == 1:
            lr_vars = lr[:,vars_in[0],:,:].unsqueeze(1)
            hr_vars = hr[:,vars_in[0],:,:].unsqueeze(1) # input lr, hr with input channels selected

        if len(vars_loss)>1:
            loss_vars = torch.stack([hr[:,x,:,:] for x in vars_loss], 1)
        elif len(vars_loss)==1:
            loss_vars = hr[:,vars_in[0],:,:].unsqueeze(1) # hr, with selected channels to calculate losses

    elif len(lr.shape) == 5:
        if len(vars_in)>1:
            lr_vars = lr[:, :, vars_in, :, :]#torch.stack([lr[:,:,x,:,:] for x in vars_in], 2)
            hr_vars = hr[:, :, vars_in, :, :]#torch.stack([hr[:,:,x,:,:] for x in vars_in], 1)
        elif len(vars_in) == 1:
            lr_vars = lr[:,:,vars_in[0],:,:].unsqueeze(1)
            hr_vars = hr[:,:,vars_in[0],:,:].unsqueeze(1) # input lr, hr with input channels selected

        if len(vars_loss)>1:
            loss_vars = hr[:,:,vars_loss,:,:]#torch.stack([hr[:,:,x,:,:] for x in vars_loss], 1)
        elif len(vars_loss)==1:
            loss_vars = hr[:,:,vars_in[0],:,:].unsqueeze(1) # hr, with selected channels to calculate losses

    else:
        print("lr, hr has invalid shape.")
        return
    return lr_vars, hr_vars, loss_vars


####################################################################
##   Model        ##################################################
####################################################################


class _Residual_Block(nn.Module):
    def __init__(self, groups, cnn_channels):
        self.cnn_channels = cnn_channels
        self.groups = groups
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=self.cnn_channels, out_channels=self.cnn_channels, kernel_size=3, stride=1, padding=1, groups = self.groups, bias=False)
        self.in1 = nn.InstanceNorm2d(self.cnn_channels, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=self.cnn_channels, out_channels=self.cnn_channels, kernel_size=3, stride=1, padding=1, groups = self.groups, bias=False)
        self.in2 = nn.InstanceNorm2d(self.cnn_channels, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output,identity_data) #residual here
        return output


def conv_size(d_shape, k_shape, stride, dilation, padding):
    return np.floor(((d_shape + (2*padding) - (dilation*(k_shape-1))-1)/stride)+1)

def de_conv_size(d_shape, k_shape, stride, dilation, padding, output_padding):
    return ((d_shape-1)*stride -(2*padding) + (dilation*(k_shape-1)) + output_padding + 1)


class _NetG(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(_NetG, self).__init__()

#         self.multihead_attn = nn.MultiheadAttention(321, 3, batch_first=True)

        self.conv_input = nn.Conv2d(in_channels=in_channels, out_channels=cnn_channels, kernel_size=3, stride=1, padding=1, groups = groups, bias=False)
        self.prelu = nn.PReLU(num_parameters=cnn_channels)
        self.conv_input_u = nn.Conv2d(in_channels=in_channels, out_channels=cnn_channels_u, kernel_size=3, stride=1, padding=1, groups = 1, bias=False)
        self.prelu_u = nn.PReLU(num_parameters=cnn_channels)

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.relu_u = nn.LeakyReLU(0.2, inplace=True)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.residual = self.make_layer(_Residual_Block(groups, cnn_channels), num_cnn_layers)
        self.residual_u = self.make_layer(_Residual_Block(1, cnn_channels_u), 4)

        self.do1 = torch.nn.Dropout(p=0.2)
        self.do2 = torch.nn.Dropout(p=0.2)
        self.conv_mid = nn.Conv2d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, stride=1, padding=1, groups = groups, bias=False)
        self.conv_mid_u = nn.Conv2d(in_channels=cnn_channels_u, out_channels=cnn_channels_u, kernel_size=3, stride=1, padding=1, groups = 1, bias=False)

        self.bn_mid = nn.InstanceNorm2d(cnn_channels, affine=True)
        self.bn_mid_u = nn.InstanceNorm2d(cnn_channels_u, affine=True)
        self.s_bn_mid = nn.InstanceNorm2d(cnn_channels, affine=True)
        self.s_bn_mid_u = nn.InstanceNorm2d(cnn_channels_u, affine=True)

        self.conv_output = nn.Conv2d(in_channels=cnn_channels, out_channels=3, kernel_size=3, stride=1, padding=1, groups = groups, bias=False)
        self.conv_output_u = nn.Conv2d(in_channels=cnn_channels_u, out_channels=3, kernel_size=3, stride=1, padding=1, groups = 1, bias=False)
        self.prelu_u = nn.PReLU(num_parameters=3)
#         self.conv_output_1 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)
        return nn.Sequential(*layers)



    def forward(self, x):

        out = self.relu(self.conv_input(x))
        out_u = self.sigmoid((self.relu_u(self.conv_input_u(x))))
        out = out*out_u
        residual = out
        out = self.residual(out) #multiple layers here
        out_mid = self.relu(self.bn_mid(self.conv_mid(out)))

        out_mid_u = self.sigmoid(self.relu_u(self.bn_mid_u(self.conv_mid_u(out_u))))
        out_mid = out_mid*out_mid_u



        out = torch.add(out_mid,residual)
        out = self.conv_output(out)
        out_u = self.sigmoid(self.relu_u(self.conv_output_u(out_mid_u)))
        out = out*out_u


        return out, out_mid


def make_model(in_channels, out_channels, opt_lr = 1e-2):

    model = _NetG(in_channels, out_channels)
    lf = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_lr)

    model = model.to(device)
    criterion = lf.to(device)

    return model, criterion, optimizer


def adjust_learning_rate(opt_lr, epoch, step = 3000):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt_lr * (0.1 ** (epoch // step))
    return lr




in_channels = len(vars_in)
out_channels = len(vars_out)
print(f"make model with {in_channels} in_channels, {out_channels} out_channels")
model, _, optimizer = make_model(in_channels, out_channels, opt_lr = 1e-4)



####################################################################
##   Train        ##################################################
####################################################################


def train_pretrain(model, criterion, optimizer, dataset, test_dataset=None, epochs = 3, batch_size = 4, scale_factor = 2.,
                     pretrained_checkpoint = 0, data_parallel = True, vars_in = vars_in, vars_out = vars_out):


    dataloader = DataLoader(dataset = dataset, batch_size = batch_size)
    if test_dataset:
        test_dataloader =  DataLoader(dataset = test_dataset, batch_size = batch_size)
    if pretrained_checkpoint > 0:
        model.load_state_dict(torch.load(f"./checkpoint/pretrained/{postfix}/epoch_{pretrained_checkpoint}_{postfix}.pt"))
    if data_parallel:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    best_mse_test = 1000.0
    best_epoch = 0
    os.makedirs(f"./checkpoint/pretrained/{postfix}", exist_ok=True)

    for epoch in range(pretrained_checkpoint+1, epochs):
        for iteration, hr in enumerate(dataloader):
            model.train()
            hr = hr.to(device)
            hr = dataset.transform(hr)

            lr_father = F.interpolate(hr, scale_factor = (1./scale_factor), mode= 'bicubic')
            lr_father = F.interpolate(lr_father, (hr.shape[2], hr.shape[3]), mode= 'bicubic')
            lr_son = F.interpolate(lr_father, scale_factor = (1./scale_factor), mode= 'bicubic')
            lr_son = F.interpolate(lr_son, (lr_father.shape[2], lr_father.shape[3]), mode= 'bicubic')


            out, out_mid = model(lr_son)
            mse_loss = criterion(out, lr_father.to(device))
            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()

        if test_dataset:
            model.eval()
            mse_test = 0.0
            for _, batch in enumerate(test_dataloader):
                hr = batch.to(device)
                hr = dataset.transform(hr)
                lr_father = F.interpolate(hr, scale_factor = (1./scale_factor), mode= 'bicubic')
                lr_father = F.interpolate(lr_father, (hr.shape[2], hr.shape[3]), mode= 'bicubic')
                lr_son = F.interpolate(lr_father, scale_factor = (1./scale_factor), mode= 'bicubic')
                lr_son = F.interpolate(lr_son, (lr_father.shape[2], lr_father.shape[3]), mode= 'bicubic')

                lr_son, lr_father, loss_hr = lr_hr(vars_in, vars_out, lr_son, lr_father)
                out, _ = model(lr_son)
                mse_loss = criterion(out, loss_hr.to(device))
                mse_test += mse_loss.item()
            if mse_test < best_mse_test and epoch > 0:
                best_mse_test = mse_test
                best_epoch = epoch
                torch.save(model.module.state_dict(), f"./checkpoint/pretrained/{postfix}/epoch_{epoch}_{postfix}.pt")
            print("===> Epoch[{}] Time {} Validation MSE: {:.5}".format(epoch, str(datetime.now()), mse_test))
    print("===> Best performance at Epoch[{}] Validation MSE: {:.5}".format(best_epoch, best_mse_test))
    return model, best_epoch


# training set for large scale learning
largeScaleDataset = PretrainDataset(hr_dir = f'', transform = True, scale_factor = scale_factor)
largeScaleDataset_test = PretrainDataset(hr_dir = f'', transform = None, data_num=100, scale_factor = scale_factor)

start_time = time.time()
model, best_epoch = train_pretrain(model, nn.MSELoss(), optimizer, dataset = largeScaleDataset,
                         test_dataset=largeScaleDataset_test, epochs=2, batch_size = 16, scale_factor = scale_factor,
                                    pretrained_checkpoint = 0, data_parallel = True)
print("Time Elapsed for Large-scale Learning: {} min".format((time.time()-start_time)//60))

del largeScaleDataset
del largeScaleDataset_test



model, _, optimizer = make_model(in_channels, out_channels, opt_lr = 1e-4)
model.load_state_dict(torch.load(f"./checkpoint/pretrained/{postfix}/epoch_{best_epoch}_{postfix}.pt"))



def train_noisyfinetune(model, criterion, NoisyDataset, epochs = 1500, scale_factor = 2.,
                       vars_in = vars_in, vars_out = vars_out, batch_lr = 1e-4, instance_lr = 1e-2):

    batch_optimizer = torch.optim.Adam(model.parameters(), lr=batch_lr)
    instance_optimizer = torch.optim.Adam(model.parameters(), lr=instance_lr)

    os.makedirs(f"./checkpoint/noisy_finetuned/{postfix}", exist_ok=True)

    lr_son_train, lr_father_train, lr_son_test, lr_father_test = NoisyDataset.make_data_tensor()
    lr_son_train, lr_father_train, loss_hr_train = lr_hr(vars_in, vars_out, lr_son_train, lr_father_train)
    lr_son_test, lr_father_test, loss_hr_test = lr_hr(vars_in, vars_out, lr_son_test, lr_father_test)

    loss_test = []
    best_test_loss = 10000.0
    for epoch in range(epochs):

        # params = model.state_dict()
        test_losses = []
        # Evaluate training loss
        for iteration in range(NoisyDataset.batch_num):
            model.train()
            out_train, out_mid = model(lr_son_train[iteration, :, :, :, :].to(device))
            mse_loss = criterion(out_train, loss_hr_train[iteration, :, :, :, :].to(device))

            instance_optimizer.zero_grad()

            mse_loss.backward(retain_graph=True)

            # Update task-level parameter
            instance_optimizer.step()

            # Compute test loss based on updated task-level parameter
            model.eval()
            with torch.no_grad():
                out_test, out_mid = model(lr_son_test[iteration, :, :, :, :].to(device))
                mse_loss_test = criterion(out_test, loss_hr_test[iteration, :, :, :, :].to(device))
                test_losses.append(mse_loss_test)
            # model.load_state_dict(params)


        batch_optimizer.zero_grad()
        mse_loss_test_sum = sum(test_losses)
        mse_loss_test_sum = torch.autograd.Variable(mse_loss_test_sum, requires_grad=True)
        mse_loss_test_sum.backward()
        batch_optimizer.step()

        if epoch % 50 == 0 and epoch != 0:
            print(f"{str(datetime.now())}: fine tuning epoch {epoch}: loss ", mse_loss_test_sum.item())

        if best_test_loss >  mse_loss_test_sum.item():
            best_test_loss = mse_loss_test_sum.item()
            if epoch > 10:
                torch.save(model.state_dict(), f"./checkpoint/noisy_finetuned/{postfix}/epoch_{epoch}_{postfix}.pt")
        if epoch % 50 == 0 and epoch > 600:
            torch.save(model.state_dict(), f"./checkpoint/noisy_finetuned/{postfix}/epoch_{epoch}_{postfix}.pt")

        loss_test.append(mse_loss_test_sum.item())

    best_epoch = np.argmin(np.array(loss_test))
    return model, loss_test, best_epoch


NoisyDataset = NoisyDataGenerator(hr_dir = f'', transform = True, scale_factor = scale_factor)

start_time = time.time()
model, loss_test, noisy_finetune_best_epoch = train_noisyfinetune(model, nn.MSELoss(), NoisyDataset, epochs = 20)

print("Time Elapsed for Fine Tuning: {} min".format((time.time()-start_time)//60))
print(noisy_finetune_best_epoch)



model.load_state_dict(torch.load(f"./checkpoint/noisy_finetuned/{postfix}/epoch_{noisy_finetune_best_epoch}_{postfix}.pt"))
del NoisyDataset


def train_singleinstance(model, criterion, data, epochs = 10, scale_factor = 2.,
                       vars_in = vars_in, vars_out = vars_out, learning_rate=1e-2):

    data_transform, means, stds, means_inv = prep_data_transform(data)
    if isinstance(scale_factor, float):
        scale_target = scale_factor
        scale_son = scale_factor
    else:
        scale_father, scale_son = scale_factor
    test_set = SingleDataset(data, scale_target = scale_target, scale_son = scale_son, transform = data_transform, crop=False, data_aug=None)
#     lr_son, lr_father = test_set[0]
    lr_son, lr_father, _, lr_father_small = test_set[0]
    gt_hr = data
    lr_son, lr_father, loss_hr = lr_hr(vars_in, vars_out, lr_son, lr_father)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        output, out_mid = model(lr_son.to(device))
        mse_loss = criterion(output, loss_hr.to(device))
        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()
        if epoch % (epochs//5) == 0:
            print(f"epoch {epoch}: loss -", mse_loss.item())

    model.eval()
    output_hr, _ = model(lr_father.to(device))
    bilinear = F.interpolate(lr_father_small, (lr_father.shape[2], lr_father.shape[3]), mode= 'bilinear')

    fin_mses = {}
    fin_mses['lr'] = {}
    fin_mses['hr'] = {}
    fin_mses['out'] = {}
    fin_mses['gt'] = {}
    fin_mses['bilinear'] = {}

    for j, chn in enumerate(vars_out):
        out_inv = (torch.clamp(inv(chn,output_hr[:,j,:,:], means, stds), min = 0)) # compute inverse of out and hr
        hrs_inv = (torch.clamp(inv(chn,lr_father[:,j,:,:], means, stds), min = 0))
        lrs_inv = (torch.clamp(inv(chn,lr_son[:,j,:,:], means, stds), min = 0))
        bilinear_inv = (torch.clamp(inv(chn,bilinear[:,j,:,:], means, stds), min = 0))

        fin_mses['out'][chn] = out_inv
        fin_mses['hr'][chn] = hrs_inv
        fin_mses['bilinear'][chn] = bilinear_inv

    gt_hr[:2, :, :] = torch.clamp(gt_hr[:2, :, :], min = 0)

    results = []
    results_criterion = []
    for chn in vars_out:
        results.append(
            (
                nn.MSELoss()(fin_mses['out'][chn].reshape(gt_hr[chn,:,:].shape), gt_hr[chn,:,:].to(device)).item(), \
                nn.MSELoss()(fin_mses['hr'][chn].to(device).reshape(gt_hr[chn,:,:].shape), gt_hr[chn,:,:].to(device)).item(),\
                nn.MSELoss()(fin_mses['bilinear'][chn].to(device).reshape(gt_hr[chn,:,:].shape), gt_hr[chn,:,:].to(device)).item()
            )
        )
    for chn in vars_out:
        results_criterion.append(
            (
                criterion(fin_mses['out'][chn].reshape(gt_hr[chn,:,:].shape), gt_hr[chn,:,:].to(device)).item(), \
                criterion(fin_mses['hr'][chn].to(device).reshape(gt_hr[chn,:,:].shape), gt_hr[chn,:,:].to(device)).item()
            )
        )

    out_combined = torch.cat([fin_mses['out'][chn] for chn in vars_out], 0).cpu().detach().numpy()
    hr_combined = torch.cat([fin_mses['hr'][chn] for chn in vars_out], 0).cpu().detach().numpy()

    best_outs = [out_combined, hr_combined]

    return results, results_criterion, best_outs



dataset = HRClimateDataset(hr_dir = f'')#change data path


start_time = time.time()

results = []
best_outs_list = []
final_scale_factor = 4.
for i in range(len(dataset)):
    model.load_state_dict(torch.load(f"./checkpoint/noisy_finetuned/{postfix}/epoch_{noisy_finetune_best_epoch}_{postfix}.pt"))
    mse_result, _, best_outs= train_singleinstance(model, nn.MSELoss(), dataset[i], epochs = 20,
                                        scale_factor = final_scale_factor, learning_rate=1e-4)
    results.append(mse_result)
    best_outs_list.append(best_outs)

print("Elapsed time : {:.5} sec ".format(time.time() - start_time))

print(np.mean(np.sqrt(results[:5]).reshape(5,-1), 0))
print(np.mean(np.sqrt(results).reshape(len(dataset),-1), 0))
