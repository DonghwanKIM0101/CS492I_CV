from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import argparse

import numpy as np
import shutil
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, models, transforms

import torch.nn.functional as F

from ImageDataLoader import SimpleImageLoader
from models import Res18, Res50, Dense121, Res18_basic

import nsml
from nsml import DATASET_PATH, IS_ON_NSML

NUM_CLASSES = 265

def top_n_accuracy_score(y_true, y_prob, n=5, normalize=True):
    num_obs, num_labels = y_prob.shape
    idx = num_labels - n - 1
    counter = 0
    argsorted = np.argsort(y_prob, axis=1)
    for i in range(num_obs):
        if y_true[i] in argsorted[i, idx+1:]:
            counter += 1
    if normalize:
        return counter * 1.0 / num_obs
    else:
        return counter
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def adjust_learning_rate(opts, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = opts.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)
        
class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, final_epoch):
        # for threshold
        outputs_u_temp = []
        targets_u_temp = []
        prob_threshold = []
        for prob in targets_u:
            prob_threshold.append(max(prob).item())
        threshold = np.quantile(prob_threshold, (linear_rampup(epoch, final_epoch) * -0.7 + 0.7)) # threshold scheduling y = -0.7x + 0.7

        # filter by threshold
        for idx, prob in enumerate(targets_u):
            if (max(prob) >= threshold):
                targets_u_temp.append(prob)
                outputs_u_temp.append(outputs_u[idx])
        targets_u = torch.stack(targets_u_temp)
        outputs_u = torch.stack(outputs_u_temp)

        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
        return Lx, Lu, opts.lambda_u * linear_rampup(epoch, final_epoch)

class WeightEMA(object):
    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr * alpha

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def split_ids(path, ratio):
    with open(path) as f:
        ids_l = []
        ids_u = []
        for i, line in enumerate(f.readlines()):
            if i == 0 or line == '' or line == '\n':
                continue
            line = line.replace('\n', '').split('\t')
            if int(line[1]) >= 0:
                ids_l.append(int(line[0]))
            else:
                ids_u.append(int(line[0]))

    ids_l = np.array(ids_l)
    ids_u = np.array(ids_u)

    perm = np.random.permutation(np.arange(len(ids_l)))
    cut = int(ratio*len(ids_l))
    train_ids = ids_l[perm][cut:]
    val_ids = ids_l[perm][:cut]

    return train_ids, val_ids, ids_u


### NSML functions
def _infer(model, root_path, test_loader=None):
    if test_loader is None:
        test_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(root_path, 'test',
                               transform=transforms.Compose([
                                   transforms.Resize(opts.imResize),
                                   transforms.CenterCrop(opts.imsize),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                               ])), batch_size=opts.batchsize, shuffle=False, num_workers=4, pin_memory=True)
        print('loaded {} test images'.format(len(test_loader.dataset)))

    outputs = []
    s_t = time.time()
    for idx, image in enumerate(test_loader):
        if torch.cuda.is_available():
            image = image.cuda()
        _, probs = model(image)
        output = torch.argmax(probs, dim=1)
        output = output.detach().cpu().numpy()
        outputs.append(output)

    outputs = np.concatenate(outputs)
    return outputs

def bind_nsml(model):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = model.state_dict()
        torch.save(state, os.path.join(dir_name, 'model.pt'))
        print('saved')

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pt'))
        state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(state)
        print('loaded')

    def infer(root_path):
        return _infer(model, root_path)

    nsml.bind(save=save, load=load, infer=infer)


######################################################################
# Options
######################################################################
parser = argparse.ArgumentParser(description='Sample Product200K Training')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N', help='number of start epoch (default: 1)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--steps_per_epoch', type=int, default=30, metavar='N', help='number of steps to train per epoch (-1: num_data//batchsize)')

# basic settings
parser.add_argument('--name',default='FixMixMatch', type=str, help='output model name')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--batchsize', default=40, type=int, help='batchsize')
parser.add_argument('--seed', type=int, default=123, help='random seed')

# basic hyper-parameters
parser.add_argument('--momentum', type=float, default=0.9, metavar='LR', help=' ')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate')
parser.add_argument('--imResize', default=256, type=int, help='')
parser.add_argument('--imsize', default=224, type=int, help='')
parser.add_argument('--ema_decay', type=float, default=0.999, help='ema decay rate (0: no ema model)')

# arguments for logging and backup
parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='logging training status')
parser.add_argument('--save_epoch', type=int, default=50, help='saving epoch interval')

# hyper-parameters for mix-match
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=1000, type=float)
parser.add_argument('--T', default=0.5, type=float)

### DO NOT MODIFY THIS BLOCK ###
# arguments for nsml 
parser.add_argument('--pause', type=int, default=0)
parser.add_argument('--mode', type=str, default='train')
################################

def main():
    global opts, global_step
    opts = parser.parse_args()
    opts.cuda = 0

    global_step = 0

    print(opts)

    # Set GPU
    seed = opts.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_ids
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        opts.cuda = 1
        print("Currently using GPU {}".format(opts.gpu_ids))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")


    # Set model
    model = Dense121(NUM_CLASSES)
    model.eval()

    # set EMA model
    ema_model = Dense121(NUM_CLASSES)
    for param in ema_model.parameters():
        param.detach_()
    ema_model.eval()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    if use_gpu:
        model.cuda()
        ema_model.cuda()

    model_for_test = ema_model # change this to model if ema_model is not used.

    ### DO NOT MODIFY THIS BLOCK ###
    if IS_ON_NSML:
        bind_nsml(model_for_test)
        if opts.pause:
            nsml.paused(scope=locals())
    ################################

    if opts.mode == 'train':
        # set multi-gpu
        if len(opts.gpu_ids.split(',')) > 1:
            model = nn.DataParallel(model)
            ema_model = nn.DataParallel(ema_model)
        model.train()
        ema_model.train()

        # Set dataloader
        train_ids, val_ids, unl_ids = split_ids(os.path.join(DATASET_PATH, 'train/train_label'), 0.2)
        print('found {} train, {} validation and {} unlabeled images'.format(len(train_ids), len(val_ids), len(unl_ids)))
        train_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(DATASET_PATH, 'train', train_ids,
                              transform=transforms.Compose([
                                  transforms.Resize(opts.imResize),
                                  transforms.RandomResizedCrop(opts.imsize),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])),
                                batch_size=opts.batchsize, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        print('train_loader done')

        unlabel_loaderweak = torch.utils.data.DataLoader(
            SimpleImageLoader(DATASET_PATH, 'unlabel', unl_ids,
                              transform=transforms.Compose([
                                  transforms.Resize(opts.imResize),
                                  transforms.RandomResizedCrop(opts.imsize),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])),
                                batch_size=opts.batchsize, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        print('unlabel_loaderweak done')  

        # strong data augmentation: Crop, Horizontal Flip, Vertical Flip, Rotation, Color Jitter, and Cutout
        unlabel_loaderstrong = torch.utils.data.DataLoader(
            SimpleImageLoader(DATASET_PATH, 'unlabel', unl_ids,
                              transform=transforms.Compose([
                                  transforms.Resize(opts.imResize),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  transforms.RandomRotation(90,expand=True),
                                  transforms.RandomResizedCrop(opts.imsize),
                                  transforms.ColorJitter(brightness=(0.2,3),contrast=(0.2,3),saturation=(0.2,3),hue=(-0.3,0.3)),
                                  transforms.ToTensor(),
                                  transforms.RandomErasing(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])),
                                batch_size=opts.batchsize, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        print('unlabel_loaderstrong done')    

        validation_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(DATASET_PATH, 'val', val_ids,
                               transform=transforms.Compose([
                                   transforms.Resize(opts.imResize),
                                   transforms.CenterCrop(opts.imsize),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])),
                               batch_size=opts.batchsize, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
        print('validation_loader done')

        if opts.steps_per_epoch < 0:
            opts.steps_per_epoch = len(train_loader)

        # Set optimizer
        optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=5e-4)
        ema_optimizer= WeightEMA(model, ema_model, lr=opts.lr, alpha=opts.ema_decay)

        # INSTANTIATE LOSS CLASS
        train_criterion = SemiLoss()

        # INSTANTIATE STEP LEARNING SCHEDULER CLASS
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones=[50, 150], gamma=0.1)

        # Train and Validation 
        best_acc = -1
        for epoch in range(opts.start_epoch, opts.epochs + 1):
            # print('start training')
            loss, loss_x, loss_u, avg_top1, avg_top5 = train(opts, train_loader, unlabel_loaderweak, unlabel_loaderstrong, model, train_criterion, optimizer, ema_optimizer, epoch, use_gpu)
            print('epoch {:03d}/{:03d} finished, loss: {:.3f}, loss_x: {:.3f}, loss_un: {:.3f}, avg_top1: {:.3f}%, avg_top5: {:.3f}%'.format(epoch, opts.epochs, loss, loss_x, loss_u, avg_top1, avg_top5))
            # scheduler.step()

            # print('start validation')
            acc_top1, acc_top5 = validation(opts, validation_loader, ema_model, epoch, use_gpu)
            is_best = acc_top1 > best_acc
            best_acc = max(acc_top1, best_acc)
            if is_best:
                print('model achieved the best accuracy ({:.3f}%) - saving best checkpoint...'.format(best_acc))
                if IS_ON_NSML:
                    nsml.save(opts.name + '_best')
                else:
                    torch.save(ema_model.state_dict(), os.path.join('runs', opts.name + '_best'))
            if (epoch + 1) % opts.save_epoch == 0:
                if IS_ON_NSML:
                    nsml.save(opts.name + '_e{}'.format(epoch))
                else:
                    torch.save(ema_model.state_dict(), os.path.join('runs', opts.name + '_e{}'.format(epoch)))

                
def train(opts, train_loader, unlabel_loaderweak, unlabel_loaderstrong, model, criterion, optimizer, ema_optimizer, epoch, use_gpu):
    global global_step

    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_un = AverageMeter()
    
    losses_curr = AverageMeter()
    losses_x_curr = AverageMeter()
    losses_un_curr = AverageMeter()

    weight_scale = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()
    
    model.train()
    
    # nCnt =0 
    out = False
    local_step = 0
    while not out:
        labeled_train_iter = iter(train_loader)
        unlabeledweak_train_iter = iter(unlabel_loaderweak)
        unlabeledstrong_train_iter = iter(unlabel_loaderstrong)
        for batch_idx in range(len(train_loader)):
            try:
                data = labeled_train_iter.next()
                inputs_x, targets_x = data
            except:
                labeled_train_iter = iter(train_loader)       
                data = labeled_train_iter.next()
                inputs_x, targets_x = data
            try:
                data = unlabeledweak_train_iter.next()
                inputs_uw1, inputs_uw2 = data
                data = unlabeledstrong_train_iter.next()
                inputs_us1, inputs_us2 = data
            except:
                unlabeledweak_train_iter = iter(unlabel_loaderweak)
                data = unlabeledweak_train_iter.next()
                inputs_uw1, inputs_uw2 = data
                unlabeledstrong_train_iter = iter(unlabel_loaderstrong)
                data = unlabeledstrong_train_iter.next()
                inputs_us1, inputs_us2 = data         
        
            batch_size = inputs_x.size(0)
            # Transform label to one-hot
            classno = NUM_CLASSES 
            targets_org = targets_x
            targets_x = torch.zeros(batch_size, classno).scatter_(1, targets_x.view(-1,1), 1)        
            
            if use_gpu :
                inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda()
                inputs_uw1, inputs_uw2 = inputs_uw1.cuda(), inputs_uw2.cuda()
                inputs_us1, inputs_us2 = inputs_us1.cuda(), inputs_us2.cuda()    
            
            with torch.no_grad():
                # compute guessed labels of unlabel samples
                embed_uw1, pred_uw1 = model(inputs_uw1)
                embed_uw2, pred_uw2 = model(inputs_uw2)
                pred_u_all = (torch.softmax(pred_uw1, dim=1) + torch.softmax(pred_uw2, dim=1)) / 2
                pt = pred_u_all**(1/opts.T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()
                
            # mixup
            all_inputs = torch.cat([inputs_x, inputs_us1, inputs_us2], dim=0) #labeled data and strongly augmented unlabeled data
            all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0) #labeled data and weakly augmented unlabeled data
            
            lamda = np.random.beta(opts.alpha, opts.alpha)        
            lamda= max(lamda, 1-lamda)    
            newidx = torch.randperm(all_inputs.size(0))
            input_a, input_b = all_inputs, all_inputs[newidx]
            target_a, target_b = all_targets, all_targets[newidx]        
            
            mixed_input = lamda * input_a + (1 - lamda) * input_b
            mixed_target = lamda * target_a + (1 - lamda) * target_b
            
            # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
            mixed_input = list(torch.split(mixed_input, batch_size))
            mixed_input = interleave(mixed_input, batch_size)

            optimizer.zero_grad()
            
            fea, logits_temp = model(mixed_input[0])
            logits = [logits_temp]
            for newinput in mixed_input[1:]:
                fea, logits_temp = model(newinput)
                logits.append(logits_temp)        
                
            # put interleaved samples back
            logits = interleave(logits, batch_size)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)            
            
            loss_x, loss_un, weigts_mixing = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch+batch_idx/len(train_loader), opts.epochs)
            loss = loss_x + weigts_mixing * loss_un

            losses.update(loss.item(), inputs_x.size(0))
            losses_x.update(loss_x.item(), inputs_x.size(0))
            losses_un.update(loss_un.item(), inputs_x.size(0))
            weight_scale.update(weigts_mixing, inputs_x.size(0))

            losses_curr.update(loss.item(), inputs_x.size(0))
            losses_x_curr.update(loss_x.item(), inputs_x.size(0))
            losses_un_curr.update(loss_un.item(), inputs_x.size(0))
                    
            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()
            ema_optimizer.step()
            
            with torch.no_grad():
                # compute guessed labels of unlabel samples
                embed_x, pred_x1 = model(inputs_x)

            if IS_ON_NSML and global_step % opts.log_interval == 0:
                nsml.report(step=global_step, loss=losses_curr.avg, loss_x=losses_x_curr.avg, loss_un=losses_un_curr.avg)
                losses_curr.reset()
                losses_x_curr.reset()
                losses_un_curr.reset()

            acc_top1b = top_n_accuracy_score(targets_org.data.cpu().numpy(), pred_x1.data.cpu().numpy(), n=1)*100
            acc_top5b = top_n_accuracy_score(targets_org.data.cpu().numpy(), pred_x1.data.cpu().numpy(), n=5)*100    
            acc_top1.update(torch.as_tensor(acc_top1b), inputs_x.size(0))        
            acc_top5.update(torch.as_tensor(acc_top5b), inputs_x.size(0))   

            local_step += 1
            global_step += 1

            if local_step >= opts.steps_per_epoch:
                out = True
                break
        
    return losses.avg, losses_x.avg, losses_un.avg, acc_top1.avg, acc_top5.avg


def validation(opts, validation_loader, model, epoch, use_gpu):
    model.eval()
    avg_top1= 0.0
    avg_top5 = 0.0
    nCnt =0 
    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader):
            inputs, labels = data
            if use_gpu :
                inputs = inputs.cuda()
            nCnt +=1
            embed_fea, preds = model(inputs)

            acc_top1 = top_n_accuracy_score(labels.numpy(), preds.data.cpu().numpy(), n=1)*100
            acc_top5 = top_n_accuracy_score(labels.numpy(), preds.data.cpu().numpy(), n=5)*100
            avg_top1 += acc_top1
            avg_top5 += acc_top5

        avg_top1 = float(avg_top1/nCnt)   
        avg_top5= float(avg_top5/nCnt)   
    
    if IS_ON_NSML:
        nsml.report(step=epoch, avg_top1=avg_top1, avg_top5=avg_top5)

    return avg_top1, avg_top5



if __name__ == '__main__':
    main()

