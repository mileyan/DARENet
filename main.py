import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
import Datasets


import extract_features
import my_logger
from models import dare_models

parser = argparse.ArgumentParser()
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='dare_R',
                    help='model architecture: (default: dare_R)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--crop_size', default=[224, 224], type=int, nargs='+')
parser.add_argument('--extract_features', action='store_true')
parser.add_argument('--gen_stage_features', action='store_true')
parser.add_argument('--extract_features_folder', type=str, default='features/triplet_features')
parser.add_argument('--info_folder', type=str, default='~/Datasets/')
parser.add_argument('--checkpoint_folder', type=str, default='checkpoint/defaults')
parser.add_argument('--num_sample_persons', type=int, default=18, help='num of sampled persons, (default=18)')
parser.add_argument('--num_sample_imgs', type=int, default=4, help='num of sampled images for each person, (default=4)')
parser.add_argument('--mean_loss', action='store_true', help='loss divides batch size')
parser.add_argument('--dataset', type=str, choices=['MARS', 'Market1501','Duke','CUHK03'], default='MARS')
parser.add_argument('--margin', type=float, default=2.)
parser.add_argument('--log_path', type=str, default='logs/triplet/defaults.log',
                    help='log path, (default: logs/triplet/defaults.log)')
parser.add_argument('--ten_crop', action='store_true')
parser.add_argument('--random_mask', action='store_true')

## decay lr
parser.add_argument('--lr_decay_point', type=int, default=30000, help='learning rate decay point, default:30000')
parser.add_argument('--max_iter', type=int, default=60000, help='maximum iterations, default:60000')

# resume
parser.add_argument('--start_iteration', type=int, default=0, help='start iteration, default:0')
parser.add_argument('--eps', type=float, default=1e-8, help='adam params, default:1e-8')


def main():
    global args, best_prec1
    args = parser.parse_args()
    log_handler = my_logger.setup_logger(args.log_path)
    for key, value in sorted(vars(args).items()):
        log_handler.info(str(key) + ': ' + str(value))

    # best_prec1 = 0
    best_loss = 999999.
    iter_count = 0

    # pooling size
    gap_size = [x // 32 for x in args.crop_size]
    # load resent
    if args.pretrained:
        log_handler.info("=> using pre-trained model '{}'".format(args.arch))
    else:
        log_handler.info("=> create model '{}'".format(args.arch))

    model = getattr(dare_models, args.arch)(pretrained=args.pretrained, gap_size=gap_size, gen_stage_features=args.gen_stage_features)
    # model.gen_stage_features = args.gen_stage_features

    model = nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    log_handler.info('Criterion type: Optimized Batch Hard Mining')
    criterion = OptiHardTripletLoss(mean_loss=args.mean_loss, margin=args.margin, eps=args.eps).cuda()

    log_handler.info('Loss type: Adam')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    # Best loss not best predict
    if args.resume:
        if os.path.isfile(args.resume):
            log_handler.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            try:
                args.start_iteration = checkpoint['iterations']
            except:
                args.start_iteration = 0

            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log_handler.info("=> loaded checkpoint '{}' "
                             .format(args.resume))
        else:
            log_handler.error("=> no checkpoint found at '{}'".format(args.resume))

    iter_count = args.start_iteration
    cudnn.benchmark = True

    if args.extract_features:
        if args.dataset == 'MARS':
            extract_features.extract_features_MARS(model, args.crop_size, args.info_folder, args.data,
                                                   args.extract_features_folder, log_handler,
                                                   batch_size=args.batch_size,
                                                   workers=args.workers, is_tencrop=args.ten_crop)
        elif args.dataset == 'Market1501' or args.dataset == 'Duke':
            extract_features.extract_features_Market1501(model, args.crop_size, args.data,
                                                         args.extract_features_folder, log_handler,
                                                         batch_size=args.batch_size,
                                                         workers=args.workers,
                                                         is_tencrop=args.ten_crop,
                                                         gen_stage_features=args.gen_stage_features)
        else:
            extract_features.extract_features_CUHK03(model, args.crop_size, args.data,
                                                     args.extract_features_folder, log_handler,
                                                     batch_size=args.batch_size,
                                                     workers=args.workers, is_tencrop=args.ten_crop)
        log_handler.info('Finish Extracting Features')
        return

    # split dataset for validation and training
    assert os.path.isdir(args.data)
    train_person_ids = os.listdir(args.data)
    log_handler.info('Number of people in the training set: ' + str(len(train_person_ids)))

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    scale_image_size = [int(x * 1.125) for x in args.crop_size]

    train_dataset = Datasets.TrainingDataset(
        data_folder=args.data,
        person_ids=train_person_ids,
        num_sample_persons=args.num_sample_persons,
        num_sample_imgs=args.num_sample_imgs,
        transform=transforms.Compose([
            transforms.Resize(scale_image_size),
            transforms.RandomCrop(args.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), random_mask=args.random_mask)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    if not os.path.isdir(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)
    log_handler.info('Checkpoint folder: ' + str(args.checkpoint_folder))

    while iter_count < args.max_iter:
        # train for one epoch
        loss_train, iter_count = train(train_loader, model, criterion, optimizer, iter_count, log_handler)

        # remember best prec@1 and save checkpoint
        is_best = loss_train < best_loss
        best_loss = min(loss_train, best_loss)
        save_checkpoint({
            'iterations': iter_count,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, folder=args.checkpoint_folder)


def train(train_loader, model, criterion, optimizer, iter_count, log_handler):
    """Train Function"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, input in enumerate(train_loader):
        adjust_lr_adam(optimizer, iter_count)
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(input)

        # compute output
        outs = model(input_var)
        if type(outs) is list or type(outs) is tuple:
            loss_list = []
            for out in outs:
                loss_list.append(
                    criterion(out, num_sample_persons=args.num_sample_persons, num_sample_imgs=args.num_sample_imgs)
                )
            loss = sum(loss_list)
            losses.update(loss.data[0], input.size(0))
        else:
            loss = criterion(outs, num_sample_persons=args.num_sample_persons, num_sample_imgs=args.num_sample_imgs)
            losses.update(loss.data[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log_handler.info('Iter [{0}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                iter_count, batch_time=batch_time,
                data_time=data_time, loss=losses))
        iter_count += 1

    return losses.avg, iter_count


def save_checkpoint(state, is_best, folder, filename='checkpoint.pth.tar'):
    filename = os.path.join(folder, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(folder, 'model_best.pth.tar'))


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


def adjust_lr_adam(optimizer, iter_count):
    if iter_count < args.lr_decay_point:
        lr = args.lr
        betas = (0.9, 0.999)
    else:
        lr = args.lr * (0.001 ** (1. * (iter_count - args.lr_decay_point) / (args.max_iter - args.lr_decay_point)))
        betas = (0.5, 0.999)
    for index, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr
        param_group['betas'] = betas


class OptiHardTripletLoss(torch.nn.Module):
    def __init__(self, margin=2., mean_loss=True, eps=1e-8):
        super(OptiHardTripletLoss, self).__init__()
        self.margin = margin
        self.mean_loss = mean_loss
        self.eps = eps

    def forward(self, features, num_sample_persons, num_sample_imgs):
        loss_list = []

        D = features.mm(features.transpose(-2, -1))
        norms = D.diag().expand(features.size(0), features.size(0))
        D = norms + norms.transpose(-2, -1) - 2. * D
        D = D + self.eps
        D = torch.sqrt(D)
        for i in range(D.size(0)):
            person_id = int(i / num_sample_imgs)

            # same person
            temp_same_person_loss = torch.max(D[i][person_id * num_sample_imgs:(person_id + 1) * num_sample_imgs])
            # different person

            if person_id == 0:
                temp_diff_person_loss = torch.min(D[i][(person_id + 1) * num_sample_imgs:])
            elif person_id == num_sample_persons - 1:
                temp_diff_person_loss = torch.min(D[i][0:person_id * num_sample_imgs])
            else:
                temp_diff_person_loss = torch.min(
                    torch.cat((D[i][0:person_id * num_sample_imgs], D[i][(person_id + 1) * num_sample_imgs:])))
            loss = F.softplus(self.margin + temp_same_person_loss - temp_diff_person_loss)
            loss_list.append(loss)
        if self.mean_loss:
            return sum(loss_list) / float(features.size()[0])
        return sum(loss_list)


if __name__ == '__main__':
    main()
