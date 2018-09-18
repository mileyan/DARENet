import torch
import torch.utils.data
import torchvision.transforms as transforms
import Datasets
import numpy as np

import os
import scipy.io as sio


def extract_features_MARS(model, scale_image_size, info_folder, data, extract_features_folder, logger, batch_size=128, workers=4, is_tencrop=False):
    logger.info('Begin extract features')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if is_tencrop:
        logger.info('==> Using TenCrop')
        tencrop = transforms.Compose([
            transforms.Resize([int(x*1.125) for x in scale_image_size]),
            transforms.TenCrop(scale_image_size)])
    else:
        tencrop = None
    transform = transforms.Compose([
        transforms.Resize(scale_image_size),
        transforms.ToTensor(),
        normalize, ])
    train_name_path = os.path.join(info_folder, 'train_name.txt')
    test_name_path = os.path.join(info_folder, 'test_name.txt')
    train_data_folder = os.path.join(data, 'bbox_train')
    test_data_folder = os.path.join(data, 'bbox_test')
    logger.info('Train data folder: '+train_data_folder)
    logger.info('Test data folder: '+test_data_folder)
    logger.info('Begin load train data')
    train_dataloader = torch.utils.data.DataLoader(
        Datasets.MARSEvalDataset(folder=train_data_folder,
                                    image_name_file=train_name_path,
                                    transform=transform, tencrop=tencrop),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    logger.info('Begin load test data')
    test_dataloader = torch.utils.data.DataLoader(
        Datasets.MARSEvalDataset(folder=test_data_folder,
                                    image_name_file=test_name_path,
                                    transform=transform, tencrop=tencrop),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    train_features = extract_features(model, train_dataloader, is_tencrop)
    test_features = extract_features(model, test_dataloader, is_tencrop)
    if os.path.isdir(extract_features_folder) is False:
        os.makedirs(extract_features_folder)

    sio.savemat(os.path.join(extract_features_folder, 'train_features.mat'), {'feature_train_new': train_features})
    sio.savemat(os.path.join(extract_features_folder, 'test_features.mat'), {'feature_test_new': test_features})
    return


def extract_features_Market1501(model, scale_image_size, data, extract_features_folder, logger, batch_size=128, workers=4, is_tencrop=False, gen_stage_features = False):
    logger.info('Begin extract features')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if is_tencrop:
        logger.info('==> Using TenCrop')
        tencrop = transforms.Compose([
            transforms.Resize([int(x*1.125) for x in scale_image_size]),
            transforms.TenCrop(scale_image_size)])
    else:
        tencrop = None
    transform = transforms.Compose([
        transforms.Resize(scale_image_size),
        transforms.ToTensor(),
        normalize, ])
    train_data_folder = os.path.join(data, 'bounding_box_train')
    test_data_folder = os.path.join(data, 'bounding_box_test')
    query_data_folder = os.path.join(data, 'query')
    logger.info('Begin load train data from '+train_data_folder)
    train_dataloader = torch.utils.data.DataLoader(
        Datasets.Market1501EvaluateDataset(folder=train_data_folder, transform=transform, tencrop=tencrop),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    logger.info('Begin load test data from '+test_data_folder)
    test_dataloader = torch.utils.data.DataLoader(
        Datasets.Market1501EvaluateDataset(folder=test_data_folder, transform=transform, tencrop=tencrop),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    logger.info('Begin load query data from '+query_data_folder)
    query_dataloader = torch.utils.data.DataLoader(
        Datasets.Market1501EvaluateDataset(folder=query_data_folder, transform=transform, tencrop=tencrop),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    if not gen_stage_features:
        train_features = extract_features(model, train_dataloader, is_tencrop)
        test_features = extract_features(model, test_dataloader, is_tencrop)
        query_features = extract_features(model, query_dataloader, is_tencrop)
        if os.path.isdir(extract_features_folder) is False:
            os.makedirs(extract_features_folder)

        sio.savemat(os.path.join(extract_features_folder, 'train_features.mat'), {'feature_train_new': train_features})
        sio.savemat(os.path.join(extract_features_folder, 'test_features.mat'), {'feature_test_new': test_features})
        sio.savemat(os.path.join(extract_features_folder, 'query_features.mat'), {'feature_query_new': query_features})
    else:
        # model.gen_stage_features = True

        train_features = extract_stage_features(model, train_dataloader, is_tencrop)
        test_features = extract_stage_features(model, test_dataloader, is_tencrop)
        query_features = extract_stage_features(model, query_dataloader, is_tencrop)
        if os.path.isdir(extract_features_folder) is False:
            os.makedirs(extract_features_folder)

        for i in range(4):
            sio.savemat(os.path.join(extract_features_folder, 'train_features_{}.mat'.format(i + 1)), {'feature_train_new': train_features[i]})
            sio.savemat(os.path.join(extract_features_folder, 'test_features_{}.mat'.format(i + 1)), {'feature_test_new': test_features[i]})
            sio.savemat(os.path.join(extract_features_folder, 'query_features_{}.mat'.format(i + 1)), {'feature_query_new': query_features[i]})

        sio.savemat(os.path.join(extract_features_folder, 'train_features_fusion.mat'), {'feature_train_new': train_features[4]})
        sio.savemat(os.path.join(extract_features_folder, 'test_features_fusion.mat'), {'feature_test_new': test_features[4]})
        sio.savemat(os.path.join(extract_features_folder, 'query_features_fusion.mat'), {'feature_query_new': query_features[4]})



def extract_features_CUHK03(model, scale_image_size, data, extract_features_folder, logger, batch_size=128, workers=4, is_tencrop=False,normalize=None):
    logger.info('Begin extract features')
    if normalize == None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    if is_tencrop:
        logger.info('==> Using TenCrop')
        tencrop = transforms.Compose([
            transforms.Resize([int(x*1.125) for x in scale_image_size]),
            transforms.TenCrop(scale_image_size)])
    else:
        tencrop = None
    transform = transforms.Compose([
        transforms.Resize(scale_image_size),
        transforms.ToTensor(),
        normalize, ])
    train_data_folder = data
    logger.info('Begin load train data from '+train_data_folder)
    train_dataloader = torch.utils.data.DataLoader(
        Datasets.CUHK03EvaluateDataset(folder=train_data_folder, transform=transform, tencrop=tencrop),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    train_features = extract_features(model, train_dataloader, is_tencrop)
    if os.path.isdir(extract_features_folder) is False:
        os.makedirs(extract_features_folder)

    sio.savemat(os.path.join(extract_features_folder, 'train_features.mat'), {'feature_train_new': train_features})
    return

def extract_stage_features(net, dataloader, is_tencrop=False):
    net.eval()
    # we have five stages in total
    features_list = []
    for i in range(5):
        features_list.append([])
    count = 0
    for i, input in enumerate(dataloader):
        if is_tencrop:
            input = input.view((-1, *input.size()[-3:]))
        input_var = torch.autograd.Variable(input, volatile=True)
        features = net(input_var)
        for j in range(5):
            feature = features[j].cpu().data.numpy()
            if is_tencrop:
                feature = feature.reshape((-1, 10, feature.shape[1]))
                feature = feature.mean(1)
            features_list[j].append(feature)
        if is_tencrop:
            count += int(input.size()[0]/10)
        else:
            count += input.size()[0]
        print('finish ' + str(count) + ' images')
    for j in range(5):
        features_list[j] = np.concatenate(features_list[j]).T
    return features_list

def extract_features(net, dataloader, is_tencrop=False):
    net.eval()
    features_list = []
    count = 0
    for i, input in enumerate(dataloader):
        if is_tencrop:
            input = input.view((-1, *input.size()[-3:]))
        input_var = torch.autograd.Variable(input, volatile=True)
        feature = net(input_var)
        feature = feature.cpu().data.numpy()
        if is_tencrop:
            feature = feature.reshape((-1, 10, feature.shape[1]))
            feature = feature.mean(1)
        features_list.append(feature)
        if is_tencrop:
            count += int(input.size()[0]/10)
        else:
            count += input.size()[0]
        print('finish ' + str(count) + ' images')
    return np.concatenate(features_list).T