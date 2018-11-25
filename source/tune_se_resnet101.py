import os
import yaml
import pickle
import logging
import random

import numpy as np
import torch
torch.backends.cudnn.benchmark=True
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from tensorboardX import SummaryWriter

from datasets import LabeledImages
from torchvision import transforms
from models.se_resnet import se_resnet101


CONFIG_PATH = '/source/config.yaml'


def get_path(rel_path, config):
    return os.path.join(config['TUNING']['ARTIFACTS_ROOT'], rel_path)


class Classifier(nn.Module):
    def __init__(self, n_classes):
        super(Classifier, self).__init__()
        self.n_classes = n_classes
        self.features = se_resnet101()
        self.classifier = nn.Linear(2048, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class ModelWithLoss(nn.Module):
    def __init__(self, classifier):
        super(ModelWithLoss, self).__init__()
        self.classifier = classifier
        self.criterion = nn.CrossEntropyLoss(size_average=False)

    def forward(self, x, labels):
        assert len(x) == len(labels)

        predictions = self.classifier(x)

        all_classes = np.arange(self.classifier.n_classes, dtype=np.int64)
        zero_label = torch.tensor([0]).to(x.device)

        loss = 0
        denominator = 0
        for prediction, positives in zip(predictions, labels):
            negatives = np.setdiff1d(all_classes, positives, assume_unique=True)
            negatives_tensor = torch.tensor(negatives).to(x.device)
            positives_tensor = torch.tensor(positives).to(x.device).unsqueeze(dim=1)

            for positive in positives_tensor:
                indices = torch.cat((positive, negatives_tensor))
                loss = loss + self.criterion(prediction[indices].unsqueeze(dim=0), zero_label)
                denominator += 1

        loss /= denominator

        return loss

    def predict(self, x, top_k):
        predictions = self.classifier(x)
        scores, labels = predictions.sort(dim=1, descending=True)

        pred_scores = np.zeros(shape=(len(scores), top_k), dtype=np.float32)
        pred_labels = labels[:, :top_k].cpu().numpy()

        for i in range(top_k):
            i_scores = torch.cat((scores[:, i:i + 1], scores[:, top_k:]), dim=1)
            pred_scores[:, i] = F.softmax(i_scores, dim=1)[:, 0].cpu().numpy()

        return pred_scores, pred_labels


class AverageMeter(object):
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


def calculate_f2_measure(pred_labels, pred_scores, gt_labels, thresh):
    pred_labels = pred_labels[pred_scores > thresh]
    pred_labels = set(pred_labels.tolist())
    gt_labels = set(gt_labels.tolist())
    tp = pred_labels & gt_labels
    prec = (len(tp) / len(pred_labels)) if len(pred_labels) > 0 else 1
    rec = (len(tp) / len(gt_labels)) if len(gt_labels) > 0 else 1
    f2_measure = ((5 * prec * rec) / (4 * prec + rec)) if (4 * prec + rec) > 0 else 0

    return f2_measure


def validate(model, dataset, indices, batch_size, top_k):
    model.eval()
    sampler = SubsetRandomSampler(indices)
    loader = DataLoader(dataset, batch_size, sampler=sampler, num_workers=0, collate_fn=lambda X: X)

    pred_scores_all = []
    pred_labels_all = []
    gt_labels_all = []

    for samples in loader:
        input_tensor = torch.stack([sample['image'] for sample in samples]).cuda()
        labels = [sample['labels'] for sample in samples]

        with torch.no_grad():
            pred_scores, pred_labels = model.predict(input_tensor, top_k)
            pred_scores_all.extend(pred_scores)
            pred_labels_all.extend(pred_labels)
            gt_labels_all.extend(labels)

    thresholds = np.arange(0.0, 1.0, 0.05, dtype=np.float64)
    results = np.zeros_like(thresholds)

    for i, thresh in enumerate(thresholds):
        f2_measure = 0
        for pred_scores, pred_labels, gt_labels in zip(pred_scores_all, pred_labels_all, gt_labels_all):
            f2_measure += calculate_f2_measure(pred_labels, pred_scores, gt_labels, thresh)
        results[i] = f2_measure / len(gt_labels_all)

    best_idx = np.argmax(results)

    return results[best_idx], thresholds[best_idx]


def adjust_learning_rate(optimizer, epoch, initial_lr, lr_decay, lr_step):
    lr = initial_lr * (lr_decay ** (epoch // lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.load(f)
    os.makedirs(get_path('snapshots', config), exist_ok=True)

    logger = logging.getLogger('tuning')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(get_path('tuning.log', config))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    random.seed(config['TUNING']['SEED'])
    np.random.seed(config['TUNING']['SEED'])
    torch.manual_seed(config['TUNING']['SEED'])

    with open(config['DATASET']['NAME_TO_LABEL_PATH'], 'rb') as f:
        name_to_label = pickle.load(f)
    n_classes = len(name_to_label)
    logger.info('Total number of classes: {}.'.format(n_classes))

    img_path_to_labels = {}
    with open(config['DATASET']['TUNING_LABELS_PATH'], 'r') as f:
        for line in f:
            img_id, names = line.strip().split(',')
            img_path = img_id + '.jpg'
            labels = [name_to_label[name] for name in names.split(' ')]
            img_path_to_labels[img_path] = labels

    content = sorted(img_path_to_labels)
    with open(config['DATASET']['TUNING_LIST_PATH'], 'w') as f:
        for img_path in content:
            labels = img_path_to_labels[img_path]
            line = ' '.join([img_path] + [str(label) for label in labels])
            f.write('{}\n'.format(line))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])
    val_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize])
    train_dataset = LabeledImages(config['DATASET']['TUNING_LIST_PATH'], config['DATASET']['STAGE1_TEST_IMAGES_ROOT'],
                                  train_transform)
    val_dataset = LabeledImages(config['DATASET']['TUNING_LIST_PATH'], config['DATASET']['STAGE1_TEST_IMAGES_ROOT'],
                                val_transform)

    indices = np.arange(len(train_dataset), dtype=np.int64)
    np.random.shuffle(indices)
    train_size = round(config['DATASET']['TUNING_TRAIN_RATIO'] * len(indices))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    logger.info('Train size: {}. Val size: {}.'.format(len(train_indices), len(val_indices)))

    classifier = Classifier(n_classes)
    model_with_loss = ModelWithLoss(classifier).cuda()

    initial_weights_path = os.path.join(config['TRAINING']['ARTIFACTS_ROOT'], 'snapshots',
                                        'snapshot_epoch_{}.pth.tar'.format(config['TRAINING']['N_EPOCH']))
    logger.info('Finetuning from {}'.format(initial_weights_path))
    state = torch.load(initial_weights_path, map_location=lambda storage, loc: storage)
    model_with_loss.load_state_dict(state['model'])

    optimizer = torch.optim.SGD(model_with_loss.parameters(),
                                lr=config['TUNING']['INITIAL_LR'],
                                momentum=config['TUNING']['MOMENTUM'],
                                weight_decay=config['TUNING']['WEIGHT_DECAY'],
                                nesterov=True)

    loss_meter = AverageMeter()

    iteration = 0
    snapshots = [(int(path.split('_epoch_')[-1].split('.')[0]), path) for path in os.listdir(get_path('snapshots', config))
                 if path.startswith('snapshot') and path.endswith('.pth.tar')]
    if len(snapshots) > 0:
        snapshots.sort(key=lambda t: t[0])
        logger.info('Finetuning from {}'.format(snapshots[-1][1]))
        state = torch.load(os.path.join(get_path('snapshots', config), snapshots[-1][1]), map_location=lambda storage, loc: storage)
        model_with_loss.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch'] + 1
    else:
        start_epoch = 1
    writer = SummaryWriter(get_path(os.path.join('tensorboard', 'run_epoch_{}'.format(start_epoch)), config))

    for epoch in range(start_epoch, config['TUNING']['N_EPOCH'] + 1):
        adjust_learning_rate(optimizer, epoch, config['TUNING']['INITIAL_LR'], config['TUNING']['LR_DECAY'],
                             config['TUNING']['LR_STEP'])
        logger.info('Start epoch {} / {}.'.format(epoch, config['TUNING']['N_EPOCH']))
        val_score, val_thresh = validate(model_with_loss, val_dataset, val_indices, config['TUNING']['BATCH_SIZE'],
                                         config['VALIDATION']['TOP_K'])
        logger.info('Val score: {}. Val thresh: {}.'.format(val_score, val_thresh))
        writer.add_scalar('val_score', val_score, iteration)
        writer.add_scalar('val_thresh', val_thresh, iteration)

        model_with_loss.train()
        sampler = SubsetRandomSampler(train_indices)
        train_loader = DataLoader(train_dataset,
                                  batch_size=config['TUNING']['BATCH_SIZE'],
                                  sampler=sampler,
                                  num_workers=4,
                                  collate_fn=lambda X: X,
                                  drop_last=True)
        for samples in train_loader:
            input_tensor = torch.stack([sample['image'] for sample in samples]).cuda()
            labels = [sample['labels'] for sample in samples]

            optimizer.zero_grad()
            loss = model_with_loss(input_tensor, labels)
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), len(input_tensor))

            if iteration % config['TUNING']['LOG_FREQUENCY'] == 0:
                logger.info('Iteration {}. Loss {}.'.format(iteration, loss_meter.avg))
                writer.add_scalar('train_loss', loss_meter.avg, iteration)
                loss_meter.reset()

                for i, param_group in enumerate(optimizer.param_groups):
                    writer.add_scalar('lr/group_{}'.format(i), param_group['lr'], iteration)

            iteration += 1

        state = {'model': model_with_loss.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, get_path(os.path.join('snapshots', 'snapshot_epoch_{}.pth.tar'.format(epoch)), config))


if __name__ == '__main__':
    main()