import os
import yaml
import pickle
import argparse

import numpy as np
import torch
torch.backends.cudnn.benchmark=True
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets import LabeledImages
from torchvision import transforms
from models.se_resnet import se_resnet101


CONFIG_PATH = '/source/config.yaml'


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
        input_shape = x.shape
        if len(input_shape) == 5:
            x = x.view(-1, input_shape[2], input_shape[3], input_shape[4])
            predictions = self.classifier(x)
            predictions = predictions.view(input_shape[0], input_shape[1], -1).mean(dim=1)
        else:
            predictions = self.classifier(x)

        scores, labels = predictions.sort(dim=1, descending=True)

        pred_scores = np.zeros(shape=(len(scores), top_k), dtype=np.float32)
        pred_labels = labels[:, :top_k].cpu().numpy()

        for i in range(top_k):
            i_scores = torch.cat((scores[:, i:i + 1], scores[:, top_k:]), dim=1)
            pred_scores[:, i] = F.softmax(i_scores, dim=1)[:, 0].cpu().numpy()

        return pred_scores, pred_labels


def main():
    parser = argparse.ArgumentParser(description='Generates submission (stage 1 or stage 2)')
    parser.add_argument('stage', type=int, choices=[1, 2])
    args = parser.parse_args()

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.load(f)

    dataset_root = config['DATASET']['STAGE{}_TEST_IMAGES_ROOT'.format(args.stage)]
    assert os.path.exists(dataset_root), dataset_root
    sample_submission_path = config['DATASET']['STAGE{}_SAMPLE_SUBMISSION_PATH'.format(args.stage)]
    assert os.path.exists(sample_submission_path), sample_submission_path

    output_root = config['SUBMISSION']['OUTPUT_ROOT']
    assert os.path.exists(output_root), output_root

    test_list_path = os.path.join(output_root, 'test_stage{}.txt'.format(args.stage))

    with open(sample_submission_path, 'r') as f_in, open(test_list_path, 'w') as f_out:
        f_in.readline()
        for line in f_in:
            img_id, _ = line.split(',')
            img_name = img_id + '.jpg'
            if os.path.exists(os.path.join(dataset_root, img_name)):
                f_out.write('{}\n'.format(img_name))
            else:
                print('Warning: file {} does not exist'.format(os.path.join(dataset_root, img_name)))

    with open(config['DATASET']['NAME_TO_LABEL_PATH'], 'rb') as f:
        name_to_label = pickle.load(f)
    label_to_name = {label: name for name, label in name_to_label.items()}
    n_classes = len(name_to_label)

    classifier = Classifier(n_classes)
    model = ModelWithLoss(classifier).cuda().eval()

    snapshot_path = os.path.join(config['TUNING']['ARTIFACTS_ROOT'], 'snapshots',
                                 'snapshot_epoch_{}.pth.tar'.format(config['TUNING']['N_EPOCH']))
    state = torch.load(snapshot_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state['model'])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.Compose([transforms.ToTensor(), normalize])
    tta_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.TenCrop(224),
                                        transforms.Lambda(
                                            lambda crops: torch.stack([to_tensor(crop) for crop in crops]))])
    dataset = LabeledImages(test_list_path, dataset_root, tta_transform)
    loader = DataLoader(dataset, config['SUBMISSION']['BATCH_SIZE'], num_workers=4, collate_fn=lambda X: X)

    pred_scores_all = []
    pred_labels_all = []

    for samples in loader:
        input_tensor = torch.stack([sample['image'] for sample in samples]).cuda()

        with torch.no_grad():
            pred_scores, pred_labels = model.predict(input_tensor, config['SUBMISSION']['TOP_K'])
            pred_scores_all.extend(pred_scores)
            pred_labels_all.extend(pred_labels)

    image_id_to_names = {}

    threshold = config['SUBMISSION']['THRESHOLD']
    min_preds = config['SUBMISSION']['MIN_PREDS']
    max_preds = config['SUBMISSION']['MAX_PREDS']

    for pred_scores, pred_labels, img_name in zip(pred_scores_all, pred_labels_all, dataset._content):
        best_indices = np.argsort(pred_scores)
        best_labels = pred_labels[best_indices]
        best_scores = pred_scores[best_indices]
        pred_labels = best_labels[best_scores > threshold]
        if len(pred_labels) > max_preds:
            pred_labels = pred_labels[-max_preds:]
        if len(pred_labels) >= min_preds:
            pred_names = [label_to_name[label] for label in pred_labels.tolist()]
        else:
            pred_names = [label_to_name[label] for label in best_labels[-min_preds:].tolist()]
        image_id = img_name.split('.')[0]
        image_id_to_names[image_id] = pred_names

    submission_path = os.path.join(output_root, 'submission_stage{}.csv'.format(args.stage))

    with open(sample_submission_path, 'r') as f_in, open(submission_path, 'w') as f_out:
        f_out.write(f_in.readline())
        for line in f_in:
            img_id, _ = line.split(',')
            if img_id in image_id_to_names:
                names = image_id_to_names[img_id]
                f_out.write('{},'.format(img_id))
                f_out.write('{}\n'.format(' '.join(names)))
            else:
                f_out.write('{},\n'.format(img_id))


if __name__ == '__main__':
    main()
