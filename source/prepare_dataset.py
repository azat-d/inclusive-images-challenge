import os
import pickle
import yaml
import random
from collections import defaultdict
import cv2


CONFIG_PATH = '/source/config.yaml'
LOG_FREQUENCY = 10000


def main():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.load(f)

    name_to_label = {}
    img_path_to_labels = defaultdict(list)

    with open(config['DATASET']['HUMAN_LABELS_PATH'], 'r') as f:
        f.readline()
        for line in f:
            parts = line.strip().split(',')
            img_path = os.path.join('train', parts[0] + '.jpg')
            name = parts[2]
            if name not in name_to_label:
                name_to_label[name] = len(name_to_label)
            label = name_to_label[name]
            img_path_to_labels[img_path].append(label)

    print('Total number of images: {}. Total number of labels: {}.'.format(len(img_path_to_labels), len(name_to_label)))

    content = sorted(img_path_to_labels)
    print('Resizing images...')
    for i, rel_path in enumerate(content):
        src_path = os.path.join(config['DATASET']['ORIGINAL_IMAGES_ROOT'], rel_path)
        dst_path = os.path.join(config['DATASET']['RESIZED_IMAGES_ROOT'], rel_path)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        img = cv2.imread(src_path)
        height, width, channels = img.shape
        if height < width:
            dst_height = config['DATASET']['SHORTEST_SIZE']
            dst_width = round((config['DATASET']['SHORTEST_SIZE'] / height) * width)
        else:
            dst_width = config['DATASET']['SHORTEST_SIZE']
            dst_height = round((config['DATASET']['SHORTEST_SIZE'] / width) * height)
        img = cv2.resize(img, (dst_width, dst_height))

        cv2.imwrite(dst_path, img)
        if i % LOG_FREQUENCY == 0:
            print('{} / {} processed'.format(i + 1, len(content)))

    random.seed(0xDEADFACE)
    random.shuffle(content)

    n_val = round(config['DATASET']['VAL_RATIO'] * len(content))
    n_train = len(content) - n_val

    train_content = content[:n_train]
    val_content = content[n_train:]

    print('Train size: {}. Val size: {}.'.format(len(train_content), len(val_content)))

    with open(config['DATASET']['TRAIN_LIST_PATH'], 'w') as f:
        for img_path in train_content:
            labels = img_path_to_labels[img_path]
            line = ' '.join([img_path] + [str(label) for label in labels])
            f.write('{}\n'.format(line))

    with open(config['DATASET']['VAL_LIST_PATH'], 'w') as f:
        for img_path in val_content:
            labels = img_path_to_labels[img_path]
            line = ' '.join([img_path] + [str(label) for label in labels])
            f.write('{}\n'.format(line))

    with open(config['DATASET']['NAME_TO_LABEL_PATH'], 'wb') as f:
        pickle.dump(name_to_label, f)


if __name__ == '__main__':
    main()