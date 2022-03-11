import os
from PIL import Image


def create_split_dirs(source_dir):
    train_dir = os.path.join(source_dir, 'train')
    validation_dir = os.path.join(source_dir, 'validation')
    test_dir = os.path.join(source_dir, 'testing')

    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)

    if not os.path.isdir(validation_dir):
        os.mkdir(validation_dir)

    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)

    print('Created Split Directories')


def create_class_dirs(split_dir, split_name, dest_dir):
    classes = os.listdir(split_dir)
    for class_ in classes:
        class_dir = os.path.join(dest_dir, split_name, class_)
        if not os.path.isdir(class_dir):
            os.mkdir(class_dir)

    print('Created Class Directories')


def resize(source_dir, dest_dir):
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)

    create_split_dirs(dest_dir)

    split_dir = [os.path.join(source_dir, dir_) for dir_ in os.listdir(source_dir)]
    print(split_dir)
    for split in split_dir:
        split_dirname = split.split('/')[-1]
        create_class_dirs(split, split_dirname, dest_dir)

        class_paths = [os.path.join(split, class_) for class_ in os.listdir(split)]

        for class_ in class_paths:
            class_dirname = class_.split('/')[-1]
            image_paths = [os.path.join(class_, image) for image in os.listdir(class_)]
            for image_path in image_paths:
                image_name = image_path.split('/')[-1]
                image = Image.open(image_path)
                image = image.resize((150, 150))
                print('resized')
                save_dir = os.path.join(dest_dir, split_dirname, class_dirname)
                save = image.save(f'{save_dir}/{image_name}')


if __name__ == '__main__':
    source_dir = '/home/marin/Desktop/Dataset/PlantVillage'
    destination_dir = '/home/marin/Desktop/Dataset/PlantVillageResized'

    resize(source_dir, destination_dir)
