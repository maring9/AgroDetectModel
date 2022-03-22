import os
from PIL import Image


class DatasetResizer():
    """
        Class used to resize image dataset directories
    """

    def __init__(self, source_dir, dest_dir):
        """Constructor

        Args:
            source_dir (str): Path of the dataset to be resized
            dest_dir (str): Path where to output the resized dataset
        """

        self.source_dir = source_dir
        self.dest_dir = dest_dir

    def create_split_dirs(self):
        """
            Method to create train / validation / testing directories
        """

        # Create split paths
        self.train_dir = os.path.join(self.source_dir, 'train')
        self.validation_dir = os.path.join(self.source_dir, 'validation')
        self.test_dir = os.path.join(self.source_dir, 'testing')

        # Create directories
        if not os.path.isdir(self.train_dir):
            os.mkdir(self.train_dir)

        if not os.path.isdir(self.validation_dir):
            os.mkdir(self.validation_dir)

        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        print('Created Split Directories')

    def create_class_dirs(self, source_split_dir, split_name, dest_dir):
        """
            Method to create classes subdirectories for each split

        Args:
            source_split_dir (str): Path of the split (train / val / test) directory
            split_name (str): Split name
            dest_dir (str): Destination path
        """

        # List containing classes
        classes = os.listdir(source_split_dir)

        # Loop over the classes
        for class_ in classes:
            # Create destination split directory for the class
            class_dir = os.path.join(dest_dir, split_name, class_)

            # Create class directory
            if not os.path.isdir(class_dir):
                os.mkdir(class_dir)

        print('Created Class Directories')

    def resize(self, TARGET_SIZE=(150, 150)):
        """
            Main method used to resize the dataset

        Args:
            TARGET_SIZE (tuple, optional):  Target image resolution.
                                            Defaults to (150, 150).
        """

        # Create destination directory
        if not os.path.isdir(self.dest_dir):
            os.mkdir(self.dest_dir)

        # Create train / validation / test directories
        self.create_split_dirs(self.dest_dir)

        # Create source splits paths
        source_split_dir = [os.path.join(source_dir, dir_) for dir_ in
                            os.listdir(self.source_dir)]

        # Main loop over the splits
        for split in source_split_dir:
            # Get split name
            split_dirname = split.split('/')[-1]

            # Create class subdirectories for the split
            self.create_class_dirs(split, split_dirname, self.dest_dir)

            # Create class paths
            class_paths = [os.path.join(split, class_) for class_ in
                           os.listdir(split)]

            # Sub loop over the classes
            for class_ in class_paths:
                # Get class name
                class_dirname = class_.split('/')[-1]

                # Create image paths for each class subdirectory
                image_paths = [os.path.join(class_, image) for image in
                               os.listdir(class_)]

                # Third loop over all the images
                for image_path in image_paths:
                    # Get image name
                    image_name = image_path.split('/')[-1]

                    # Open image
                    image = Image.open(image_path)

                    # Resize image to target resolution
                    image = image.resize(TARGET_SIZE)

                    # Create path where to save the image
                    save_path = os.path.join(self.dest_dir,
                                             split_dirname,
                                             class_dirname)

                    # Save image
                    _ = image.save(f'{save_path}/{image_name}')


if __name__ == '__main__':
    source_dir = '/home/marin/Desktop/Dataset/PlantVillage'
    destination_dir = '/home/marin/Desktop/Dataset/PlantVillageResized'

    dataset_processor = DatasetResizer(source_dir, destination_dir)

    dataset_processor.resize()
