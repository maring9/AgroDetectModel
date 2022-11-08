from cProfile import label
from cgi import test
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

splits = ['/home/marin/Desktop/Dataset/PlantVillageResized/train',
          '/home/marin/Desktop/Dataset/PlantVillageResized/validation',
          '/home/marin/Desktop/Dataset/PlantVillageResized/testing']


def create_dict(directory_):
    data_dict = {}
    for split in directory_:
        class_names = os.listdir(split)
        for class_ in class_names:
            data_dict[class_] = []
    return data_dict


def get_samples_number(directory_):
    # total_samples = 0
    split_metrics = create_dict(directory_)
    for split in directory_:
        # split_name = split.split('/')[-1]
        print('Split name: split_name')
        split_samples = 0
        class_names = os.listdir(split)
        print('Class names: ', class_names)

        for class_ in class_names:
            class_dir = os.path.join(split, class_)

            class_samples = len(os.listdir(class_dir))

            print('Class samples: ', class_samples)

            split_samples += class_samples
            print('Split samples: ', split_samples)

            split_metrics[class_].append(class_samples) 

    print('Metrics: ', split_metrics)
    return split_metrics


def create_plot3(data):
    train_samples = []
    val_samples = []
    test_samples = []
    for key in data.keys():
        train_samples.append(data[key][0])
        val_samples.append(data[key][1])
        test_samples.append(data[key][2])
    keys = data.keys()
    df = pd.DataFrame({
        'Train Samples': train_samples,
        'Validation Samples': val_samples,
        'Test Samples': test_samples
    }, index=keys)

    ax = df.plot.barh(title='Number Of Samples Per Class',
                      figsize=(35, 25),
                      fontsize=20)

    plt.savefig('samples2.png')



def main():
    data = get_samples_number(splits)
    create_plot3(data)


if __name__ == "__main__":
    main()
