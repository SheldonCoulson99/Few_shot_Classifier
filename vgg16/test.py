import os
import sys
import json
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from module import vgg
from sklearn.metrics import classification_report

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {

        "test": transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.getcwd()  # get data root path
    image_path = os.path.join(data_root, "musicdataset")  # data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)



    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                        transform=data_transform["test"])
    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=16, shuffle=False,
                                              num_workers=nw)
    print("{} images for test.".format(test_num))


    # create model
    model = vgg(model_name="vgg16music", num_classes=2).to(device)
    # load model weights
    weights_path = "./vgg16musicNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.to(device)

    # test
    model.eval()
    all_labels=[]
    all_predict=[]
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        test_bar: tqdm = tqdm(test_loader, file=sys.stdout)
        for test_data in test_bar:
            test_images, test_labels = test_data

            outputs = model(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc = acc+torch.eq(predict_y, test_labels.to(device)).sum().item()

            test_labels=test_labels.tolist()

            all_labels=all_labels+test_labels
            predict_y=predict_y.tolist()

            all_predict= all_predict + predict_y
    test_accurate = acc / test_num
    print('test_accuracy: ',test_accurate)

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_predict)

    # Print Confusion Matrix
    print("confusion matrix:")
    print(cm)
    # calculate f1
    report= classification_report(all_labels, all_predict)
    print(report)









if __name__ == '__main__':
    main()