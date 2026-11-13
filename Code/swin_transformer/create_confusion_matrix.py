import os
import json
import argparse
import sys

import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from my_dataset import MyDataSet
from model import swin_base_patch4_window7_224_in22k as create_model
import csv
from sklearn.metrics import classification_report, confusion_matrix
from prettytable import PrettyTable
from utilsnew import read_split_data, evaluate, save_roc_curve, save_pr_curve
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class ConfusionMatrix(object):
    """
    Incomplete figure rendering may depend on the Matplotlib version.
    The original implementation reports correct rendering with Matplotlib 3.2.1 on Windows and Ubuntu.
    The prettytable package is required for tabular output.
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        plt.yticks(range(self.num_classes), self.labels)
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


def plot_confusion_matrix(cm, class_names):
    """Render an annotated confusion matrix and save the figure."""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('./logs/confusion_matrix.png')
    plt.close()

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    _, _, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes)
    assert os.path.exists(args.weights), "cannot find {} file".format(args.weights)
    model.load_state_dict(
        torch.load(args.weights, map_location=device, weights_only=True)
    )
    model.to(device)

    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=args.num_classes, labels=labels)
    model.eval()
    loss, acc, true_labels, probs, pred_labels = evaluate(model, val_loader, device, epoch=0, return_outputs=True)

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation Loss: {loss:.4f}")

    with open('./logs/predictions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'true_label', 'predicted_label', 'probability'])
        for i in range(len(val_images_path)):
            writer.writerow([val_images_path[i], true_labels[i], pred_labels[i], probs[i]])

    roc_auc = save_roc_curve(true_labels, probs)

    avg_precision = save_pr_curve(true_labels, probs)

    report = classification_report(true_labels, pred_labels, target_names=labels)
    print("\nClassification Report:")
    print(report)

    with open('./logs/classification_report.txt', 'w') as f:
        f.write(report)

    cm = confusion_matrix(true_labels, pred_labels)

    plot_confusion_matrix(cm, class_names=labels)

    print(f"\nAUC: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)

    parser.add_argument('--data-path', type=str,
                        default=r"D:\Pycharm_object\Calculate_program\code\swin transformer\egg_photos")

    parser.add_argument('--weights', type=str, default='./weights/model-32.pth',
                        help='initial weights path')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')


    opt = parser.parse_args()

    main(opt)
