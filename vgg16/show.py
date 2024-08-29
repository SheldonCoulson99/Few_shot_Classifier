import os
import sys
import torch
from torchvision import transforms, datasets
from tqdm import tqdm
from module import vgg
from sklearn.metrics import confusion_matrix, classification_report
import tkinter as tk
from PIL import Image, ImageTk

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    root = tk.Tk()
    root.title("Test Dataset Predictions")

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_root = os.getcwd()  # 获取数据根路径
    image_path = os.path.join(data_root, "musicdataset", "test")  # 数据集路径
    assert os.path.exists(image_path), "{}路径不存在。".format(image_path)

    test_dataset = datasets.ImageFolder(root=image_path, transform=data_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 创建模型并加载权重
    model = vgg(model_name="vgg16music", num_classes=2).to(device)
    weights_path = "./vgg16musicNet.pth"
    assert os.path.exists(weights_path), "文件'{}'不存在。".format(weights_path)
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    model.eval()

    row = 0
    label_map = {0: "No", 1: "Yes"}  # 模型类别映射

    # 创建滚动条框架
    canvas = tk.Canvas(root)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    scrollable_frame.bind("<Configure>", on_frame_configure)

    for test_data in tqdm(test_loader):
        test_images, test_labels = test_data
        test_images = test_images.to(device)

        with torch.no_grad():
            outputs = model(test_images)
            _, predicted = torch.max(outputs, 1)
            predicted_label = label_map[predicted.item()]
            true_label = label_map[test_labels.item()]

        # 显示图片
        img = test_images.squeeze(0).cpu().numpy().transpose((1, 2, 0))
        img = (img * 0.5 + 0.5) * 255  # 反归一化
        img = Image.fromarray(img.astype('uint8'))
        img = ImageTk.PhotoImage(img)
        panel = tk.Label(scrollable_frame, image=img)
        panel.image = img
        panel.grid(row=row, column=0, padx=10, pady=10)

        # 显示真实标签
        true_label_label = tk.Label(scrollable_frame, text="True Label: " + true_label)
        true_label_label.grid(row=row, column=1, padx=10, pady=10)

        # 显示预测标签
        predicted_label_label = tk.Label(scrollable_frame, text="Predicted Label: " + predicted_label)
        predicted_label_label.grid(row=row, column=2, padx=10, pady=10)

        row += 1

    root.mainloop()

if __name__ == '__main__':
    main()



