import argparse
import os
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from PIL import Image

from common import load_model, load_predict_image_names, load_single_image, load_image_labels


########################################################################################################################

def generate_image_list(directory, predict_image_list):

    # Define the image file extensions to look for
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # Get the path for the output file in the current directory
    output_path = os.path.join(os.getcwd(), predict_image_list)
    
    with open(output_path, 'w') as file:
        for filename in os.listdir(directory):
            if filename.lower().endswith(valid_extensions):
                file.write(filename + '\n')


def parse_args():
    """
    Helper function to parse command line arguments
    :return: args object
    """
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--predict_data_image_dir', required=True, help='Path to image data directory')
    parser.add_argument('-l', '--predict_image_list', required=True,
                        help='Path to text file listing file names within predict_data_image_dir')
    parser.add_argument('-t', '--target_column_name', required=True,
                        help='Name of column to write prediction when generating output CSV')
    parser.add_argument('-m', '--trained_model_dir', required=True,
                        help='Path to directory containing the model to use to generate predictions')
    parser.add_argument('-o', '--predicts_output_csv', required=True, help='Path to CSV where to write the predictions')
    args = parser.parse_args()
    return args


def predict(model: Any, image: Image) -> str:
    """
    Generate a prediction for a single image using the model, returning a label of 'Yes' or 'No'

    IMPORTANT: The return value should ONLY be either a 'Yes' or 'No' (Case sensitive)

    :param model: the model to use.
    :param image: the image file to predict.
    :return: the label ('Yes' or 'No)
    """
    # predicted_label = 'No'
    # # TODO: Implement your logic to generate prediction for an image here.
    # raise RuntimeError("predict() is not implemented.")
    # return predicted_label

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image1 = Image.open(image)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image1).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)

    # model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        label = 'Yes' if predicted.item() == 1 else 'No'
    return label


def main(predict_data_image_dir: str,
         predict_image_list: str,
         target_column_name: str,
         trained_model_dir: str,
         predicts_output_csv: str):
    """
    The main body of the predict.py responsible for:
     1. load model
     2. load predict image list
     3. for each entry,
           load image
           predict using model
     4. write results to CSV

    :param predict_data_image_dir: The directory containing the prediction images.
    :param predict_image_list: Name of text file within predict_data_image_dir that has the names of image files.
    :param target_column_name: The name of the prediction column that we will generate.
    :param trained_model_dir: Path to the directory containing the model to use for predictions.
    :param predicts_output_csv: Path to the CSV file that will contain all predictions.
    """
    # Check if the image list exists, if not, generate it
    image_list_path = os.path.join(predict_data_image_dir, predict_image_list)
    if not os.path.exists(image_list_path):
        generate_image_list(predict_data_image_dir, predict_image_list)


    # load pre-trained models or resources at this stage.
    model = load_model(trained_model_dir, target_column_name)
    # print(model)

    # Load in the image list
    # image_list_file = os.path.join(predict_data_image_dir, predict_image_list)
    image_filenames = load_predict_image_names(image_list_path)

    # Iterate through the image list to generate predictions
    predictions = []
    for filename in image_filenames:
        try:
            image_path = os.path.join(predict_data_image_dir, filename)
            image = load_single_image(image_path)
            print(image)
            label = predict(model, image)
            predictions.append(label)
        except Exception as ex:
            # print(ex)
            print(f"Error generating prediction for {filename} due to {ex}")
            predictions.append("Error")

    df_predictions = pd.DataFrame({'Filenames': image_filenames, target_column_name: predictions})

    # Finally, write out the predictions to CSV
    df_predictions.to_csv(predicts_output_csv, index=False)


if __name__ == '__main__':
    """
    Example usage:

    python predict.py -d "path/to/Data - Is Epic Intro Full" -l "Is Epic Files.txt" -t "Is Epic" -m "path/to/Is Epic/model" -o "path/to/Is Epic Full Predictions.csv"

    """
    args = parse_args()
    predict_data_image_dir = args.predict_data_image_dir
    predict_image_list = args.predict_image_list
    target_column_name = args.target_column_name
    trained_model_dir = args.trained_model_dir
    predicts_output_csv = args.predicts_output_csv

    main(predict_data_image_dir, predict_image_list, target_column_name, trained_model_dir, predicts_output_csv)

########################################################################################################################
