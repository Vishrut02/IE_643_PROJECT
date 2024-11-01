import streamlit as st
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, Dropout, UpSampling2D, Concatenate
from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from pdf2image import convert_from_bytes
import cv2
import ast
import pytesseract
from pytesseract import image_to_string
import re



# Set paths for your models
FASTER_RCNN_MODEL_PATH = 'Faster_RCNN/best_model (1).pth'  # Update with actual path
VGG19_MODEL_PATH = 'VGG_19/mymodel_277.keras'  # Update with actual path
TEST_IMAGES_DIR = os.path.join(os.getcwd(), "test_images")

class table_decoder(tf.keras.layers.Layer):
    def __init__(self, name="table_mask"):
        super().__init__(name=name)
        self.conv1 = Conv2D(filters =512,kernel_size = (1, 1), activation = 'relu')
        self.upsample1 = UpSampling2D(size=(2, 2))
        self.upsample2 = UpSampling2D(size=(2, 2))
        self.upsample3 = UpSampling2D(size=(2,2))
        self.upsample4 = UpSampling2D(size=(2,2))
        self.convtraspose = tf.keras.layers.Conv2DTranspose(3, 3, strides=2,padding='same')

        
    def call(self, X):
      input , pool_3, pool_4 = X[0] , X[1], X[2]


      result = self.conv1(input)
      result = self.upsample1(result)
      result = Concatenate()([result,pool_4])
      result = self.upsample2(result)
      result = Concatenate()([result,pool_3])
      result = self.upsample3(result)
      result = self.upsample4(result)
      result = self.convtraspose(result)

      return result

# Register custom layers with Keras
@register_keras_serializable(package="Custom", name="table_decoder")
class table_decoder(tf.keras.layers.Layer):
    def __init__(self, name="table_mask", trainable=True, dtype=None, **kwargs):
        super().__init__(name=name, trainable=trainable, dtype=dtype, **kwargs)
        self.conv1 = Conv2D(filters=512, kernel_size=(1, 1), activation='relu')
        self.upsample1 = UpSampling2D(size=(2, 2))
        self.upsample2 = UpSampling2D(size=(2, 2))
        self.upsample3 = UpSampling2D(size=(2, 2))
        self.upsample4 = UpSampling2D(size=(2, 2))
        self.convtraspose = tf.keras.layers.Conv2DTranspose(3, 3, strides=2, padding='same')

    def call(self, X):
        input, pool_3, pool_4 = X[0], X[1], X[2]
        result = self.conv1(input)
        result = self.upsample1(result)
        result = Concatenate()([result, pool_4])
        result = self.upsample2(result)
        result = Concatenate()([result, pool_3])
        result = self.upsample3(result)
        result = self.upsample4(result)
        result = self.convtraspose(result)
        return result

@register_keras_serializable(package="Custom", name="column_decoder")
class column_decoder(tf.keras.layers.Layer):
    def __init__(self, name="column_mask", trainable=True, dtype=None, **kwargs):
        super().__init__(name=name, trainable=trainable, dtype=dtype, **kwargs)
        self.conv1 = Conv2D(filters=512, kernel_size=(1, 1), activation='relu')
        self.drop = Dropout(0.8)
        self.conv2 = Conv2D(filters=512, kernel_size=(1, 1), activation='relu')
        self.upsample1 = UpSampling2D(size=(2, 2))
        self.upsample2 = UpSampling2D(size=(2, 2))
        self.upsample3 = UpSampling2D(size=(2, 2))
        self.upsample4 = UpSampling2D(size=(2, 2))
        self.convtraspose = tf.keras.layers.Conv2DTranspose(3, 3, strides=2, padding='same', name='column_mask')

    def call(self, X):
        input, pool_3, pool_4 = X[0], X[1], X[2]
        result = self.conv1(input)
        result = self.drop(result)
        result = self.conv2(result)
        result = self.upsample1(result)
        result = Concatenate()([result, pool_4])
        result = self.upsample2(result)
        result = Concatenate()([result, pool_3])
        result = self.upsample3(result)
        result = self.upsample4(result)
        result = self.convtraspose(result)
        return result



# Load Faster R-CNN model
@st.cache_resource
def load_faster_rcnn_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_classes=4)
    model.load_state_dict(torch.load(FASTER_RCNN_MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# Load VGG-19 model
@st.cache_resource
def load_vgg19_model():
    return load_model(VGG19_MODEL_PATH, custom_objects={'table_decoder': table_decoder, 'column_decoder': column_decoder})

# Helper function to process masks
def process_masks(table_mask, column_mask):
    table_mask = tf.argmax(table_mask, axis=-1)[..., tf.newaxis]
    column_mask = tf.argmax(column_mask, axis=-1)[..., tf.newaxis]
    return table_mask[0].numpy(), column_mask[0].numpy()



def detect_column_boundaries(column_mask_img,table_mask_img,threshold_ratio=0.08):
  
    # print(column_mask_array)
    white_pixel_ratio = np.sum(table_mask_img == 255) / table_mask_img.size
    if white_pixel_ratio < 0.5:
        print("Not enough white pixels in table mask; skipping column detection.")
        return [], 0  # Return no columns if it's not classified as a table
    _, binary_column_mask = cv2.threshold(column_mask_img, 127, 255, cv2.THRESH_BINARY)
    # print(binary_column_mask)
    vertical_projection = np.sum(binary_column_mask, axis=0)
    threshold = threshold_ratio * np.max(vertical_projection)
    # print(vertical_projection)
    
    column_starts = (vertical_projection > threshold).astype(int)
    transitions = np.diff(column_starts)
    column_boundaries = []
    start = None

    for i, val in enumerate(transitions):
        if val == 1:  # Start of a column
            if start is None:
                start = i
        elif val == -1 and start is not None:  # End of a column
            end = i
            column_boundaries.append((start, end))
            start = None

    return column_boundaries , len(column_boundaries)

def count_columns(column_mask_img):
    # Ensure the column mask is in uint8 format
   
    column_mask_array = (column_mask_img).astype(np.uint8)

    # Apply binary threshold to create a binary mask
    _, binary_column_mask = cv2.threshold(column_mask_array, 127, 255, cv2.THRESH_BINARY)

    # Sum the binary mask vertically to get the projection
    vertical_projection = np.sum(binary_column_mask, axis=0)

    # Set a threshold to detect column areas based on the projection
    threshold = np.max(vertical_projection) * 0.5  # Dynamic threshold based on maximum projection

    # Detect column start and end positions
    column_starts = (vertical_projection > threshold).astype(int)
    transitions = np.diff(column_starts)

    # Identify column boundaries
    column_boundaries = []
    start = None
    
    for i, val in enumerate(transitions):
        if val == 1:  # Column start
            if start is None:
                start = i
        elif val == -1 and start is not None:  # Column end
            end = i
            column_boundaries.append((start, end))
            start = None

    # Calculate the number of columns based on detected boundaries
    num_columns = len(column_boundaries)

    # Visualization of the column mask and detected boundaries
    plt.figure(figsize=(10, 5))
    plt.imshow(column_mask_array, cmap='gray')
    for start, end in column_boundaries:
        plt.axvline(x=start, color='red', linestyle='--')
        plt.axvline(x=end, color='blue', linestyle='--')
    plt.title(f"Column Mask with Detected Boundaries - {num_columns} Columns")
    plt.show()

    return num_columns


# Find the closest text box
def find_closest_text_box(target_box, text_boxes):
    target_center = np.array([(target_box[0].cpu().item() + target_box[2].cpu().item()) / 2,
                              (target_box[1].cpu().item() + target_box[3].cpu().item()) / 2])
    closest_box = min(text_boxes, key=lambda text_box: np.linalg.norm(target_center - np.array(
                    [(text_box[0].cpu().item() + text_box[2].cpu().item()) / 2,
                     (text_box[1].cpu().item() + text_box[3].cpu().item()) / 2])), default=None)
    return closest_box

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pytesseract import image_to_string

# Updated function to run Faster R-CNN, save results to CSV, and visualize
def process_with_faster_rcnn(images, faster_rcnn_model, confidence_threshold=0.75):
    results = []

    for image_name in images:
        image_path = os.path.join(TEST_IMAGES_DIR, image_name)
        image = Image.open(image_path).convert("RGB")
        image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(torch.device('cpu'))

        # Run inference with Faster R-CNN
        with torch.no_grad():
            outputs = faster_rcnn_model(image_tensor)[0]

        # Filter detections based on confidence threshold
        filtered_boxes = [
            (box, label, score) 
            for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores'])
            if score.item() >= confidence_threshold
        ]

        # Separate boxes by label type
        text_boxes = [box for box, label, _ in filtered_boxes if label.item() == 1]
        table_boxes = [box for box, label, _ in filtered_boxes if label.item() == 2]
        figure_boxes = [box for box, label, _ in filtered_boxes if label.item() == 3]

        # Prepare figure for visualization
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        label_colors = { 'Table/Figure': 'g', 'Text': 'b'}
        label_names = {'Table': 'Table', 'Figure': 'Figure', 'Text': 'Closest Text','Table/Figure':'Table/Figure'}

        # Process each detected table
        for table_box in table_boxes:
            closest_text_box = find_closest_text_box(table_box, text_boxes)

            # Extract closest text using OCR
            closest_text = ""
            if closest_text_box is not None:
                xmin, ymin, xmax, ymax = map(int, closest_text_box.cpu().numpy())
                text_region = image.crop((xmin, ymin, xmax, ymax))
                closest_text = image_to_string(text_region, config='--psm 6').strip()

                # Draw closest text box
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                         linewidth=1, edgecolor=label_colors['Text'], facecolor='none')
                ax.add_patch(rect)
                ax.text(xmin, ymin - 10, f"{label_names['Text']}: {closest_text[:30]}...", color=label_colors['Text'], fontsize=9)

            # Draw table box
            xmin, ymin, xmax, ymax = map(int, table_box.cpu().numpy())
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor=label_colors['Table/Figure'], facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 10, f"{label_names['Table/Figure']}", color=label_colors['Table/Figure'], fontsize=12)

            # Append table result to CSV results
            results.append({
                'image': image_name,
                'width': image.width,
                'height': image.height,
                'type': 'Table',
                'box': table_box.cpu().numpy().tolist(),
                'closest_text_box': closest_text_box.cpu().numpy().tolist() if closest_text_box is not None else None,
                'closest_text': closest_text
            })

        # Process each detected figure
        for figure_box in figure_boxes:
            closest_text_box = find_closest_text_box(figure_box, text_boxes)

            # Extract closest text using OCR
            closest_text = ""
            if closest_text_box is not None:
                xmin, ymin, xmax, ymax = map(int, closest_text_box.cpu().numpy())
                text_region = image.crop((xmin, ymin, xmax, ymax))
                closest_text = image_to_string(text_region, config='--psm 6').strip()

                # Draw closest text box
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                         linewidth=1, edgecolor=label_colors['Text'], facecolor='none')
                ax.add_patch(rect)
                ax.text(xmin, ymin - 10, f"{label_names['Text']}: {closest_text[:30]}...", color=label_colors['Text'], fontsize=9)

            # Draw figure box
            xmin, ymin, xmax, ymax = map(int, figure_box.cpu().numpy())
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor=label_colors['Table/Figure'], facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 10, f"{label_names['Table/Figure']}", color=label_colors['Table/Figure'], fontsize=12)

            # Append figure result to CSV results
            results.append({
                'image': image_name,
                'width': image.width,
                'height': image.height,
                'type': 'Figure',
                'box': figure_box.cpu().numpy().tolist(),
                'closest_text_box': closest_text_box.cpu().numpy().tolist() if closest_text_box is not None else None,
                'closest_text': closest_text
            })

        # Finalize and display visualization
        plt.axis('off')
        st.pyplot(fig)

    # Save results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv('bounding_boxes.csv', index=False)
    print("\nResults saved to 'bounding_boxes.csv'")



def process_table_ocr(column_boundaries, table_region):
    # Run OCR on the entire table region
    ocr_text = pytesseract.image_to_string(table_region, config='--psm 6')
    print("Full OCR Text:\n", ocr_text)  # Display full OCR text

    # Split OCR output into lines for processing
    ocr_lines = ocr_text.splitlines()
    print("\nSplit OCR Lines:")
    for line_num, line in enumerate(ocr_lines, 1):
        print(f"Line {line_num}: {line}")

    # Initialize structured OCR output
    split_ocr_data = []

    # Process each OCR-detected line
    for line_num, line in enumerate(ocr_lines, 1):
        words = line.split()  # Split the line into individual words
        print(f"\nProcessing Line {line_num}: {words}")

        line_segments = [''] * len(column_boundaries)  # Initialize each segment for columns

        # Analyze each word within the current line
        for col_index, (xmin, xmax) in enumerate(column_boundaries):
            # Crop the specific column segment in the table region for each boundary
            column_segment = table_region.crop((xmin, 0, xmax, table_region.height))
            column_text = pytesseract.image_to_string(column_segment, config='--psm 6')
            print(f"Column {col_index + 1} Text:\n{column_text}")

            # Append each column's OCR text after processing line by line
            line_segments[col_index] = column_text.strip()  # Strip to clean up extra spaces

        split_ocr_data.append(line_segments)

    # Determine max column length for uniformity
    max_columns = max(len(row) for row in split_ocr_data)
    # Normalize rows by padding with empty strings to match max column count
    split_ocr_data = [row + [''] * (max_columns - len(row)) for row in split_ocr_data]

    # Print final structured OCR output for debugging
    print("\nStructured OCR Output (Final Table):")
    for row in split_ocr_data:
        print(row)

    # Convert to DataFrame and display in Streamlit
    df = pd.DataFrame(split_ocr_data)
    st.write("### Structured OCR Output (Final Table)")
    st.write("Below is the structured table data based on OCR column segmentation:")
    st.dataframe(df)

    # Prepare CSV for download
    csv_data = df.to_csv(index=False, header=False).encode('utf-8')
    st.download_button(
        label="Download Structured Table as CSV",
        data=csv_data,
        file_name="structured_ocr_table.csv",
        mime="text/csv"
    )

    # Visualize the OCR-processed table region with boundaries in Streamlit
    st.write("### Column Segments for OCR")
    for col_index, (xmin, xmax) in enumerate(column_boundaries):
        # Crop each column segment based on boundaries
        column_segment = table_region.crop((xmin, 0, xmax, table_region.height))
        
        # Display cropped column segment image in Streamlit
        st.image(column_segment, caption=f"Column {col_index + 1} Boundary: ({xmin}, {xmax})", use_column_width=True)

# Sample usage


def save_and_display_figure_caption(image, caption_coordinates, page_number, output_dir):
    """Handles figure caption extraction, saving, and display in Streamlit."""
    xmin, ymin, xmax, ymax = caption_coordinates

    # Crop the figure caption area and perform OCR
    figure_caption_region = image.crop((xmin, ymin, xmax, ymax))
    figure_caption_text = pytesseract.image_to_string(figure_caption_region, config='--psm 6').strip()

    # Save the figure caption text to a file
    figure_caption_dir = os.path.join(output_dir, 'figure_captions')
    os.makedirs(figure_caption_dir, exist_ok=True)
    figure_caption_path = os.path.join(figure_caption_dir, f"page_{page_number}_figure_caption.txt")
    with open(figure_caption_path, 'w') as caption_file:
        caption_file.write(figure_caption_text)

    # Display the figure caption text in Streamlit
    st.write(f"### Figure Caption for Page {page_number}")
    st.text(figure_caption_text)

    return figure_caption_text



def save_and_display_table_caption(image, caption_coordinates, page_number, output_dir):
    """Handles table caption extraction, saving, and display in Streamlit."""
    xmin, ymin, xmax, ymax = caption_coordinates

    # Crop the table caption area and perform OCR
    table_caption_region = image.crop((xmin, ymin, xmax, ymax))
    table_caption_text = pytesseract.image_to_string(table_caption_region, config='--psm 6').strip()

    # Save the table caption text to a file
    table_caption_dir = os.path.join(output_dir, 'table_captions')
    os.makedirs(table_caption_dir, exist_ok=True)
    table_caption_path = os.path.join(table_caption_dir, f"page_{page_number}_table_caption.txt")
    with open(table_caption_path, 'w') as caption_file:
        caption_file.write(table_caption_text)

    # Display the table caption text in Streamlit
    st.write(f"### Table Caption for Page {page_number}")
    st.text(table_caption_text)

    return table_caption_text



def process_with_vgg(csv_path, vgg19_model):
    df = pd.read_csv(csv_path)
    output_directory = "output_directory"  # Set this to your desired output folder
    table_dir = os.path.join(output_directory, "tables")
    figure_dir = os.path.join(output_directory, "figures")

    # Create directories if they don't already exist
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    page_number = 1

    for _, row in df.iterrows():
        print("K1")
        image_name = row['image']
        orig_width, orig_height = row['width'], row['height']
        box_coordinates = np.array(ast.literal_eval(row['box']))
        caption_text_coordinates = np.array(ast.literal_eval(row['closest_text_box']))
        print("Hello")
        
        # Load image and crop based on scaled box coordinates
        image_path = os.path.join(TEST_IMAGES_DIR, image_name)
        image = Image.open(image_path).convert('RGB')

        print("Hire")
        
        # Coordinates for cropping
        xmin, ymin, xmax, ymax = map(int, box_coordinates)
        table_region = image.crop((xmin, ymin, xmax, ymax))
        # orignal_text_ocr  =  pytesseract.image_to_string(table_region, config='--psm 6')

        scale_x = 1024 / orig_width
        scale_y = 1024 / orig_height
        scaled_box = box_coordinates * np.array([scale_x, scale_y, scale_x, scale_y])

        print("Jik")
        
        # Resize image for VGG-19 model
        resized_image = tf.image.resize(image, [1024, 1024])
        resized_image_array = tf.keras.preprocessing.image.img_to_array(resized_image) / 255.0
        resized_image_array = np.expand_dims(resized_image_array, axis=0)

        # Run the VGG-19 model and process masks
        mask1, mask2 = vgg19_model.predict(resized_image_array)
        table_mask, column_mask = process_masks(mask1, mask2)
        # tf.experimental.numpy.experimental_enable_numpy_behavior()
        column_mask_2d = np.squeeze(column_mask)
        table_mask_2d = np.squeeze(table_mask)

        print("Kyun")

        
        # Resize column mask back to original dimensions
        original_column_mask = Image.fromarray((column_mask_2d * 255).astype(np.uint8)).resize(
            (orig_width, orig_height), Image.NEAREST)
        original_column_mask = np.array(original_column_mask)
        print("Kiase")
        # st.image(original_column_mask, caption="Original Column Mask", use_column_width=True)
        original_table_mask = Image.fromarray((table_mask_2d * 255).astype(np.uint8)).resize(
            (orig_width, orig_height), Image.NEAREST)
        original_table_mask = np.array(original_table_mask)

        print("Hiel")


        # Check if it's a table based on column count
        cropped_column_mask = original_column_mask[ymin:ymax, xmin:xmax]
        cropped_table_mask = original_table_mask[ymin:ymax, xmin:xmax]
        # Display the cropped column mask
        # st.image(cropped_column_mask, caption="Cropped Column Mask for Column Detection", use_column_width=True)
        # print(cropped_column_mask)
        print("Kiiee")

        # Optionally, display the cropped mask's dimensions for additional context
        # print("Cropped Column Mask Shape:", cropped_column_mask.shape)
        column_boundaries, num_columns = detect_column_boundaries(original_column_mask[ymin:ymax, xmin:xmax],cropped_table_mask)
        label = 'Table' if num_columns > 0 else 'Figure'
        # print(num_columns)

        print("tillu")

        # Process and display tables
        if label == 'Table':

            ocr_text = pytesseract.image_to_string(table_region, config='--psm 6')
            # print("Full OCR Text:\n", ocr_text)
            # st.write("### Full OCR Text:")
            # st.text(ocr_text)

            # Split OCR output into lines for processing
            ocr_lines = ocr_text.splitlines()
            split_ocr_data = []

            # Initialize column data storage for each boundary
            column_data = [[] for _ in range(len(column_boundaries))]

            # Process each column based on boundaries and display column-wise OCR
            for col_index, (xmin, xmax) in enumerate(column_boundaries):
                # Crop the specific column segment in the table region
                column_segment = table_region.crop((xmin, 0, xmax, table_region.height))
                column_text = pytesseract.image_to_string(column_segment, config='--psm 6')
                if not column_text:
                    continue  # Skip to the next column if this one is empty

                # Append each column's OCR text after processing line by line
                for line in column_text.splitlines():
                    column_data[col_index].append(line.strip())

            # Determine the maximum number of rows across all columns
            max_rows = max(len(col) for col in column_data) if column_data else 0

            # Remove empty cells and push up non-empty cells in each column
            processed_columns = []
            for col in column_data:
                # Filter out empty cells
                non_empty_cells = [cell for cell in col if cell]
                
                if non_empty_cells:
                    padded_column = non_empty_cells + [''] * (max_rows - len(non_empty_cells))
                    processed_columns.append(padded_column)

            # Update column_data with processed columns
            column_data = processed_columns

            # Determine the new maximum number of rows after processing
            max_rows = max(len(col) for col in column_data) if column_data else 0

            # Transpose column data to match a table format (rows by columns)
            table_data = list(map(list, zip(*column_data))) if max_rows > 0 else []

            # Print final structured OCR output for debugging
            print("\nStructured OCR Output (Final Table):")
            for row in table_data:
                print(row)

            # Convert to DataFrame and display in Streamlit
            df = pd.DataFrame(table_data)
            st.write("### Structured OCR Output (Final Table)")
            st.write("Below is the structured table data based on OCR column segmentation:")
            st.dataframe(df, key=f"structured_table_df_{page_number}")

            # Prepare CSV for download
            csv_data = df.to_csv(index=False, header=False).encode('utf-8')
            st.download_button(
                label="Download Structured Table as CSV",
                data=csv_data,
                file_name=f"structured_ocr_table_page_{page_number}.csv",
                mime="text/csv",
                key=f"download_button_{page_number}"
            )

            # Draw column boundaries on the table region for visualization
            fig, ax = plt.subplots(1, 2, figsize=(15, 10))
            ax[0].imshow(table_region, cmap='gray')
            ax[0].set_title("Masked Table Region with Column Separation")
            ax[0].axis('off')

            # Draw column boundaries
            for start, end in column_boundaries:
                ax[0].add_patch(patches.Rectangle((start, 0), end - start, table_region.height, linewidth=2, edgecolor='red', facecolor='none'))

            # Display structured OCR as a table
            table_fig = ax[1].table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.1] * len(table_data[0]))
            table_fig.auto_set_font_size(False)
            table_fig.set_fontsize(10)
            table_fig.scale(1.5, 1.5)
            ax[1].axis('off')
            ax[1].set_title("Structured OCR Table")

            st.pyplot(fig)
            
            # Save structured OCR as CSV
            table_csv_path = os.path.join(table_dir, f"page_{page_number}_table.csv")
            pd.DataFrame(table_data).to_csv(table_csv_path, index=False, header=False)

            # Save table region as an image and caption as text
            table_image_path = os.path.join(table_dir, f"page_{page_number}_table.jpeg")
            table_region.save(table_image_path)
            table_caption_text = save_and_display_table_caption(image, caption_text_coordinates, page_number, output_directory)
            print("Hi")
        else:
            figure_image_path = os.path.join(figure_dir, f"page_{page_number}_figure.jpeg")
            table_region.save(figure_image_path)
            figure_caption_text = save_and_display_figure_caption(image, caption_text_coordinates, page_number, output_directory)

            # Display figure region and caption
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(table_region)
            ax[0].set_title(f"Figure Region: {image_name}")
            ax[1].axis('off')
            ax[1].text(0.5, 0.5, figure_caption_text, fontsize=14, ha='center', va='center')  # Use figure_caption_text here
            ax[1].set_title("Detected Caption OCR")
            st.pyplot(fig)

            # Debug information for confirmation
            print(f"Figure saved: Image at {figure_image_path}, Caption text: {figure_caption_text}")
        page_number+=1
    print("\nProcessing complete. Results saved.")

# Main Streamlit app
def main():
    output_dir = 'output'
    table_dir = os.path.join(output_dir, 'tables')
    figure_dir = os.path.join(output_dir, 'figures')
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    st.title("Document Layout Analysis")
    st.markdown("Upload a PDF to convert each page into images and run object detection.")

    faster_rcnn_model = load_faster_rcnn_model()
    st.success("Models loaded successfully!")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file is not None:
        if not os.path.exists(TEST_IMAGES_DIR):
            os.makedirs(TEST_IMAGES_DIR)
        else:
            for f in os.listdir(TEST_IMAGES_DIR):
                os.remove(os.path.join(TEST_IMAGES_DIR, f))

        # Convert PDF pages to images
        # Streamlit option for user to select full PDF processing or just 10 pages
        st.markdown("### Processing Options")
        # Radio button to select whether to process the entire PDF or just the first 6 pages
        process_full_pdf = st.radio("Process Full PDF?", ("Yes", "No"))

        # Convert PDF pages to images based on the user's choice
        pages = convert_from_bytes(uploaded_file.read(), dpi=300, fmt="jpeg")
        num_pages = len(pages) if process_full_pdf == "Yes" else min(len(pages), 6)

        # Save only the selected number of pages as images
        for i, page in enumerate(pages[:num_pages]):
            page.save(os.path.join(TEST_IMAGES_DIR, f"page_{i + 1}.jpeg"), "JPEG")

        st.success(f"{num_pages} pages converted to images successfully.")

        # Process all selected pages with Faster R-CNN
        process_with_faster_rcnn(os.listdir(TEST_IMAGES_DIR), faster_rcnn_model)
        st.cache_resource.clear() 

        # Next, process tables from the generated CSV with VGG-19
        vgg19_model = load_vgg19_model()
        process_with_vgg('bounding_boxes.csv', vgg19_model)

if __name__ == "__main__":
    main()
