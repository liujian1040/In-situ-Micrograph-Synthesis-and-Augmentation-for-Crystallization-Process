import os
import cv2
import numpy as np
import pandas as pd
import json

def extract_segment_points(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    segment_points = []

    # Iterate over each shape in the JSON file
    for shape in data['shapes']:
        # Extract the segment points for each shape
        if shape['shape_type'] == 'polygon':
            segment_points.append(shape['points'])
            #segment_points.append(shape['label'])

    return segment_points


def single_img_ana(img_file, json_file):
    # Read the image
    img = cv2.imread(img_file)

    # Extract segment points from the JSON file
    segment_points = extract_segment_points(json_file)

    # Perform any other necessary analysis
    # ...

    # Create an empty black image
    image = np.zeros_like(img)
    image_with_points = image.copy()
    crystal_list = []

    # Create a blank image with the same shape as the original image
    contour_image = np.zeros_like(img)

    # Iterate over each segment
    for segment in segment_points:
        try:
            # Convert the segment points to a numpy array
            segment_array = np.array(segment, dtype=np.int32)

            # Reshape the array to match the expected format for cv2.drawContours
            segment_array = segment_array.reshape((-1, 1, 2))

            # Draw the contour of the segment on the contour image
            cv2.drawContours(contour_image, [segment_array], 0, (255), thickness=1)
            contour_area = cv2.contourArea(segment_array)
            ##print("Contour Area:", contour_area)
            ellipse = cv2.fitEllipse(segment_array)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            ##print("minor_axis:", minor_axis)
            ##print("major_axis:", major_axis)
            circ = cv2.arcLength(segment_array, True)
            ##print("circ", circ)
            aspect_Ratio = major_axis / minor_axis
        except:
            print('wrong')
            continue

        crystal = {"file": img_file, "Contour Area": contour_area, "minor_axis": minor_axis, "major_axis": major_axis,
                   "circumference": circ, "aspect_Ratio": aspect_Ratio}
        crystal_list.append(crystal)

    # Calculate the length of each mask
    #mask_lengths = [cv2.arcLength(segment, True) for segment in segment_points]

    return crystal_list


def batch_img_ana(origin_img_dir, json_dir, output_file):
    df = pd.DataFrame()

    # Iterate over each image file in the directory
    for filename in os.listdir(origin_img_dir):
        if filename.endswith('.png'):
            img_file = os.path.join(origin_img_dir, filename)
            json_file = os.path.join(json_dir, filename.replace('.png', '.json'))

            # Analyze the image and extract mask lengths
            crystal_info_list = single_img_ana(img_file, json_file)

            # Create a dictionary with the image file and corresponding mask lengths
            for crystal_info in crystal_info_list:
                temp = pd.DataFrame.from_dict(crystal_info, orient='index').T
                df = pd.concat([df, temp], ignore_index=True)
        df.to_excel(output_file, engine='xlsxwriter')
    print('Done')

# Specify the input directories and output file path
origin_img_dir = ''
json_dir = ''
output_file = '.xls'

# Call the function to analyze the images and extract mask lengths
batch_img_ana(origin_img_dir, json_dir, output_file)
