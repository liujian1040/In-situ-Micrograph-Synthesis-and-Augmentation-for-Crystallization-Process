import os
import numpy as np
import cv2
from copy import deepcopy
import time
import datetime
import torch
import argparse
import pandas as pd
from ultralytics import YOLO
from ultralytics.yolo.utils.ops import scale_boxes, xyxy2xywh, non_max_suppression, process_mask_native


def single_img_ana(img_file):
    model = YOLO('yolov8n-seg.pt')
    model = YOLO('./best.pt')
    result = model.predict(source=img_file, save=True)
    segs = result[0].masks
    seg = segs.data  # segs.masks
    org = segs.orig_shape
    mask_in = segs.xy  # segs.segments
    ##print('seg.shape ', seg.shape)
    ##print('seg ', seg)
    # print('seg0 ', seg[0])
    # print('seg1 ', seg[1])
    # print('seg2 ', seg[2].shape)
    ##print('org ', org)

    ##print('mask_in ', mask_in)
    # seg = process_mask_native()
    # print(boxes, boxes.shape, seg)

    # Assuming you have a list of points named 'segment_points'
    segment_points = mask_in  # Replace with your segment points

    # Create an empty black image
    image = np.zeros(org, dtype=np.uint8)
    ##print(image.shape)

    # Draw the segment points on the image
    '''for segment in segment_points:
        contour_image = np.zeros_like(image)
        
        for point in segment:
            #x, y = point[0], point[1]
            cv2.circle(contour_image, (int(point[0]), int(point[1])), 1, (255), -1)
            #cv2.circle(image, tuple(point), 1, (255), -1)

        # Find the contours
        contours, hierarchy = cv2.findContours(contour_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours on a new image

        cv2.drawContours(image, contours, -1, (255), thickness=1)
        '''
    image_with_points = image.copy()
    crystal_list = []

    # Iterate over each segment
    for segment in segment_points:
        # Draw the points on the image
        for point in segment:
            cv2.circle(image_with_points, (int(point[0]), int(point[1])), 1, (255), -1)

    # Create a blank image with the same shape as the original image
    contour_image = np.zeros_like(image)

    # Iterate over each segment
    for segment in segment_points:
        # Convert the segment points to a numpy array
        try:

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
            # print(aspect_Ratio)
        except:
            print('wrong')
            continue
        crystal = {"file": img_file, "Contour Area": contour_area, "minor_axis": minor_axis, "major_axis": major_axis,
                   "circumference": circ, "aspect_Ratio": aspect_Ratio}
        crystal_list.append(crystal)

    return crystal_list

    # Display the contour image
    '''result_path = 'D:/yolov8/ultralytics-main/runs/segment/pred_res/medium.png'  # 替换为所需输出路径
    print(crystal_list)
    df = pd.DataFrame()

    for crystal_info in crystal_list:
                temp = pd.DataFrame.from_dict(crystal_info, orient='index').T
                df = pd.concat([df, temp], ignore_index=True)
    df.to_excel(output_file, engine='xlsxwriter')
    print('done')
    cv2.imwrite(result_path, contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''


def batch_img_ana(origin_img_dir, output_file):
    df = pd.DataFrame()
    for filename in os.listdir(origin_img_dir):
        if filename.endswith('png'):
            origin_img_path = origin_img_dir + "/" + filename

            crystal_info_list = single_img_ana(origin_img_path)
            for crystal_info in crystal_info_list:
                temp = pd.DataFrame.from_dict(crystal_info, orient='index').T
                df = pd.concat([df, temp], ignore_index=True)
    df.to_excel(output_file, engine='xlsxwriter')
    print('done')
    # cv2.imwrite(result_path, contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    output_file = '.xlsx'
    result_path = '' # This is for image prediction
    origin_img_dir = ''
    
    batch_img_ana(origin_img_dir, output_file)
