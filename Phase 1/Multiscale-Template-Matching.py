# import all the required libraries packages
import cv2
import numpy as np
import json
import os
from timeit import default_timer as timer
from skimage.io import imread_collection
import sys



# define function for finding key matches
def find_matches(descriptor_query, descriptor_train, keypoint_query, keypoint_train):
    
    start_time = timer()
    num_matches = 0
    
# FLANN parameters
    FLANN_INDEX_KDTREE = 0
    flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(flann_params, search_params)

    if len(keypoint_query) >= 2 and len(keypoint_train) >= 2:
        matches = flann.knnMatch(descriptor_query, descriptor_train, k=2)
    
    # Need to draw only good matches, so create a mask
    matches_mask = [[0,0] for i in range(len(matches))]
    
    # Ratio test
    for i, (match1, match2) in enumerate(matches):
        if match1.distance < 0.7 * match2.distance:
            matches_mask[i] = [1, 0]
            num_matches += 1
    
    print('Number of matches:', num_matches)
    end_time = timer()
    print('Time taken to find matches:', (end_time - start_time))
    return (num_matches, matches)


def keypoints_descriptors(query_images, train_images):
    
 
    # get image
    for img in query_images:
        
        # find the keypoints and descriptors with SIFT
        keypoints_query, descriptors_query = detector.detectAndCompute(img,None)
        keypoints_descriptors_query.append((keypoints_query, descriptors_query))
     
    # get template
    for img in train_images:
        
        img_train = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # find the keypoints and descriptors with SIFT
        keypoints_train, descriptors_train = detector.detectAndCompute(img_train,None)
        keypoints_descriptors_train.append((keypoints_train, descriptors_train))
  
    return(keypoints_descriptors_query, keypoints_descriptors_train)


def get_image_names(folder_path):
    
    image_names = []
    
    for filename in os.listdir(folder_path):
        image_names.append(filename)
    return image_names


def match_template_with_query_images(train_images, query_images, train_keypoints_descriptors, query_keypoints_descriptors, query_names, train_names):
    
    results_dict = {}
    
    # loop through each template image
    for i, train_image in enumerate(train_images):
        
        # skip image with very few features or too small in size
        if (train_names[i] == 'INSERT IMAGE NAMES THAT DOES NOT HAVE MANY FEATURES OR TOO SMALL'):
            results_dict['na'].append((train_names[i], []))
            continue
        
        # convert image to grayscale and resize to work with small templates
        train_image_gray = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY)
        # run a loop through images
        for j, query_image in enumerate(query_images):
        
            # find keypoint matches between query and template image
            query_keypoints, template_keypoints = query_keypoints_descriptors[j][1], train_keypoints_descriptors[i][1]
            query_descriptors, template_descriptors = query_keypoints_descriptors[j][0], train_keypoints_descriptors[i][0]
            keypoint_matches, matches = find_matches(query_keypoints, template_keypoints, query_descriptors, template_descriptors)
            
            
            # skip if no matches found
            if matches == 0:
                results_dict['na'].append((train_names[i], []))
                continue
            
            # add query image path as key in dictionary
            if query_names[j] not in results_dict.keys():
                results_dict.setdefault(query_names[j], [])
            
            
            # to check for major matches
            if keypoint_matches > 50:
                
                # compute matches with distance less than 0.75
                good_matches=[]
                for match1, match2 in matches:
                    if match1.distance < 0.55 * match2.distance:
                        good_matches.append(match1)
                

                  # check if enough matches found and get template and query image keypoints
                if len(good_matches) > MIN_MATCH_COUNT:
                    template_points = []
                    query_points = []
                    for match in good_matches:
                        template_points.append(train_keypoints_descriptors[i][0][match.trainIdx].pt)
                        query_points.append(query_keypoints_descriptors[j][0][match.queryIdx].pt)
                    template_points, query_points = np.float32((template_points, query_points))
                    homography, status = cv2.findHomography(template_points, query_points, cv2.RANSAC, 3.0)    
                    
                     # get coordinates of corner points and add to dictionary
                    template_height, template_width = train_image_gray.shape
                    template_corners = np.float32([[[0, 0], [0, template_height-1], [template_width-1, template_height-1], [template_width-1, 0]]])
                    if homography is not None:
                        query_corners = cv2.perspectiveTransform(template_corners, homography)
                        cv2.polylines(query_image, [np.int32(query_corners)], True, (0, 255, 0), 2)
                        print('query corners: ', query_corners)
                        print("Object found - %d/%d" % (len(good_matches), MIN_MATCH_COUNT))
                        print(template_corners)
                        results_dict[query_names[j]].append(tuple((train_names[i], [int(query_corners[0][0][0]), int(query_corners[0][0][-1]), int(query_corners[0][2][0]), int(query_corners[0][2][-1])])))
                        print(results_dict)
                        break
                else:
                    print("Not enough matches found - %d/%")
        
        # if no template has found match, add it to 'na' key in dictionary
        else:
            results_dict['na'].append(tuple((train_names[i],[])))
            print(results_dict)
    
    return(results_dict)

def get_current_folder_path():

    current_file_path = os.path.abspath(sys.argv[0])
    current_file_path_parts = current_file_path.split(os.sep)
    current_working_dir = current_file_path_parts[:-1]

    #print("current_working_dir", current_working_dir)

    return (os.sep.join(current_working_dir))



def main (images_folder_path, crop_images_path):

    # Get names of images
    train_image_names = get_image_names(images_folder_path)
    query_image_names = get_image_names(crop_images_path)

    # Create file paths for image collections
    train_image_collection_path = images_folder_path + "/*.jpg"
    query_image_collection_path = crop_images_path + "/*.jpg"

    # Load image collections
    train_image_collection = imread_collection(train_image_collection_path)
    query_image_collection = imread_collection(query_image_collection_path)

    # Extract keypoints and descriptors
    keypoints_descriptors_query, keypoints_descriptors_train = keypoints_descriptors(query_image_collection, train_image_collection)

    # Match templates with query images
    matching_results = match_template_with_query_images(train_image_collection, query_image_collection, keypoints_descriptors_train, keypoints_descriptors_query, query_image_names, train_image_names)

    # Create a json file for the dictionary of matching results
    with open(os.sep.join([current_directory,'result/data.json']), 'w+') as file:
        json.dump(matching_results, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    
    # create SIFT detector
    detector=cv2.xfeatures2d.SIFT_create()

    current_directory = get_current_folder_path()
    print("current working directory",current_directory)
    
    # set path variables
    images_folder_path = os.sep.join([current_directory,"images_crops"])
    crop_images_path = os.sep.join([current_directory,"images"])

    # set global variables
    MIN_MATCH_COUNT = 20
    results_dict = {}
    results_dict.setdefault('na', [])
    keypoints_descriptors_query = []
    keypoints_descriptors_train = []
    image_collection_train = []
    image_collection_query = []

    # call main function
    main(images_folder_path, crop_images_path)