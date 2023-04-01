# MultiScale_Template_Matching_CV
### In the first phase of the code, the code creates an argument parser that accepts two arguments: 
a path to the template image and a path to the images where the template
will be matched.
Then there are three functions defined in this code: kp_des(), find_matches(), and
temp_query_match().
The kp_des() function takes in two image collections (coll_query and coll_train) and
returns their key points and descriptors. This function uses the SIFT (Scale-Invariant Feature
Transform) algorithm to detect keypoints and descriptors.
The find_matches() function takes in the descriptors of the query and training images
along with their keypoints and uses the FLANN (Fast Library for Approximate Nearest
Neighbors) algorithm to find the key matches.
The temp_query_match() function takes in two image collections (coll_train and
coll_query), their corresponding keypoints and descriptors (kp_des_train and
kp_des_query), and the names of the images (query_name and train_name). This
function matches the query image and the template image and returns a dictionary
containing the query image name as the key and the coordinates of the object in the
image as the value.
Overall, this part of code uses a combination of computer vision and machine learning
techniques to match a template image with other images and return the location of the
object in the query image.

### In the second phase of the code, the code is for identifying the student ID in an image.
The code uses computer vision techniques to detect a logo and an ID verification
template in the image, and then extracts the relevant information from the image.The
code imports the required libraries, which are cv2 for OpenCV and PIL for image
processing. It defines a function called Student_id_identifier which takes an image path
as input.
The function starts by loading the template images for the logo and the ID verification
template. It then loads the input image and gets the width and height of the templates.
The function performs template matching using the normalized correlation coefficient,
which is a measure of the similarity between two images. It calculates the correlation
coefficient between the input image and the template image at each location in the
input image. The result is a matrix of correlation coefficients, which indicates how well the
template matches the input image at each location.
The function then finds the location in the input image where the maximum correlation
coefficient occurs for both the logo and the ID verification template. If the maximum
correlation coefficient is above a threshold value of 0.9 for the logo and 0.98 for the ID
verification template, it indicates that the template has been found in the input image.
If the logo is found, the function draws a rectangle around the logo on the input image,
and extracts the logo, candidate photo, student name, and student ID from the input
image using the location of the logo. The function then displays the extracted images
and prints a message indicating that the student has been verified.
If the logo is not found, the function prints a message indicating that the input image is
invalid and asks the user to try again.
Here are the steps used in the code to match the templates:
1. Load the template and input images using cv2.imread.
2. Get the width and height of the templates using shape.
3. Perform template matching using cv2.matchTemplate, which returns a matrix of
correlation coefficients.
4. Find the location in the input image where the maximum correlation coefficient
occurs using cv2.minMaxLoc.
5. Draw a rectangle around the template on the input image using cv2.rectangle.
6. Extract the relevant information from the input image using Image. Open and
crop.
7. Display the extracted images using display.
8. Print a message indicating the status of the student ID verification.