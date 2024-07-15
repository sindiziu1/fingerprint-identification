import csv
import cv2 as cv
from glob import glob
import os
import numpy as np
from poincare import calculate_singularities
from segmentation import create_segmented_and_variance_images
from normalization import normalize
from gabor_filter import gabor_filter
from frequency import ridge_freq
import orientation
from crossing_number import calculate_minutiaes
from  tqdm import tqdm
from skeletonize import skeletonize


def f(input_img):
    # normalization -> orientation -> frequency -> mask -> filtering
    block_size = 16

    # normalization - removes the effects of sensor noise and finger pressure differences.
    normalized_img = normalize(input_img.copy(), float(100), float(100))

    # ROI and normalisation
    (segmented_img, normim, mask) = create_segmented_and_variance_images(
        normalized_img, block_size, 0.2
    )

    # orientations
    angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
    orientation_img = orientation.visualize_angles(
        segmented_img, mask, angles, W=block_size
    )

    # find the overall frequency of ridges in Wavelet Domain
    freq = ridge_freq(
        normim,
        mask,
        angles,
        block_size,
        kernel_size=5,
        minWaveLength=5,
        maxWaveLength=15,
    )

    # create gabor filter and do the actual filtering
    gabor_img = gabor_filter(normim, angles, freq)

    # thinning or skeletonize
    thin_image = skeletonize(gabor_img)

    # minutias
    minutiae, minutiae_list = calculate_minutiaes(thin_image)

    # singularities
    singularities_img, singularities_list = calculate_singularities(thin_image, angles, 1, block_size, mask)

    # Combine minutiae_list and singularities_list into features
    features = {
        "minutiae_list": minutiae_list,
        "singularities_list": singularities_list,
    }

    # visualize pipeline stage by stage
    output_imgs = [
        input_img,
        normalized_img,
        segmented_img,
        orientation_img,
        gabor_img,
        thin_image,
        minutiae,
        singularities_img,
    ]

    for i in range(len(output_imgs)):
        if len(output_imgs[i].shape) == 2:
            output_imgs[i] = cv.cvtColor(output_imgs[i], cv.COLOR_GRAY2RGB)
    results = np.concatenate(
        [np.concatenate(output_imgs[:4], 1), np.concatenate(output_imgs[4:], 1)]
    ).astype(np.uint8)

    return results, features


if __name__ == "__main__":
    # open images
    img_dir = "./input_test/*"
    output_dir = "./output_test/"

    def open_images(directory):
        images_paths = glob(directory)
        return [(img_path, cv.imread(img_path, 0)) for img_path in images_paths]

    images = open_images(img_dir)

    def extract_label(filename):
        return filename.split('_')[0]  # Extract the label from the filename


    # empty features and labels
    features = []
    labels = []

    images_data = []

    # Iterate over each image
    for img_path, img in tqdm(images):
        filename = os.path.basename(img_path)
        result_img, feature = f(img)
        label = extract_label(filename)  # Extract label from filename

        minutiae_info = feature.get('minutiae_list', [])
        singularities_info = feature.get('singularities_list', [])

        # Combine minutiae and singularity info into one list
        combined_info = minutiae_info + singularities_info

        # Create a dictionary to represent the image and its minutiae/singularities
        image_data = {
            'filename': filename,
            'label': label,
            'image': result_img,  # Store the processed image
            'features': combined_info  # Store the combined minutiae and singularities
        }

        # Append the image data dictionary to the list
        images_data.append(image_data)

        # Save the processed image
        cv.imwrite(os.path.join(output_dir, filename), result_img)

    # Now, images_data list contains all the images along with their associated minutiae/singularities


    # image pipeline
    os.makedirs(output_dir, exist_ok=True)

