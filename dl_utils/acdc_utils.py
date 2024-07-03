
import numpy as np
from math import degrees
import nibabel as nib
import SimpleITK as sitk
from math import degrees

# def nii_reader(path):
#     reader = sitk.ImageFileReader()
#     reader.SetImageIO("")
#     reader.SetFileName(path)
#     img = reader.Execute()
#     return img


def nii_reader(path):
    file = nib.load(path)
    #tmp = file.get_fdata()
    img = sitk.GetImageFromArray(file.get_fdata())
    return img

def get_data(filename):
    img = nib.load(filename)
    return img.get_fdata()


def _lv_myo_center(segmentation, labels):
    [i, j] = np.where(segmentation == (labels['LV'] or labels['Myo']))
    return np.array([np.mean(i), np.mean(j)])


def _rv_center(segmentation, labels):
    [i, j] = np.where(segmentation == labels['RV'])
    return np.array([np.mean(i), np.mean(j)])


def compute_shift(segmentation, labels):
    segmentation_center = np.array(segmentation.shape[:2]) // 2
    return (_lv_myo_center(segmentation, labels) - segmentation_center).astype(float)


def compute_rotation(segmentation, labels):
    lv_myo_center = _lv_myo_center(segmentation, labels)
    rv_center = _rv_center(segmentation, labels)
    #print(lv_myo_center,rv_center)

    centers_diff = rv_center - lv_myo_center
    #print(centers_diff)

    rotation_angle = degrees(np.arctan2(centers_diff[1], centers_diff[0]))
    # print(rotation_angle)
    rotation_angle = -90 - ((rotation_angle + 360) % 360)

    return rotation_angle


def rotate_image(image, angle):
    # Input angle in degree

    # Define the rotation center as the center of the image
    center = np.array(image.GetSize()) / 2

    # Create a rotation transform
    rotation_transform = sitk.Euler2DTransform()
    rotation_transform.SetCenter(center)
    rotation_transform.SetAngle(np.deg2rad(angle))

    # Apply the rotation transform to the image
    rotated_image = sitk.Resample(image, image, rotation_transform)

    # Convert the rotated image back to a numpy array
    rotated_array = sitk.GetArrayFromImage(rotated_image)

    return rotated_image, rotated_array


def shift_image(image, shift):
    # Create a SimpleITK image from the matrix

    shift = list(shift[::-1])
    # Create a translation transform

    if isinstance(image, np.ndarray):
        image = sitk.GetImageFromArray(image)
    translation_transform = sitk.TranslationTransform(image.GetDimension())
    translation_transform.SetParameters(shift)

    # Apply the translation transform to the image
    shifted_image = sitk.Resample(image, image, translation_transform)

    # Convert the shifted image back to a numpy array
    shifted_array = sitk.GetArrayFromImage(shifted_image)

    return shifted_image, shifted_array


def align_case(image, segmentation, labels = {'LV':1, 'Myo':2, 'RV':3}):
    ### Input as sitk image

    shift = compute_shift(segmentation, labels)
    #print(shift)

    # Apply shift
    shifted_seg, seg_array = shift_image(segmentation, shift)
    shifted_image, _ = shift_image(image, shift)

    angle = compute_rotation(seg_array, labels)
    # Apply rotation
    t_segmentation, t_seg_array = rotate_image(shifted_seg, angle)
    t_image, t_img_array = rotate_image(shifted_image, angle)

    return t_img_array, t_seg_array

"""
def apply_rotation(matrix, angle):
    # Convert the matrix to an image
    image = np.array(matrix, dtype=np.uint8)

    # Rotate the image by the specified angle
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return rotated_image


def apply_shift(matrix, shift):
    image = np.array(matrix, dtype=np.uint8)

    # Calculate the shift values for x and y axes
    shift_y, shift_x = shift

    # Create a translation matrix
    translation_matrix = np.float32([[1, 0, shift_x],
                                     [0, 1, shift_y]])
    # Apply the translation to the image
    shifted_image = cv2.warpAffine(image, translation_matrix, image.shape[1::-1])

    return shifted_image
"""
