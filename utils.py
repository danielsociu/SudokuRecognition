import cv2 as cv
import numpy as np
from parameters import *
import os


def show_image(title, image):
    resized_image = cv.resize(image.copy(), (1000, 1000))
    cv.imshow(title, resized_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_images(path, image_type):
    files = os.listdir(path)
    images = []
    for file in files:
        if file[-3:] == image_type:
            img = cv.imread(os.path.join(path, file))
            images.append(img)

    return np.array(images)


def get_files(path, file_type, bonus=False):
    all_files = os.listdir(path)
    files = []
    for file in all_files:
        if file[-3:] == file_type:
            file_path = path + '/' + file
            if bonus and "bonus" in file:
                data = get_text_file_contents(file_path)
                files.append(data)
            elif not bonus and "bonus" not in file:
                data = get_text_file_contents(file_path)
                files.append(data)
    return np.array(files)


def get_data(path, image_type, answer_type, answer_name, bonus_answer_name):
    files = os.listdir(path)
    data = []
    for file in files:
        if file[-3:] == image_type:
            number = file[:-3]
            img = cv.imread(os.path.join(path, file))
            answer_path = path + '/' + number + answer_name + answer_type
            bonus_answer_path = path + '/' + number + bonus_answer_name + answer_type
            answer = get_text_file_contents(answer_path)
            bonus_answer = get_text_file_contents(bonus_answer_path)
            data.append({
                "number": number,
                "image": img,
                "answer": answer,
                "bonus_answer": bonus_answer
            })


def get_text_file_contents(path):
    open_file = open(path, "r")
    data = "".join(open_file.readlines())
    open_file.close()
    return data


def sharpen_image(image, debug=False):
    grayed_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_median_blurred = cv.medianBlur(grayed_image, 5)
    image_gaussian_blurred = cv.GaussianBlur(grayed_image, (0, 0), 7)
    image_sharpened = cv.addWeighted(image_median_blurred, 1.2, image_gaussian_blurred, -0.8, 0)
    # _, thresh = cv.threshold(image_sharpened, 20, 255, cv.THRESH_BINARY)
    thresh = cv.adaptiveThreshold(image_sharpened, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 10)

    kernel = np.ones((6, 6), np.uint8)
    thresh = cv.erode(thresh, kernel)

    if debug:
        show_image("default", grayed_image)
        show_image("median", image_median_blurred)
        show_image("gaussian", image_gaussian_blurred)
        show_image("sharpened", image_sharpened)
        show_image("thresh", thresh)

    return thresh


def preprocess_image(image, parameters: Parameters, debug=False):
    grayed_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = sharpen_image(image, debug)

    edges = cv.Canny(thresh, 150, 400)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    top_left = 0
    top_right = 0
    bottom_left = 0
    bottom_right = 0
    max_area = 0

    for i in range(len(contours)):
        if len(contours[i]) > 3:
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point
                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + \
                        possible_bottom_right[1]:
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis=1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]

            cornersArray = np.array(
                [[possible_top_left], [possible_top_right], [possible_bottom_right], [possible_bottom_left]])
            if cv.contourArea(cornersArray) > max_area:
                max_area = cv.contourArea(cornersArray)
                top_left = possible_top_left
                top_right = possible_top_right
                bottom_right = possible_bottom_right
                bottom_left = possible_bottom_left

    corners = np.float32([top_left, top_right, bottom_right, bottom_left])
    cropped_image = crop_resize_image(image.copy(), corners)
    resized_image = cv.resize(cropped_image, (parameters.crop_width, parameters.crop_height))

    if debug:
        image_copy = cv.cvtColor(grayed_image.copy(), cv.COLOR_GRAY2BGR)
        cv.circle(image_copy, tuple(top_left), 4, (0, 0, 255), -1)
        cv.circle(image_copy, tuple(top_right), 4, (0, 0, 255), -1)
        cv.circle(image_copy, tuple(bottom_right), 4, (0, 0, 255), -1)
        cv.circle(image_copy, tuple(bottom_left), 4, (0, 0, 255), -1)

        show_image("corners image", image_copy)
        show_image("cropped image", resized_image)

    return resized_image


def get_lines_columns(width, height):
    lines_vertical = []
    for i in range(0, width, width // 9):
        line = [(i, 0), (i, height - 1)]
        lines_vertical.append(line)
    lines_horizontal = []
    for i in range(0, height, height // 9):
        line = [(i, 0), (i, width - 1)]
        lines_horizontal.append(line)
    return lines_vertical, lines_horizontal


def crop_resize_image(image, corners):
    top_left, top_right, bottom_left, bottom_right = corners
    top_width = np.sqrt((top_left[0] - top_right[0]) ** 2 + (top_left[1] - top_right[1]) ** 2)
    bottom_width = np.sqrt((bottom_left[0] - bottom_right[0]) ** 2 + (bottom_left[1] - bottom_right[1]) ** 2)
    left_height = np.sqrt((top_left[0] - bottom_left[0]) ** 2 + (top_left[1] - bottom_left[1]) ** 2)
    right_height = np.sqrt((top_right[0] - bottom_right[0]) ** 2 + (top_right[1] - bottom_right[1]) ** 2)

    maximal_width = max(int(top_width), int(bottom_width))
    maximal_height = max(int(left_height), int(right_height))

    points_coords = np.float32([[0, 0],
                                [maximal_width - 1, 0],
                                [maximal_width - 1, maximal_height - 1],
                                [0, maximal_height - 1]])

    transformation_matrix = cv.getPerspectiveTransform(corners, points_coords)
    warped_image = cv.warpPerspective(image, transformation_matrix, (maximal_width, maximal_height))

    return warped_image


def get_patches(cropped_image, vertical_lines, horizontal_lines, debug=False):
    patches = []
    for i in range(len(horizontal_lines) - 1):
        patches.append([])
        for j in range(len(vertical_lines) - 1):
            y_min = vertical_lines[j][0][0]
            y_max = vertical_lines[j + 1][0][0]
            x_min = horizontal_lines[i][0][0]
            x_max = horizontal_lines[i + 1][0][0]
            height = abs(y_max - y_min)
            width = abs(x_max - x_min)
            difference_height = height // 10
            difference_width = width // 10
            y_min += difference_height
            y_max -= difference_height
            x_min += difference_width
            x_max -= difference_width
            patch = cropped_image[x_min: x_max, y_min: y_max].copy()
            patches[i].append(patch)
            if debug:
                show_image("patch", patch)
    return patches


def decide_digit_existence(patch, debug=False):
    thresh = sharpen_image(patch)
    mean_pixels = np.mean(thresh)
    if debug:
        show_image("inital", patch)
        show_image("digit image", thresh)
        print(np.mean(thresh))
    if mean_pixels > 250:
        return False
    else:
        return True


