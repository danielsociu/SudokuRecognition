from helpers.parameters import *
import os


def show_image(title, image):
    """
    Displays an image resized to 1000x1000
    """
    resized_image = cv.resize(image.copy(), (1000, 1000), interpolation=cv.INTER_NEAREST)
    cv.imshow(title, resized_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_data(path, image_type, answer_type, answer_name, bonus_answer_name, answers_included=False):
    """
    Gets the data from the path (images and answers)
    """
    files = os.listdir(path)
    data = []
    for file in files:
        if file[-3:] == image_type:
            number = file[:-4]
            img = cv.imread(os.path.join(path, file))
            img = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
            if answers_included:
                answer_path = path + '/' + number + answer_name + '.' + answer_type
                bonus_answer_path = path + '/' + number + bonus_answer_name + '.' + answer_type
                answer = get_text_file_contents(answer_path)
                bonus_answer = get_text_file_contents(bonus_answer_path)
                data.append({
                    "number": number,
                    "image": img,
                    "true_answer": answer,
                    "ture_bonus_answer": bonus_answer
                })
            else:
                data.append({
                    "number": number,
                    "image": img,
                })
    return data


def write_answers(data, answers_path, answer_type, answer_name):
    """
    Writes the answers to the files
    """
    final_path = os.path.join(os.getcwd(), answers_path)
    if not os.path.isdir(final_path):
        os.makedirs(final_path)
    for items in data:
        file_name = items["number"] + answer_name + '.' + answer_type
        file_path = os.path.join(final_path, file_name)
        with open(file_path, 'w') as f:
            f.write(items["answer"])


def get_text_file_contents(path):
    open_file = open(path, "r")
    data = "".join(open_file.readlines())
    open_file.close()
    return data


def sharpen_image(image, debug=False):
    """
    Sharpens the image, applying median and gaussian blur, then doing the difference
    After that we apply an adaptive binary threshold
    """
    grayed_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_median_blurred = cv.medianBlur(grayed_image, 5)
    image_gaussian_blurred = cv.GaussianBlur(grayed_image, (0, 0), 9)
    image_sharpened = cv.addWeighted(image_median_blurred, 1.2, image_gaussian_blurred, -0.8, 0)
    # _, thresh = cv.threshold(image_sharpened, 20, 255, cv.THRESH_BINARY)
    thresh = cv.adaptiveThreshold(image_sharpened, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 8)

    kernel = np.ones((6, 6), np.uint8)
    thresh = cv.erode(thresh, kernel)

    if debug:
        show_image("default", grayed_image)
        show_image("median", image_median_blurred)
        show_image("gaussian", image_gaussian_blurred)
        show_image("sharpened", image_sharpened)
        show_image("thresh", thresh)

    return thresh


def sharpen_digit(image, debug=False):
    """
    Sharpen digit for the purpose of deciding if there exists a digit or not
    """
    grayed_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_median_blurred = cv.medianBlur(grayed_image, 5)
    image_gaussian_blurred = cv.GaussianBlur(grayed_image, (0, 0), 9)
    image_sharpened = cv.addWeighted(image_median_blurred, 1.2, image_gaussian_blurred, -0.8, 0)
    _, thresh = cv.threshold(image_sharpened, 20, 255, cv.THRESH_BINARY)

    kernel = np.ones((6, 6), np.uint8)
    thresh = cv.erode(thresh, kernel)

    if debug:
        show_image("default", grayed_image)
        show_image("median", image_median_blurred)
        show_image("gaussian", image_gaussian_blurred)
        show_image("sharpened", image_sharpened)
        show_image("thresh", thresh)

    return thresh


def preprocess_image(image, parameters: Parameters, bigger=False, debug=False):
    """
    Preprocesses the vanilla image, and selects the corners of the sudoku field
    """
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
    if bigger:
        size = (
        parameters.crop_width + parameters.crop_width // 5, parameters.crop_height + parameters.crop_height // 5)
    else:
        size = (parameters.crop_width, parameters.crop_height)

    resized_image = cv.resize(cropped_image, size)

    if debug:
        image_copy = cv.cvtColor(grayed_image.copy(), cv.COLOR_GRAY2BGR)
        cv.circle(image_copy, tuple(top_left), 4, (0, 0, 255), -1)
        cv.circle(image_copy, tuple(top_right), 4, (0, 0, 255), -1)
        cv.circle(image_copy, tuple(bottom_right), 4, (0, 0, 255), -1)
        cv.circle(image_copy, tuple(bottom_left), 4, (0, 0, 255), -1)

        show_image("corners image", image_copy)
        show_image("cropped image", resized_image)

    return resized_image


def border_gray_image(image, percentage, color):
    """
    Adds a border border to the image
    """
    height, width = image.shape
    for i in range(width * percentage // 100):
        for j in range(height):
            image[i][j] = color
            image[width - i - 1][j] = color
    for i in range(height * percentage // 100):
        for j in range(width):
            image[j][i] = color
            image[j][height - i - 1] = color

    return image


def zones_image(image, horizontal_lines, vertical_lines, percentage, debug=False):
    """
    Applies blurs and thresholds such that our jigsaw image will remain only with the area lines
    """
    grayed_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    grayed_image = border_gray_image(grayed_image, 1, 50)
    # first preprocessing
    image_median_blurred = cv.medianBlur(grayed_image, 15)
    image_gaussian_blurred = cv.GaussianBlur(grayed_image, (0, 0), 9)
    image_sharpened = cv.addWeighted(image_median_blurred, 2.0, image_gaussian_blurred, -0.3, 0)
    re_median_image = cv.medianBlur(image_sharpened, 11)
    thresh = cv.adaptiveThreshold(re_median_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 99, 20)
    # _, thresh = cv.threshold(re_median_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv.erode(thresh, kernel)

    for i in range(len(horizontal_lines) - 1):
        for j in range(len(vertical_lines) - 1):
            y_min = vertical_lines[j][0][0]
            y_max = vertical_lines[j + 1][0][0]
            x_min = horizontal_lines[i][0][0]
            x_max = horizontal_lines[i + 1][0][0]
            height = abs(y_max - y_min)
            width = abs(x_max - x_min)
            difference_height = height * percentage // 100
            difference_width = width * percentage // 100
            y_min += difference_height
            y_max -= difference_height
            x_min += difference_width
            x_max -= difference_width
            thresh[x_min: x_max, y_min: y_max] = 255

    thresh = border_gray_image(thresh, 1, 0)
    if debug:
        # image_sharpened = cv.addWeighted(thresh, 2.0, image_gaussian_blurred, -1.0, 0)
        show_image("median", image_median_blurred)
        show_image("gaussian", image_gaussian_blurred)
        show_image("sharpened", image_sharpened)
        show_image("re-median", re_median_image)
        show_image("thresh", thresh)
    return np.array(thresh)


def fill_zone(image, zone_map, x, y, value):
    """
    Does a fill for a zone in the jigsaw sudoku
    """
    height, width = zone_map.shape
    dx = [1, -1, 0, 0]
    dy = [0, 0, -1, 1]
    zone_map[x][y] = value
    que = [(x, y)]
    while len(que) > 0:
        cur_x, cur_y = que.pop(0)
        for i in range(len(dx)):
            new_x = cur_x + dx[i]
            new_y = cur_y + dy[i]
            if 0 <= new_x < height and 0 <= new_y < width and image[new_x][new_y] == 255 and zone_map[new_x][
                new_y] == 0:
                zone_map[new_x][new_y] = value
                que.append((new_x, new_y))
    return zone_map


def get_zones(zonal_image, vertical_lines, horizontal_lines, percentage):
    """
    Returns the a zonal map that represents the jigsaw sudoku areas
    """
    current_zone = 1
    zonal_map = np.zeros(zonal_image.shape, np.uint8)
    for i in range(len(horizontal_lines) - 1):
        for j in range(len(vertical_lines) - 1):
            y_min = vertical_lines[j][0][0]
            y_max = vertical_lines[j + 1][0][0]
            x_min = horizontal_lines[i][0][0]
            x_max = horizontal_lines[i + 1][0][0]
            height = abs(y_max - y_min)
            width = abs(x_max - x_min)
            difference_height = height * percentage // 100
            difference_width = width * percentage // 100
            y_min += difference_height
            y_max -= difference_height
            x_min += difference_width
            x_max -= difference_width
            # counter = 0
            for x in range(x_min, x_max):
                # if counter > 150:
                #     continue
                for y in range(y_min, y_max):
                    if zonal_image[x][y] == 0:
                        zonal_map[x][y] = -1
                    # if zonal_map[x][y] != 0:
                    #     counter += 1
                    #     continue
                    elif zonal_image[x][y] == 255 and zonal_map[x][y] == 0:
                        zonal_map = fill_zone(zonal_image, zonal_map, x, y, current_zone)
                        current_zone += 1
    return zonal_map


def get_frequencies(matrix):
    """
    Frequencies of the areas in a small image
    """
    freq = {}
    for line in matrix:
        for elem in line:
            if elem != 255 and elem != 0:
                freq[elem] = freq.get(elem, 0) + 1
    return freq


def decide_zone(zone_map, x, y, vertical_lines, horizontal_lines, percentage):
    """
    Given a patch indexing will return the corresponding patch zone
    """
    y_min = vertical_lines[y][0][0]
    y_max = vertical_lines[y + 1][0][0]
    x_min = horizontal_lines[x][0][0]
    x_max = horizontal_lines[x + 1][0][0]
    height = abs(y_max - y_min)
    width = abs(x_max - x_min)
    difference_height = height * percentage // 100
    difference_width = width * percentage // 100
    y_min += difference_height
    y_max -= difference_height
    x_min += difference_width
    x_max -= difference_width
    patch = zone_map[x_min: x_max, y_min: y_max].copy()
    frequencies = get_frequencies(patch)
    frequencies = sorted(frequencies.items(), key=lambda kv: kv[1])
    return frequencies[0][0]


def get_lines_columns(width, height):
    lines_vertical = []
    for i in range(0, width, width // 9):
        line = [(i, 0), (i, height - 1)]
        lines_vertical.append(line)
    lines_horizontal = []
    for i in range(0, height, height // 9):
        line = [(i, 0), (i, width - 1)]
        lines_horizontal.append(line)
    return np.array(lines_vertical), np.array(lines_horizontal)


def crop_resize_image(image, corners):
    """
    Resizes and crops the initial image
    """
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


def get_patches(cropped_image, vertical_lines, horizontal_lines, percentage, debug=False):
    """
    Returns the vector of patches based on the image size
    """
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
            difference_height = height * percentage // 100
            difference_width = width * percentage // 100
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
    """
    Decides whether an digit exists in the patch or not
    """
    thresh = sharpen_digit(patch)
    mean_pixels = np.mean(thresh)
    if debug:
        show_image("initial", patch)
        show_image("digit image", thresh)
        print(np.mean(thresh))
    if mean_pixels > 230:
        return False
    else:
        return True
