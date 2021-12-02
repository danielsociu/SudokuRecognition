from helpers.utils import *
import cv2 as cv

params = Parameters("./antrenare/jigsaw")

# Getting the data
data = get_data(params.train_path, params.image_type, params.answer_type, params.answer_name, params.bonus_answer_name, params.answers_included)
data = sorted(data, key=lambda item: item["number"].lower())
processed_images = []

# Processing the data
for items in data:
    processed_image = preprocess_image(items["image"], params, bigger=False, debug=False)
    items["processed_image"] = processed_image

# Main algorithm
for items in data:
    vertical_lines, horizontal_lines = get_lines_columns(params.crop_width, params.crop_height)
    answer = ""

    # computation too big for 1000x1000 maps, so we resized them to 100x100 and made some sharpening and thresholds
    # so we use 100x100 maps for zonal decision
    items["zone_image"] = zones_image(items["processed_image"], horizontal_lines, vertical_lines, params.percentage, debug=False)
    items["zone_image"] = cv.resize(items["zone_image"], (0, 0), fx=1/params.lee_speed, fy=1/params.lee_speed, interpolation=cv.INTER_LINEAR)
    items["zone_image"] = cv.GaussianBlur(items["zone_image"], (0, 0), 1)
    _, items["zone_image"] = cv.threshold(items["zone_image"], 200, 255, cv.THRESH_BINARY)
    vertical_lines_lee, horizontal_lines_lee = get_lines_columns(*items["zone_image"].shape)
    items["zone_matrix"] = get_zones(items["zone_image"], vertical_lines_lee, horizontal_lines_lee, params.percentage)

    # getting the patches
    patches = get_patches(items["processed_image"], vertical_lines, horizontal_lines, params.percentage, debug=False)

    for line in range(len(patches)):
        for column, patch in enumerate(patches[line]):
            zone = decide_zone(items["zone_matrix"], line, column, vertical_lines_lee, horizontal_lines_lee, params.percentage)
            exists = decide_digit_existence(patch, debug=False)
            answer += str(zone)
            answer += "x" if exists else "o"
        answer += '\n' if (line < len(patches) - 1) else ""

    items["patches"] = patches
    items["answer"] = answer
    print("********************")
    print("Example " + items["number"] + ": ")
    print(items["answer"])
    # print(items["answer"] == items["true_answer"])
    # print(items["answer"])
    # print("----------------")
    # print(items["true_answer"])
    # print("********************")

# Writes the answers to files
write_answers(data, params.jigsaw_answer_path, params.answer_type, params.predicted_answer_name)



