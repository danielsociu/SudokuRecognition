from helpers.utils import *

params = Parameters("./antrenare/clasic")

model = keras.models.load_model(params.model_path)

# Getting the data
data = get_data(params.train_path, params.image_type, params.answer_type, params.answer_name, params.bonus_answer_name, params.answers_included)
data = sorted(data, key=lambda item: item["number"].lower())
processed_images = []

# Processing the data
for items in data:
    processed_image = preprocess_image(items["image"], params, debug=False)
    items["processed_image"] = processed_image

# Main algorithm
for items in data:
    vertical_lines, horizontal_lines = get_lines_columns(params.crop_width, params.crop_height)
    answer = ""
    answer_bonus = ""

    patches = get_patches(items["processed_image"], vertical_lines, horizontal_lines, params.percentage, debug=False)

    for line in range(len(patches)):
        for patch in patches[line]:
            exists = decide_digit_existence(patch, debug=False)
            if exists:
                digit = guess_digit(model, patch)
                answer_bonus += str(digit)
            else:
                answer_bonus += "o"
            answer += "x" if exists else "o"
        answer += '\n' if (line < len(patches) - 1) else ""
        answer_bonus += '\n' if (line < len(patches) - 1) else ""

    items["patches"] = patches
    items["answer"] = answer
    items["answer_bonus"] = answer_bonus
    print("Example " + items["number"] + ": ")
    print(items["answer"])
    print("----------------")
    print(items["answer_bonus"])
    print("********************")
    # print(items["answer"] == items["true_answer"])
    # print(items["answer"])
    # print("----------------")
    # print(items["true_answer"])
    # print("********************")

# Writes the answers to files
write_answers(data, params.answer_path, params.answer_type, params.predicted_answer_name)
write_answers(data, params.answer_path, params.answer_type, params.predicted_bonus_answer_name, True)



