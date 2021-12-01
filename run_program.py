from parameters import *
from utils import *

params = Parameters("./antrenare/clasic")

data = get_data(params.train_path, params.image_type, params.answer_type, params.answer_name, params.bonus_answer_name)
data = sorted(data, key=lambda item: item["number"].lower())
processed_images = []

# preprocess_image(images[2], params, debug=True)

for items in data:
    processed_image = preprocess_image(items["image"], params, debug=False)
    items["processed_image"] = processed_image

for items in data:
    vertical_lines, horizontal_lines = get_lines_columns(params.crop_width, params.crop_height)
    answer = ""
    patches = get_patches(items["processed_image"], vertical_lines, horizontal_lines, debug=False)
    for line in range(len(patches)):
        for patch in patches[line]:
            exists = decide_digit_existence(patch)
            answer += "x" if exists else "o"
        answer += '\n' if (line < len(patches) - 1) else ""
    items["patches"] = patches
    items["answer"] = answer
    print("Example " + items["number"] + ": ", end="")
    print(items["answer"] == items["true_answer"])
    # print(items["answer"])
    # print("----------------")
    # print(items["true_answer"])
    # print("********************")

write_answers(data, params.answer_path, params.answer_type, params.predicted_answer_name)



