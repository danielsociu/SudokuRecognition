from parameters import *
from utils import *

params = Parameters("./antrenare/clasic")

data = get_data(params.train_path)
images = get_images(params.train_path, params.image_type)
gt_answers = get_files(params.train_path, params.answer_type)
processed_images = []

# preprocess_image(images[2], params, debug=True)

for image in images:
    processed_image = preprocess_image(image, params, debug=False)
    processed_images.append(processed_image)

for index, processed_image in enumerate(processed_images):
    vertical_lines, horizontal_lines = get_lines_columns(params.crop_width, params.crop_height)
    answer = ""
    patches = get_patches(processed_image, vertical_lines, horizontal_lines, debug=False)
    for line in range(len(patches)):
        for patch in patches[line]:
            exists = decide_digit_existence(patch)
            answer += "x" if exists else "o"
        answer += '\n'
    print (answer)
    print ("----------------")
    print (gt_answers[index])
    print ("********************")




