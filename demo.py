import tensorflow as tf
import gradio as gr
from tensorflow import keras
from keras.preprocessing import image
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics 
import matplotlib.pyplot as plt
import numpy as np
import cv2

model = keras.models.load_model('./model')

class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

def predict(img):
    img_width, img_height = 96, 96

    img = cv2.resize(img, (img_width, img_height))
    img_tensor = image.img_to_array(img)    
    img_tensor = np.expand_dims(img_tensor, axis=0)      

    predictions = model.predict(img_tensor)

    return class_names[np.argmax(predictions[0])]

def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b: b[1][i], reverse=reverse))
    return cnts

def segment_characters(img):

    cipher_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # convert to grayscale and blur the image
    blur = cv2.GaussianBlur(cipher_image,(7,7),0)
    
    # Applied inversed thresh_binary 
    binary = cv2.threshold(blur, 180, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

    cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list which will be used to append charater image
    crop_characters = []

    # define standard width and height of character
    digit_w, digit_h = 100, 200

    contours_detected = sort_contours(cont)

    test_roi = img.copy()

    for i in range(0, len(contours_detected), 2):
        print(i)
        if len(contours_detected) == 1:
            (x, y, w, h) = cv2.boundingRect(contours_detected[i])
            cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)
            curr_num = binary[y:y+h,x:x+w]
            curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
            _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            crop_characters.append(curr_num)
        if i < len(contours_detected)-1:
            (x, y, w, h) = cv2.boundingRect(contours_detected[i])
            (x_next, y_next, w_next, h_next) = cv2.boundingRect(contours_detected[i+1])
            ratio = h/w
            if 1<=ratio<= 5: # Only select contour with defined ratio
                if (w*h < 0.2*w_next*h_next or w_next*h_next < 0.2*w*h) and (abs(y-y_next) <= 1.05*h) and abs(x-x_next)<=300:
                    new_x = min(x, x_next)
                    new_y = min(y, y_next)
                    new_w = max(w, w_next)
                    new_h = h + h_next
                    cv2.rectangle(test_roi, (new_x, new_y), (new_x+new_w, new_y+new_h), (0, 255,0), 2)
                    curr_num = cipher_image[new_y:new_y+new_h,new_x:new_x+new_w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    crop_characters.append(curr_num)
                else:
                    cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)
                    cv2.rectangle(test_roi, (x_next, y_next), (x_next + w_next, y_next + h_next), (0, 255,0), 2)
                    curr_num = cipher_image[y:y+h,x:x+w]
                    curr_num_next = cipher_image[y_next:y_next+h_next,x_next:x_next+w_next]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    curr_num_next = cv2.resize(curr_num_next, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    _, curr_num_next = cv2.threshold(curr_num_next, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    crop_characters.append(curr_num)
                    crop_characters.append(curr_num_next)
        else:
                (x, y, w, h) = cv2.boundingRect(contours_detected[i])
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)
                curr_num = cipher_image[y:y+h,x:x+w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)

    return crop_characters, test_roi

def translation(img):
    crop_characters, _ = segment_characters(img)
    for i in range(len(crop_characters)):
        letter = predict(crop_characters[i])
        crop_characters[i] = letter
    return crop_characters

code_image_path = './materials/dancing_men_symbol.jpeg'


with gr.Blocks(title = "Can you read me?") as demo:

    gr.Label("Can you read me?")
    with gr.Row():
        image_input = gr.Image()
        text_output = gr.Textbox()
    image_button = gr.Button("Submit")
    with gr.Accordion("Readme", open=True):
        gr.Markdown("This is a demonstration of a machine learning model that can read the Dancing men cipher. By uploading an image, you consent to its use for improving our model. Thanks :)")


    with gr.Accordion("The Dancing Men Cipher", open=False):
        gr.Markdown("Dancing Men Cipher on dCode.fr [online website], https://www.dcode.fr/dancing-men-cipher")
        alphabet = gr.Image(code_image_path, label="Image")

    image_button.click(translation, inputs=image_input, outputs=text_output)

demo.launch(share=True)