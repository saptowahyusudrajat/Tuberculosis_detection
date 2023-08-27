import streamlit as st
st.set_page_config(page_title="Bacteria Detection", layout="wide")
import warnings
warnings.filterwarnings("ignore")
from streamlit_option_menu import option_menu
from PIL import Image
from pathlib import Path
from streamlit_extras import word_importances
import cv2
import tempfile
import torch

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

selected = option_menu(
    "Bacteria Detection",
    ["Home", "Detect", "About"],
    icons=["house", "play-fill", "info-circle"],
    menu_icon="window-dock",
    default_index=0,
    orientation="horizontal",
)
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


path_yolo = Path(__file__).parent 
path_model = Path(__file__).parent / 'model/best_BTA.pt'
# image_directory = os.path.join(script_directory, "image", "train")
# val_directory = os.path.join(script_directory, "image", "train")

container = [st.container()]
container2 = [st.container()]
colArray1 = [2, 5]
colArray2 = [5, 5, 5, 5]
colArray3 = [5, 5]
colArray4 = [1, 2, 1]
col = [0, 1, 0]
ctsingle = container[0]
ctleft, ctmid, ctright = container2[0].columns(colArray4)
ctleft2,ctright2 =container2[0].columns(colArray1)



def detectx(frame, model):
    frame = [frame]
    info_placeholder = st.empty()  # Create an empty placeholder
    info_placeholder.text("[INFO] Detecting Bacteria. . . ") 
    results = model(frame)
    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    info_placeholder.empty()  # Clear the waiting text
    return labels, cordinates


def plot_boxes(results, frame, classes,threshold):
    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels
    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    info_placeholder1 = st.empty()  # Create an empty placeholder
    info_placeholder2 = st.empty()  # Create another empty placeholder
    info_placeholder1.text(f"[INFO] Total {n} detections. . .")  # Display waiting text
    info_placeholder2.text("[INFO] Looping through all detections. . . ")  # Display waiting text

    # print(f"[INFO] Total {n} detections. . .")
    # print(f"[INFO] Looping through all detections. . . ")

    ### looping through the detections
    count_b = 0
    for i in range(n):
        row = cord[i]
        if (
            row[4] >= threshold
        ):  ### threshold value for detection. We are discarding everything below this value
            print(f"[INFO] Extracting BBox coordinates. . . ")
            count_b += 1
            x1, y1, x2, y2 = (
                int(row[0] * x_shape),
                int(row[1] * y_shape),
                int(row[2] * x_shape),
                int(row[3] * y_shape),
            )  ## BBOx coordniates
            text_d = classes[int(labels[i])]

            if text_d == r"BTA":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  ## BBox
                cv2.rectangle(
                    frame, (x1, y1 - 20), (x2, y1), (0, 255, 0), -1
                )  ## for text label background
                cv2.putText(
                    frame,
                    text_d,
                    (x1, y1-15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 0, 0),
                    1,
                )
                threshold_text = f"{row[4]:.2f}"
                cv2.putText(
                    frame,
                    threshold_text,
                    (x1, y1),  # Adjust the position as needed
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.2,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
    info_placeholder1.empty()  # Clear the waiting text
    info_placeholder2.empty()  # Clear the waiting text
    return frame, count_b


def detect():
    ctsingle.title("Bacteria detection dashboard @Achmad Agus")
    with ctleft2:
        threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.00, step=0.01)
    # Upload gambar dari pengguna

    uploaded_image = ctsingle.file_uploader("Upload Images...", type=["jpg", "png", "jpeg"])

    model = torch.hub.load(
        path_yolo,
        "custom",
        source="local",
        path=path_model,
        force_reload=True,
    )

    if uploaded_image is not None:
        # Baca gambar menggunakan OpenCVv
        image_path = uploaded_image.read()
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(image_path)
        temp_file.close()

        frame = cv2.imread(temp_file.name)

        results = detectx(frame, model=model)
        classes = model.names

        frame,count_b = plot_boxes(results, frame, classes,threshold)
        
        with ctright2:
            print(frame.shape)
            frame = cv2.putText(
                frame,
                f"Number of Bacteria:",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            count_b_text = str(count_b)

            kategori = "Negative"
            if count_b>=1 and count_b <=9:
                kategori = "Scanty"
            elif count_b>=10 and count_b <=99:
                kategori = "1+"
            elif count_b>=100 and count_b <1000:
                kategori = "2+"
            elif count_b>=1000:
                kategori = "3+"
            
            frame = cv2.putText(
                frame,
                count_b_text,
                (10, 80),  # Adjust the position as needed
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

            frame = cv2.putText(
                frame,
                f"Skala IUATLD: {kategori}",
                (10, 120),  # Adjust the position as needed
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

            st.image(frame, caption="Detection Result", channels="BGR")
        # ctleft2.write(count_b)
        #st.write(count_b)
        # # Konversi dari BGR ke RGB (untuk menyesuaikan format warna)
        # # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # # Tampilkan gambar hasil menggunakan Streamlit
        # st.image(image, caption="Hasil Deteksi Objek", use_column_width=True)

# from ultralytics import YOLO
# import os
# import yaml


# # Get the directory path of the Python script
# script_directory_raw = os.path.dirname(os.path.abspath(__file__))
# subdirectory = 'source'
# script_directory=os.path.join(script_directory_raw,subdirectory)
# image_directory = os.path.join(script_directory, "image", "train")
# val_directory = os.path.join(script_directory, "image", "train")

# # Get the directory path of the YAML file
# yaml_content = {
#     "path": script_directory,
#     "train": image_directory,
#     "val": val_directory,
#     "name": {
#         "0": "Bacteria"
#     }
# }
# # # Create a dictionary with the directory path
# # data = {"yaml_directory": yaml_directory}

# # Write the dictionary to a YAML file
# with open("config.yaml", "w") as yaml_file:
#     yaml.dump(yaml_content, yaml_file)




if selected == "Home":
    ctmid.subheader("Welcome")
    ctmid.write(
        f'<hr style="background-color: {"#803df5"}; margin-top: 0;'
        ' margin-bottom: 0; height: 3px; border: none; border-radius: 3px;">',
        unsafe_allow_html=True,
    )
    text = """&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This application is designed to detect bacteria of an image using
                 |Computer Vision|. It uses a latest computer vision model that specify to detect bacteria of microscopic images with mere sample of trained images. Please upload the microscopic
                 images of an area to start the process |This detector runs on static quality.
                 """
    words = text.split("|")
    desc = word_importances.format_word_importances(
        words,
        importances=(0, 0.5, 0, 0.5),  # fmt: skip
    )
    ctmid.write(desc, unsafe_allow_html=True)
    # image = Image.open(Path(__file__).parent / "img/Yolov5.png")
    # ctmid.image(image, use_column_width="auto")


if selected == "Detect":
    ##### put code here for image detection
    detect()


if selected == "About":
    ctmid.subheader("About")
    ctmid.write(
        f'<hr style="background-color: {"#803df5"}; margin-top: 0;'
        ' margin-bottom: 0; height: 3px; border: none; border-radius: 3px;">',
        unsafe_allow_html=True,
    )
    text = """&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This application may be optimize to make it more dynamic based on tresholded qualities.
                 Overall, any error that may produced by this detector |could be refine with more trained images| and |uniforming preprocessing images| for getting better result of detection.
                 """
    words = text.split("|")
    desc = word_importances.format_word_importances(
        words,
        importances=(0, 0.5,0,0.5,0),  # fmt: skip
    )
    ctmid.write(desc, unsafe_allow_html=True)
