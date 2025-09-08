# Prototype Interface for Ocular Disease Classification Using Deep Learning
## Introduction
Visual impairments such as diabetic retinopathy, cataracts, and glaucoma affect millions worldwide, underscoring the need for early diagnosis solutions. In Colombia alone, approximately 4,000 per million inhabitants suffer irreversible blindness-causing eye diseases (primarily glaucoma, cataract, and diabetic retinopathy). This project addresses that problem by providing a prototype interface that uses deep learning to classify retinal fundus images into those key pathologies. The system leverages a pre-trained EfficientNetB3 convolutional neural network model to process and classify fundus photographs, distinguishing among Cataract, Diabetic Retinopathy, Glaucoma, or Normal retina conditions. This tool aims to aid in rapid screening of eye diseases from retinal images, a significant contribution in contexts with limited medical resources. 
This prototype was developed as a final thesis project at the Universidad de Investigación y Desarrollo (UDI) in 2024. Owing to its successful evaluation (final grade 4.6/5), the work was featured in the university’s scientific repository. 
It is important to note that the interface is a proof-of-concept prototype; a prominent notice on the UI warns users that the classifications are not 100% accurate and the tool should not be used for definitive medical diagnosis. Instead, it demonstrates the potential of deep learning in ophthalmic disease classification.
<img width="442" height="115" alt="e61d3ffa-5a01-46b1-bab5-57c3237c61dc" src="https://github.com/user-attachments/assets/70359052-18fe-4029-abda-c9c1f8c1494e" />

## System Requirements
Before using the application, ensure your system meets the following prerequisites:
* **Hardware:** At minimum, a dual-core 2.0 GHz CPU with 8 GB RAM and ~10 GB of free disk space is required. For optimal performance, a modern multi-core processor (e.g. Intel Core i5/i7 or AMD Ryzen 5/7), 16 GB RAM, and 20 GB (or more) of SSD storage are recommended. No dedicated GPU is strictly required for running a single prediction, but it can accelerate model loading and inference if available.
* **Operating System:** The prototype has been tested on Windows; Windows 10 is the minimum OS requirement, with Windows 11 recommended for best compatibility. (The interface could potentially run on Linux/Mac with similar setup, but the provided user guide instructions target Windows environments.)
* **Python Environment:** Python 3.11.4 must be installed on the system. This specific version is required because the deep learning model depends on TensorFlow 2.15.1, which has compatibility constraints with Python versions (Python 3.11 in this case). Ensure you have administrative rights to install packages and run commands.
* **Software Dependencies:** You will need to have pip (Python's package manager) available to install required libraries. Once Python 3.11.4 is set up, use pip to install the Jupyter environment and other dependencies. The essential packages include:
  * *Jupyter Notebook/Lab:* for running the interface notebook (install with pip install notebook or pip install jupyterlab).
  * *Voilá:* to serve the notebook as a web app (install with `pip install voila`).
  * *IPyWidgets:* for the interface's interactive widgets (install with `pip install ipywidgets`).
  * *Pillow:* for image processing (install with `pip install pillow`).
  * *NumPy:* for numerical array handling (install with `pip install numpy`).
  * *Matplotlib:* for image display in the notebook (install with `pip install matplotlib`).
  * *TensorFlow 2.15.1:* the deep learning framework to run the model (install with `pip install tensorflow==2.15.1`).

Ensure each of these installations completes without errors. If any installation fails (for example, pip not being recognized), double-check that Python and pip are correctly installed and added to your system PATH. The user manual provides troubleshooting tips – for instance, if `get-pip.py` fails, it suggests using an alternative curl command to download pip. After installing the above packages, you will have the necessary environment to run the retina classification interface.
## Installation and Setup of the Interface
With the environment prepared, follow these steps to set up the interface prototype:
1. Clone this GitHub repository to your local machine: The folder jupyter_interfaz already contains:
  - Interfaz.ipynb → the Jupyter Notebook with the user interface code.
  - modelo/efficientnet_retina_v1_40_learn_3.0_batch_32_rep_3.h5 → the pre-trained EfficientNetB3 model.
  - imagen_udi/Logo-udi-web.png → the UDI university logo used in the interface.

2. Create a Virtual Environment (recommended):
```
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```
3. Make sure you have Python 3.11.4 installed.
4. From inside the jupyter_interfaz folder, run:
```
voila Interfaz.ipynb
```
  This will open the interface in your default browse. The interface loads the pre-trained model into memory upon startup, which might take a few seconds. Once loaded, you can begin classifying images as described below.

## Interface Features and Layout

When the interface is running, the web page will display a simple graphical user interface with several elements designed to guide the user through the image classification process. The key features of the interface include:
* **UDI Logo:** The top of the interface shows the logo of the *Universidad de Investigación y Desarrollo (UDI)* as an image, giving credit to the institution where the project was developed.
* **Title:** Just below the logo, a title heading is displayed (in Spanish) as *"Clasificación de imágenes"*, indicating that this application is for image classification. This provides a clear identification of the interface’s purpose to the user.
* **Prototype Warning:** A prominent warning message is shown in a highlighted style (e.g. red or bold text). This message explains that the tool is *"solamente un prototipo"* – only a prototype – and that it is an approach to classifying ocular diseases that cause irreversible blindness. It cautions that the results will not be 100% accurate and should be interpreted with care. This is meant to manage user expectations and remind them that the system is for research/demo purposes, not a definitive medical device.
* **Image Upload Instructions:** The interface provides a brief instruction to the user, typically a label saying (in Spanish) to *"Por favor, sube una imagen en formato .jpg o .jpeg"* – "Please upload an image in .jpg or .jpeg format.". This message appears near the file upload control, informing the user of the required image format.
* **File Upload Button:** A button (or widget) labeled *"Upload"* allows the user to browse their device and select an image file to classify. The interface accepts standard JPEG images (with .jpg or .jpeg extensions). <ins>Only one image can be uploaded at a time</ins> (multiple file selection is disabled). Once a file is chosen, the upload begins automatically and the image will be sent to the model for processing.
* **Output Display Area:** Below the upload button, there is an output area where the interface will display the content relevant to the classification. Initially this area may be blank. After an image is uploaded, this area will show a rendered view of the uploaded fundus photograph (retina image) for confirmation
* **Progress Bar:** While the image is being processed by the model, a progress bar is shown to indicate that work is in progress. The progress bar starts from 0 and fills to 100% as the system loads the image, preprocesses it, runs the prediction, and prepares the output.
* **Prediction Result Display:** Once the model has finished analyzing the image, the interface presents the predicted classification result. This result is shown as a text message just below the image (and progress bar). It is typically styled in a noticeable color (the code uses a blue-colored message for the prediction) to draw attention. The message will read *"Predicción del modelo:"* followed by the name of the predicted class. For example, it might say *"Predicción del modelo: Catarata"* if the model thinks the image is of an eye with cataracts. The possible outputs correspond to the four categories the model was trained on: **Catarata (Cataract)**, **Retinopatía Diabética (Diabetic Retinopathy)**, **Glaucoma**, or **Normal**. If the model does not find a high confidence for any of these classes (which is rare, given it always outputs one of the four by design), the interface would display the result as *"Clase desconocida"* (Unknown class).
* **Error Message for Invalid Input:** If the user attempts to upload a file that is not a JPEG image, the interface will detect this and display an error message in red text instead of a result. The error message alerts the user that an invalid file was provided (for example, if the file is in PNG format or is not an image at all) and reminds them to upload a .jpg or .jpeg image.
<img width="1352" height="776" alt="9dd40890-bcea-4a9f-8793-05ccf4982789" src="https://github.com/user-attachments/assets/d0883fad-c381-4881-830b-3fcc79add933" />
<img width="892" height="777" alt="83ba26c2-2fbd-4a9e-b7bd-3c4ba253f02c" src="https://github.com/user-attachments/assets/b8666b6d-48fb-4f1b-9544-60e8010f8726" />
<img width="887" height="763" alt="aaa22887-e9da-4d9a-b6f4-b7ba5e890f63" src="https://github.com/user-attachments/assets/0a6d56cd-769a-4403-a331-513d06bacb0d" />
<img width="887" height="782" alt="d9cd7ce1-622f-4b4a-ab14-bac9bf6eeacb" src="https://github.com/user-attachments/assets/87c75be5-4660-4a9b-925a-afb362edfd4e" />

## Using the Interface for Classification

![ocular_interface_flow_en](https://github.com/user-attachments/assets/ee8af62c-710a-4f12-8ce0-6e2de057474f)

## Final notes on the interface
The described prototype provides a user-friendly way to leverage a deep learning model for classifying retinal images into key eye disease categories. By following the installation steps and usage guidelines above, a user (even without deep learning expertise) can load an image and obtain an instant prediction about the presence of Cataract, Diabetic Retinopathy, or Glaucoma.
Future improvements could involve expanding the training dataset, improving the model’s accuracy, and adding explainability to the predictions (such as heatmaps showing which part of the retina led to the classification).


## Model Development & Experiments
The project team conducted a structured, three-cycle experimentation process to select the most suitable convolutional neural network (CNN) for ocular disease classification.
### Methodology
 * **Dataset:** “Ocular Disease Recognition Dataset” compiled by Guna Venkat Doddi (2022), ~1,000 fundus images per class.
 * **Evaluation metrics:** accuracy, validation loss, per-class precision/recall, F1-score, and confusion matrices.
 * **Regularization:** early stopping and adaptive learning-rate reduction (“patience”).
 * **Hyperparameters:** tuned per cycle (batch size, epochs, learning rates).
 * **Reproducibility:** each configuration was repeated multiple times to account for variance.

Two pipeline diagrams (dataset processing and model training/validation) summarize the workflow.
<img width="2142" height="3642" alt="image" src="https://github.com/user-attachments/assets/55139fd0-6e4c-4afe-94dd-5634d6efd7c2" />

<img width="1725" height="5708" alt="image" src="https://github.com/user-attachments/assets/1722c448-a098-430d-af77-a9b7512e7c9c" />

#### Cycle 1 - Preliminary Screening
Five CNN architectures were compared: **VGG19, ResNet50, EfficientNetB3, Xception, and InceptionV3**.
**Setup:** 40 epochs, batch size 16, 10 repetitions per learning rate (0.01, 0.001, 0.0001).
**Findings:**
* EfficientNetB3 achieved the highest mean accuracy (96.9%).
* ResNet50 followed with 95.6%, while VGG19 averaged 94%.
* Xception (<80%) and InceptionV3 (~74%) showed poor stability and were excluded from subsequent cycles

#### Cycle 2 – Focused Comparison
The three best candidates from Cycle 1 (VGG19, ResNet50, EfficientNetB3) were retrained with all four classes (Normal, Cataract, Glaucoma, Diabetic Retinopathy).
**Setup:** 60 epochs, batch size 16, five repetitions per model. Learning rates: 1e-4 (VGG19), 1e-3 (ResNet50, EfficientNet).

**Findings:**

* VGG19 reached 93.7% validation accuracy but showed higher variance.
* ResNet50 achieved 95.2% with low variance but occasional instability in loss.
* EfficientNetB3 delivered the best balance, with 95% global accuracy, strong per-class F1-scores, and robustness across runs

#### Cycle 3 – Final Optimization

EfficientNetB3, selected as the final model, was fully optimized.

**Setup:** 100 epochs, batch size 32, learning rate 1e-3, five repetitions.

**Findings:**
* Achieved 96.1% validation accuracy, with validation loss reduced to 0.2188.
* Global accuracy improved from 95% in Cycle 2 to up to 99% after optimization.
* Demonstrated superior generalization across all four classes.

#### Summary of Results
| Model              | Cycle | Accuracy | Strengths                                                      | Limitations                              |
| ------------------ | ----- | -------- | -------------------------------------------------------------- | ---------------------------------------- |
| **EfficientNetB3** | 1-3   | 96.9-99% | Best overall performance, efficient, balanced precision/recall | Slightly higher loss variance in Cycle 2 |
| ResNet50           | 1-2   | \~95%    | Stable gradients via residual blocks                           | Lower recall in Glaucoma class           |
| VGG19              | 1-2   | \~94%    | Simple, adaptable                                              | Higher variance, less consistent         |
| Xception           | 1     | <80%     | Depthwise convolutions                                         | Unstable accuracy/loss                   |
| InceptionV3        | 1     | \~74%    | Multi-scale convolutions                                       | Underperformed relative to others        |

#### Important note

The .ipynb files used during this experimentation phase are in the ‘colabs used for experiments’ folder. Each block of code is documented (although only in Spanish) and contains the outputs of the graphs used for the analysis of results. Their nomenclature is:
`retina_{model}_sequential_v{cycle}`
