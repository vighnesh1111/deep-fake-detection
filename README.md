# Techrunners_Agnethon

# Deep fake detection Django Application
## Requirements:

You can find the list of requirements in [requirements.txt](https://github.com/RahulNair15/Techrunners_Agnethon/blob/main/requirements.txt). Main requirements are listed below:

```
Python >= v3.8
Django >= v3.0
```

## Directory Structure

- ml_app -> Directory containing code in views.py file
- static -> Contains all css, js and json files (for face-api)
- templates -> Template files for HTML

<b>Note:</b> Before running the project make sure you have created directories namely <strong>models, uploaded_images, uploaded_videos</strong> in the project root and that you have proper permissions to access them.

# Running application locally on your machine

### Prerequisite
1. Copy your trained model to the models folder.
   - You can download our trained models from the [Google Drive](https://drive.google.com/drive/folders/1UX8jXUXyEjhLLZ38tcgOwGsZ6XFSLDJ-?usp=sharing) or you can train your models using the steps mentioned in Model Creation directory.

#### Step 1 : Clone the repo and Navigate to Django Application

`git clone https://github.com/RahulNair15/Techrunners_Agnethon.git`

#### Step 2: Create virtualenv (optional)

`python -m venv venv`

#### Step 3: Activate virtualenv (optional)

`venv\Scripts\activate`

#### Step 4: Install requirements

`pip install -r requirements.txt`

#### Step 5: Copy Models

`Copy your trained model to the models folder i.e Django Application/models/`

- You can download our trained models from [Google Drive](https://drive.google.com/file/d/1D_jPqYe6mE6Ae-lpUwxYoQ4bZ_nTQhFI/view?usp=sharing)

**Note :** The model name must be in specified format only i.e *model_84_acc_20_frames_final_data.pt*. Make sure that no of frames must be mentioned after certain 3 underscores `_` , in the above example the model is for 20 frames.

#  Step 5:Deepfake Image Detection Setup

This project focuses on detecting Deepfake images. Follow the steps below to set up and run the project.

## Installation
#Note:Ensure that Jupyter Notebook is installed on your system.
5.1. Navigate to the `deepfake-Image-detection` folder.

5.2. Run the `deepfake-detection.bat` script.

5.3.Wait for the installation process to complete. When prompted, enter 'y' to continue.

5.4. Jupyter Notebook will automatically open in your default web browser.

5.5. In Jupyter Notebook, navigate to the 'Run' menu and select 'Run All Cells.'

5.6. Wait for the execution of all cells to complete.

After these steps, the Deepfake image detection will be ready for use.

Note:If this process fails, then deepfake-Image-detection will only not work rest feature will work fine.



### Step 6: Run project

`python manage.py runserver`

## Demo 
### You can watch the [youtube video](https://youtu.be/ZvVPkyUvrjk) for demo of the project.

