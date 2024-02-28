# Deep fake detection Django Application
## Requirements:

## Directory Structure

- ml_app -> Directory containing code in views.py file
- static -> Contains all css, js and json files
- templates -> Template files for HTML

<b>Note:</b> Before running the project make sure you have created directories namely <strong>models, uploaded_images, uploaded_videos</strong> in the project root and that you have proper permissions to access them.

# Running application locally on your machine

#### Step 1 : Clone the repo and Navigate to Django Application

#### Step 2: Create virtualenv (optional)

`python -m venv venv`

#### Step 3: Activate virtualenv (optional)

`venv\Scripts\activate`

#### Step 4: Install requirements

`pip install -r requirements.txt`

#### Step 5: Copy Models

`Copy your trained model to the models folder i.e Django Application/models/`

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