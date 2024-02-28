from django.shortcuts import render, redirect
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
from torch.autograd import Variable
import time
import sys
from torch import nn
import json
import glob
import copy
from torchvision import models
import shutil
from PIL import Image as pImage
import time
from django.conf import settings
from .forms import VideoUploadForm
from .forms import ImageUploadForm
from googleapiclient.discovery import build
from pytube import YouTube
from django.http import HttpResponse


index_template_name = 'index.html'
predict_template_name = 'predict.html'

im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
sm = nn.Softmax()
inv_normalize =  transforms.Normalize(mean=-1*np.divide(mean,std),std=np.divide([1,1,1],std))

train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])

class Model(nn.Module):

    def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained = True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048,num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


class validation_dataset(Dataset):
    def __init__(self,video_names,sequence_length=60,transform = None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self,idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100/self.count)
        first_frame = np.random.randint(0,a)
        for i,frame in enumerate(self.frame_extract(video_path)):
            #if(i % a == first_frame):
            faces = face_recognition.face_locations(frame)
            try:
              top,right,bottom,left = faces[0]
              frame = frame[top:bottom,left:right,:]
            except:
              pass
            frames.append(self.transform(frame))
            if(len(frames) == self.count):
                break
        """
        for i,frame in enumerate(self.frame_extract(video_path)):
            if(i % a == first_frame):
                frames.append(self.transform(frame))
        """        
        # if(len(frames)<self.count):
        #   for i in range(self.count-len(frames)):
        #         frames.append(self.transform(frame))
        #print("no of frames", self.count)
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)
    
    def frame_extract(self,path):
      vidObj = cv2.VideoCapture(path) 
      success = 1
      while success:
          success, image = vidObj.read()
          if success:
              yield image

def im_convert(tensor, video_file_name):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image.clip(0, 1)
    # This image is not used
    # cv2.imwrite(os.path.join(settings.PROJECT_DIR, 'uploaded_images', video_file_name+'_convert_2.png'),image*255)
    return image

def im_plot(tensor):
    image = tensor.cpu().numpy().transpose(1,2,0)
    b,g,r = cv2.split(image)
    image = cv2.merge((r,g,b))
    image = image*[0.22803, 0.22145, 0.216989] +  [0.43216, 0.394666, 0.37645]
    image = image*255.0
    plt.imshow(image.astype(int))
    plt.show()

def predict(model, img, path='./', video_file_name=""):
    fmap, logits = model(img.to('cpu'))  # Move the model to CPU
    img = im_convert(img[:, -1, :, :, :], video_file_name)
    params = list(model.parameters())
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    print('confidence of prediction:', logits[:, int(prediction.item())].item() * 100)
    return [int(prediction.item()), confidence]

def plot_heat_map(i, model, img, path='./', video_file_name=''):
    fmap, logits = model(img.to('cpu'))  # Move the model to CPU
    params = list(model.parameters())
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    idx = np.argmax(logits.detach().cpu().numpy())
    bz, nc, h, w = fmap.shape
    out = np.dot(fmap[i].detach().cpu().numpy().reshape((nc, h * w)).T, weight_softmax[idx, :].T)
    predict = out.reshape(h, w)
    predict = predict - np.min(predict)
    predict_img = predict / np.max(predict)
    predict_img = np.uint8(255 * predict_img)
    out = cv2.resize(predict_img, (im_size, im_size))
    heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
    img = im_convert(img[:, -1, :, :, :], video_file_name)
    result = heatmap * 0.5 + img * 0.8 * 255
    # Saving heatmap - Start
    heatmap_name = video_file_name + "_heatmap_" + str(i) + ".png"
    image_name = os.path.join(settings.PROJECT_DIR, 'uploaded_images', heatmap_name)
    cv2.imwrite(image_name, result)
    # Saving heatmap - End
    result1 = heatmap * 0.5 / 255 + img * 0.8
    r, g, b = cv2.split(result1)
    result1 = cv2.merge((r, g, b))
    return image_name

# Model Selection
def get_accurate_model(sequence_length):
    model_name = []
    sequence_model = []
    final_model = ""
    list_models = glob.glob(os.path.join(settings.PROJECT_DIR, "models", "*.pt"))
    for i in list_models:
        model_name.append(i.split("\\")[-1])
    for i in model_name:
        try:
            seq = i.split("_")[3]
            if (int(seq) == sequence_length):
                sequence_model.append(i)
        except:
            pass

    if len(sequence_model) > 1:
        accuracy = []
        for i in sequence_model:
            acc = i.split("_")[1]
            accuracy.append(acc)
        max_index = accuracy.index(max(accuracy))
        final_model = sequence_model[max_index]
    else:
        final_model = sequence_model[0]
    return final_model

ALLOWED_VIDEO_EXTENSIONS = set(['mp4','gif','webm','avi','3gp','wmv','flv','mkv'])

def allowed_video_file(filename):
    #print("filename" ,filename.rsplit('.',1)[1].lower())
    if (filename.rsplit('.',1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS):
        return True
    else: 
        return False
def index(request):
    if request.method == 'GET':
        video_upload_form = VideoUploadForm()
        if 'file_name' in request.session:
            del request.session['file_name']
        if 'preprocessed_images' in request.session:
            del request.session['preprocessed_images']
        if 'faces_cropped_images' in request.session:
            del request.session['faces_cropped_images']
        return render(request, index_template_name, {"form": video_upload_form})
    else:
        video_upload_form = VideoUploadForm(request.POST, request.FILES)
        if video_upload_form.is_valid():
            video_file = video_upload_form.cleaned_data['upload_video_file']
            video_file_ext = video_file.name.split('.')[-1]
            sequence_length = 20
            video_content_type = video_file.content_type.split('/')[0]
            if video_content_type in settings.CONTENT_TYPES:
                if video_file.size > int(settings.MAX_UPLOAD_SIZE):
                    video_upload_form.add_error("upload_video_file", "Maximum file size 100 MB")
                    return render(request, index_template_name, {"form": video_upload_form})

            if sequence_length <= 0:
                video_upload_form.add_error("sequence_length", "Sequence Length must be greater than 0")
                return render(request, index_template_name, {"form": video_upload_form})
            
            if allowed_video_file(video_file.name) == False:
                video_upload_form.add_error("upload_video_file","Only video files are allowed ")
                return render(request, index_template_name, {"form": video_upload_form})
            
            saved_video_file = 'uploaded_file_'+str(int(time.time()))+"."+video_file_ext
            if settings.DEBUG:
                with open(os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file), 'wb') as vFile:
                    shutil.copyfileobj(video_file, vFile)
                request.session['file_name'] = os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file)
            else:
                with open(os.path.join(settings.PROJECT_DIR, 'uploaded_videos','app','uploaded_videos', saved_video_file), 'wb') as vFile:
                    shutil.copyfileobj(video_file, vFile)
                request.session['file_name'] = os.path.join(settings.PROJECT_DIR, 'uploaded_videos','app','uploaded_videos', saved_video_file)
            request.session['sequence_length'] = sequence_length
            return redirect('ml_app:predict')
        else:
            return render(request, index_template_name, {"form": video_upload_form})
        
def predict_page(request):
    if request.method == "GET":
        if 'file_name' not in request.session:
            return redirect("ml_app:home")
        if 'file_name' in request.session:
            video_file = request.session['file_name']
        if 'sequence_length' in request.session:
            sequence_length = request.session['sequence_length']

        print(video_file)    
        path_to_videos = [video_file]
        print(path_to_videos)
        video_file_name = video_file.split('\\')[-1]
        print(video_file_name)
        if settings.DEBUG == False:
            production_video_name = video_file_name.split('/')[3:]
            production_video_name = '/'.join([str(elem) for elem in production_video_name])
            print("Production file name",production_video_name)
        video_file_name_only = video_file_name.split('.')[0]
        video_dataset = validation_dataset(path_to_videos, sequence_length=sequence_length, transform=train_transforms)
        model = Model(2).cpu()
        model_name = os.path.join(settings.PROJECT_DIR, 'models', get_accurate_model(sequence_length))
        models_location = os.path.join(settings.PROJECT_DIR, 'models')
        path_to_model = os.path.join(settings.PROJECT_DIR, model_name)
        model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
        model.eval()
        start_time = time.time()
        # Start: Displaying preprocessing images
        print("<=== | Started Videos Splitting | ===>")
        preprocessed_images = []
        faces_cropped_images = []
        cap = cv2.VideoCapture(video_file)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frames.append(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()

        for i in range(1, sequence_length + 1):
            frame = frames[i]
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = pImage.fromarray(image, 'RGB')
            image_name = video_file_name_only + "_preprocessed_" + str(i) + '.png'
            if settings.DEBUG:
                image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
            else:
                print("image_name", image_name)
                image_path = "/home/app/staticfiles" + image_name
            img.save(image_path)
            preprocessed_images.append(image_name)
        # ... (your existing code)
        try:
            heatmap_images = []
            for i in range(0, len(path_to_videos)):
                output = ""
                print("<=== | Started Predicition | ===>")
                prediction = predict(model, video_dataset[i], './', video_file_name_only)
                confidence = round(prediction[1], 1)
                print("<=== |  Predicition Done | ===>")
                # print("<=== | Heat map creation started | ===>")
                # for j in range(0, sequence_length):
                #     heatmap_images.append(plot_heat_map(j, model, video_dataset[i], './', video_file_name_only))
                if prediction[0] == 1:
                    output = "REAL"
                else:
                    output = "FAKE"
                print("Prediction : ", prediction[0], "==", output, "Confidence : ", confidence)
                print("--- %s seconds ---" % (time.time() - start_time))
            if settings.DEBUG:
                return render(request, predict_template_name, {'preprocessed_images': preprocessed_images,
                                                                'heatmap_images': heatmap_images,
                                                                "faces_cropped_images": faces_cropped_images,
                                                                "original_video": video_file_name,
                                                                "models_location": models_location,
                                                                "output": output, "confidence": confidence})
            else:
                return render(request, predict_template_name, {'preprocessed_images': preprocessed_images,
                                                                'heatmap_images': heatmap_images,
                                                                "faces_cropped_images": faces_cropped_images,
                                                                "original_video": production_video_name,
                                                                "models_location": models_location,
                                                                "output": output, "confidence": confidence})
        except:
            return render(request, 'cuda_full.html')

def about(request):
    return render(request, about_template_name)

def handler404(request,exception):
    return render(request, '404.html', status=404)
def cuda_full(request):
    return render(request, 'cuda_full.html')



# ... (your existing imports)

ALLOWED_IMAGE_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_image_file(filename):
    if filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS:
        return True
    else:
        return False

def image_index(request):
    if request.method == 'GET':
        image_upload_form = ImageUploadForm()
        return render(request, 'image_index.html', {"form": image_upload_form})
    else:
        image_upload_form = ImageUploadForm(request.POST, request.FILES)
        if image_upload_form.is_valid():
            print("success")
            image_file = image_upload_form.cleaned_data['upload_image_file']
            image_file_ext = image_file.name.split('.')[-1]
            if allowed_image_file(image_file.name) == False:
                image_upload_form.add_error("upload_image_file", "Only image files are allowed")
                return render(request, 'image_index.html', {"form": image_upload_form})
            
            saved_image_file = 'uploaded_image_' + str(int(time.time())) + "." + image_file_ext
            if settings.DEBUG:
                with open(os.path.join(settings.PROJECT_DIR, 'uploaded_images', saved_image_file), 'wb') as imgFile:
                    shutil.copyfileobj(image_file, imgFile)
                request.session['image_file'] = os.path.join(settings.PROJECT_DIR, 'uploaded_images', saved_image_file)
            else:
                with open(os.path.join(settings.PROJECT_DIR, 'uploaded_images', 'app', 'uploaded_images', saved_image_file), 'wb') as imgFile:
                    shutil.copyfileobj(image_file, imgFile)
                request.session['image_file'] = os.path.join(settings.PROJECT_DIR, 'uploaded_images', 'app', 'uploaded_images', saved_image_file)

            return redirect('ml_app:image_predict')
        else:
            return render(request, 'image_index.html', {"form": image_upload_form})

# ... (other imports)

def image_predict(request):
    if request.method == "GET":
        if 'image_file' not in request.session:
            return redirect("ml_app:image_index")
        
        image_file = request.session['image_file']
        image_file_name = os.path.basename(image_file)  # Updated line
        image_dataset = validation_dataset([image_file], sequence_length=20, transform=train_transforms)
        model = Model(2).cpu()
        model_name = os.path.join(settings.PROJECT_DIR, 'models', get_accurate_model(20))
        path_to_model = os.path.join(settings.PROJECT_DIR, model_name)
        model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
        model.eval()

        try:
            print("<=== | Started Image Prediction | ===>")
            prediction = predict(model, image_dataset[0], './', image_file_name)
            confidence = round(prediction[1], 1)
            print("<=== | Image Prediction Done | ===>")
            preprocessed_image = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_file_name)

            # Mapping numeric class label to class name
            class_names = ["REAL", "FAKE"]
            predicted_class = class_names[prediction[0]]

            print("Prediction : ", prediction[0], "==", predicted_class, "Confidence : ", confidence)

            return render(request, 'image_predict.html', {'preprocessed_image': preprocessed_image,
                                                           'output': predicted_class, 'confidence': confidence})
        except:
            return render(request, 'image_predict.html')


def analyze_video_youtube(request):
    api_key = 'AIzaSyBqr9Vdz9GnbcAioazFLGyj0JCV6897TsU'
    youtube = build('youtube', 'v3', developerKey=api_key)

    trending_videos_request = youtube.videos().list(
        part='snippet',
        chart='mostPopular',
        regionCode='IN',
        maxResults=1
    )

    try:
        response = trending_videos_request.execute()
        video_ids = [item['id'] for item in response.get('items', [])]

        if not video_ids:
            raise ValueError("No video IDs found in the response.")

        for video_id in video_ids:
            # Download videos using pytube
            yt = YouTube(f'https://www.youtube.com/watch?v={video_id}')
            stream = yt.streams.get_lowest_resolution()
            stream.download(output_path=os.path.join(settings.PROJECT_DIR, 'uploaded_videos'), filename=f'{video_id}.mp4')
            video_file_path = os.path.join(settings.PROJECT_DIR, 'uploaded_videos', f'{video_id}.mp4')

            # Call the predict_page_r function with the downloaded video file path
            return predict_page_r(request, video_file_path)

    except Exception as e:
        print('An error occurred:', e)
        return HttpResponse("Error occurred during video analysis.", status=500)

def predict_page_r(request, video_file_path):
    print(video_file_path)
    try:
        sequence_length = 20  # Adjust the default value as needed
        path_to_videos = [video_file_path]

        video_file_name = os.path.basename(video_file_path)
        video_file_name_only = os.path.splitext(video_file_name)[0]

        if not settings.DEBUG:
            production_video_name = '/'.join(video_file_name.split('/')[3:])
            print("Production file name", production_video_name)
        else:
            production_video_name = video_file_name

        video_dataset = validation_dataset(path_to_videos, sequence_length=sequence_length, transform=train_transforms)
        model = Model(2).cpu()
        model_name = os.path.join(settings.PROJECT_DIR, 'models', get_accurate_model(sequence_length))
        models_location = os.path.join(settings.PROJECT_DIR, 'models')
        path_to_model = os.path.join(settings.PROJECT_DIR, model_name)
        model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
        model.eval()
        start_time = time.time()

        preprocessed_images = []
        faces_cropped_images = []
        frames = []

        cap = cv2.VideoCapture(video_file_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        for i in range(sequence_length):
            if i < len(frames):
                frame = frames[i]
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = pImage.fromarray(image, 'RGB')
                image_name = f'{video_file_name_only}_preprocessed_{i}.png'
                image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name) if settings.DEBUG else f'/home/app/staticfiles/{image_name}'
                img.save(image_path)
                preprocessed_images.append(image_name)
            else:
                # Handle the case when there are not enough frames
                break

        heatmap_images = []
        for i in range(len(path_to_videos)):
            print("<=== | Started Prediction | ===>")
            prediction = predict(model, video_dataset[i], './', video_file_name_only)
            confidence = round(prediction[1], 1)
            print("<=== | Prediction Done | ===>")

            if prediction[0] == 1:
                output = "REAL"
            else:
                output = "FAKE"
            print("Prediction: ", prediction[0], "==", output, "Confidence: ", confidence)
            print("--- %s seconds ---" % (time.time() - start_time))

        if settings.DEBUG:
            return render(request, 'predict_template_name.html', {'preprocessed_images': preprocessed_images,
                                                                  'heatmap_images': heatmap_images,
                                                                  "faces_cropped_images": faces_cropped_images,
                                                                  "original_video": video_file_name,
                                                                  "models_location": models_location,
                                                                  "output": output, "confidence": confidence})
        else:
            return render(request, 'predict_template_name.html', {'preprocessed_images': preprocessed_images,
                                                                  'heatmap_images': heatmap_images,
                                                                  "faces_cropped_images": faces_cropped_images,
                                                                  "original_video": production_video_name,
                                                                  "models_location": models_location,
                                                                  "output": output, "confidence": confidence})

    except Exception as e:
        print("Exception:", e)
        return HttpResponse("Error occurred during video prediction.", status=500)