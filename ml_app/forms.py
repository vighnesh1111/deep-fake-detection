from django import forms

class VideoUploadForm(forms.Form):
    upload_video_file = forms.FileField(label="", required=True,widget=forms.FileInput(attrs={"accept": "video/*"}))

class ImageUploadForm(forms.Form):
    upload_image_file = forms.ImageField(
        label='',
        help_text='Allowed formats: png, jpg, jpeg, gif'
    )
