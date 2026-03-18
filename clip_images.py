import sys
import os
import io
import pickle
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import torch

#parameters
path_dataset = "Dataset/100people/images/" 
path_output = "Extracted_Features/clip_features.npy"
transform_ops = ["jpeg", "resize", "crop", "webp", "rotation", "contrast", "brightness", "blur", "social", "greyscale", "hsv", "cmyk", "median", "printscan", "twitter", "whitenoise", "perlinnoise"]
apply_transform = None
transform_params = None #leave None for randomization
generators = ["bing", "firefly", "flux-dev", "freepik", "imagen", "leonardoai", "midjourney", "nightcafe", "stabilityai", "starryai"]
path_termination = {"bing": ["-chrome-0.png", "-firefox-0.png", "-chrome-0.jpg", "-firefox-0.jpg", "chrome-0.png", "-firefox-0.jpeg"], "firefly": "-0.png", "flux-dev": ".png", "freepik": ".png", "imagen": ".jpg", "leonardoai": ".png", "midjourney": ".png", "nightcafe": ".jpg", "stabilityai": ".png", "starryai": ".png"}
num_people = 100
image_size = (1024,1024)
input_size = (336,336)

#function to randomize transform_params
def RandomParams(apply_transform):
    if apply_transform == "jpeg": #params: QF in [50,100]
        return {"QF": np.random.randint(50,100)}
    elif apply_transform == "resize": #params: scale in [0.4,2.0]
        return {"scale": 0.4 + 1.6*np.random.random()}
    elif apply_transform == "crop": #params: cover in [0.5, 0.9]
        return {"cover": 0.5 + 0.4*np.random.random()}
    elif apply_transform == "webp": #params: QF in [50,100]
        return {"QF": np.random.randint(50,100)}
    elif apply_transform == "rotation": #params: angle in [-5°,+5°]
        return {"angle": -5.0 + 10.0 * np.random.random()}
    elif apply_transform == "contrast": #params: multiplier in [1.2, 2.4]
        return {"multiplier": 1.2 + 1.2 * np.random.random()}
    elif apply_transform == "brightness": #params: multiplier in [1.2, 2.4]
        return {"multiplier": 1.2 + 1.2 * np.random.random()}
    elif apply_transform == "blur": #params: empty
        return {}
    elif apply_transform == "social": #params: social in ["instagram"]
        return {"social": "instagram"}
    elif apply_transform == "greyscale": #params: empty
        return {}
    elif apply_transform == "hsv": #params: empty
        return {}
    elif apply_transform == "cmyk": #params: empty
        return {}
    elif apply_transform == "median": #params: size in [3,5,7]
        return {"size": np.random.choice([3,5,7])}
    elif apply_transform == "printscan": #params: angle in [-5°,+5°], scale in [0.8, 1.2], noise in [8, 32]
        return {"angle": -5.0 + 10.0 * np.random.random(), "scale": 0.8 + 0.4*np.random.random(), "noise": 8.0 + 24.0*np.random.random()}
    elif apply_transform == "twitter": #params: empty
        return {}
    elif apply_transform == "whitenoise": #params: noise in [16, 64]
        return {"noise": 16.0 + 48.0*np.random.random()}
    elif apply_transform == "perlinnoise": #params: width in [4,6,8,10,12], noise in [8, 32]
        return {"width": np.random.choice([4,6,8,10,12]), "noise": 8.0 + 24.0*np.random.random()}
    return None

#function to apply a transform to an image
def Transform(image, apply_transform, transform_params):
    if apply_transform == "jpeg": #params: QF in [50,100]
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=transform_params["QF"]) 
        image.close()
        return Image.open(buffer)
    elif apply_transform == "resize": #params: scale in [0.4,2.0]
        return image.resize((int(transform_params["scale"]*image.size[0]), int(transform_params["scale"]*image.size[1])))
    elif apply_transform == "crop": #params: cover in [0.5, 0.9]
        left = image.size[0]*(1 - transform_params["cover"])/2
        right = left + image_size[0]*transform_params["cover"]
        top = image.size[1]*(1 - transform_params["cover"])/2
        bottom = top + image_size[1]*transform_params["cover"]
        return image.crop([left,top,right,bottom])
    elif apply_transform == "webp": #params: QF in [50,100]
        buffer = io.BytesIO()
        image.save(buffer, format="WEBP", quality=transform_params["QF"]) 
        image.close()
        return Image.open(buffer)
    elif apply_transform == "rotation": #params: angle in [-5°,+5°]
        return image.rotate(transform_params["angle"])
    elif apply_transform == "contrast": #params: multiplier in [1.2, 2.4]
        return ImageEnhance.Contrast(image).enhance(transform_params["multiplier"]) 
    elif apply_transform == "brightness": #params: multiplier in [1.2, 2.4]
        return ImageEnhance.Brightness(image).enhance(transform_params["multiplier"])
    elif apply_transform == "blur": #params: empty
        return image.filter(ImageFilter.BLUR)
    elif apply_transform == "social": #params: social in ["instagram"]
        if transform_params["social"] == "instagram":
            image.thumbnail((1080, 1080), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            buffer.seek(0)
            return Image.open(buffer)
        else:
            sys.exit("Error! Unimplemented social network : "+transform_params["social"])
    elif apply_transform == "greyscale": #params: empty
        return image.convert("L").convert("RGB")
    elif apply_transform == "hsv": #params: empty
        return image.convert("HSV").convert("RGB")
    elif apply_transform == "cmyk": #params: empty
        return image.convert("CMYK").convert("RGB")
    elif apply_transform == "median": #params: size in [3,5,7]
        return image.filter(ImageFilter.MedianFilter(transform_params["size"]))
    elif apply_transform == "printscan": #params: angle in [-5°,+5°], scale in [0.8, 1.2], noise in [8, 32]
        img_np = np.asarray(image.scale(transform_params["scale"]).rotate(transform_params["angle"]))
        image.close()
        img_np = img_np + np.random.normal(0.0, transform_params["noise"], img_np.shape)
        image = Image.fromarray(img_np)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image.close()
        return Image.open(buffer)
    elif apply_transform == "twitter": #params: empty
        image.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        image.close()
        return Image.open(buffer)
    elif apply_transform == "whitenoise": #params: noise in [16, 64]
        img_np = np.asarray(image)
        image.close()
        img_np = img_np + np.random.normal(0.0, transform_params["noise"], img_np.shape)
        image = Image.fromarray(img_np)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image.close()
        return Image.open(buffer)
    elif apply_transform == "perlinnoise": #params: width in [4,6,8,10,12], noise in [8, 32]
        img_np = np.asarray(image)
        image.close()
        noise_np = perlin_numpy.generate_perlin_noise_2d(img_np.shape, (transform_params["width"], transform_params["width"]))*transform_params["noise"]
        img_np = img_np + np.stack((noise_np,noise_np,noise_np), axis=2)
        image = Image.fromarray(img_np)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image.close()
        return Image.open(buffer)
    return None

#check if apply_transform is feasible
if apply_transform is not None:
    if apply_transform not in transform_ops:
        sys.exit("Error! Unknown transform : "+apply_transform)

#build clip model
print("Building CLIP model from pre-trained weights")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336") #"openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336") #"openai/clip-vit-base-patch32"

#scan and process images
x_list = []
for i in range(len(generators)):
    y_list = []
    for j in range(num_people):
        z_list = []
        for k in range(len(generators)):
            print("Processing generator "+str(i+1)+" of "+str(len(generators))+", image "+str(j+1)+" of "+str(num_people)+", re-generator "+str(k+1)+" of "+str(len(generators))+"                       ", end="\r")
            img = None
            #build image path
            img_path = path_dataset+generators[i]+"/images/"+generators[k]+"/"
            if j+1<10: img_path = img_path + "0000" + str(j+1)
            elif j+1<100:  img_path = img_path + "000" + str(j+1)
            else: img_path = img_path + "00" + str(j+1)
            if type(path_termination[generators[k]]) is list:
                img_path = [img_path + pt for pt in path_termination[generators[k]]]
            else:
                img_path = img_path + path_termination[generators[k]]
            #open and preprocess image
            if type(img_path) is list:
                retry = True
                tries = -1  
                while tries<len(img_path) and retry:
                    retry = False
                    tries = tries + 1
                    try: img = Image.open(img_path[tries])
                    except: retry=True
            else:
                img = Image.open(img_path)
            #apply transform if specified
            if apply_transform is not None:
                if transform_params is None:
                    transform_params = RandomParams(apply_transform)
                img = Transform(img, apply_transform, transform_params)
            #preprocess image
            inputs = processor(images=img, return_tensors="pt", padding=True)
            img.close()
            #extract features
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            #normalize features
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            #convert features to numpy array
            image_features_np = image_features.cpu().numpy()[0]
            #append features to list
            z_list.append(image_features_np)
        y_list.append(z_list)
    x_list.append(y_list)
print("")

#numpyfy and save list of lists of lists of feature vectors
print("Saving extracted features")
data = np.array(x_list)
out_file = open(path_output, 'wb')
np.save(out_file, data, allow_pickle=False)
out_file.close()

#terminate execution
print("Execution Terminated Succesfully!")