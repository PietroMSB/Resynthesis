import sys
import os
import io
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import sklearn.metrics
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

path_dataset = "Dataset/100people/images/"
generators = ["bing", "firefly", "flux-dev", "freepik", "imagen", "leonardoai", "midjourney", "nightcafe", "stabilityai", "starryai"]
path_termination = {"bing": ["-chrome-0.png", "-firefox-0.png", "-chrome-0.jpg", "-firefox-0.jpg", "chrome-0.png", "-firefox-0.jpeg"], "firefly": ["-0.png", "-0.jpg"], "flux-dev": ".png", "freepik": ".png", "imagen": ".jpg", "leonardoai": ".png", "midjourney": ".png", "nightcafe": ".jpg", "stabilityai": ".png", "starryai": ".png"}
fs_generator_list = ["bing", "firefly", "flux-dev", "freepik", "imagen", "leonardoai", "midjourney", "nightcafe", "stabilityai", "starryai", "AuraFlow", "Pixart", "Playground", "Tencent_Hunyuan"]
transform_ops = ["jpeg", "resize", "crop", "webp", "rotation", "contrast", "brightness", "blur", "social", "greyscale", "hsv", "cmyk", "median", "printscan", "twitter", "whitenoise"]
fs_repeats = 5
fs_seeds = {1: {5: [32939, 430725, 732722, 856813, 942271], 6: [32939, 430725, 732722, 856813, 942271], 7: [32939, 430725, 732722, 856813, 942271], 8: [32939, 430725, 732722, 856813, 942271], 9: [32939, 430725, 732722, 856813, 942271], 10: [32939, 430725, 732722, 856813, 942271], 11: [32939, 430725, 732722, 856813, 942271], 12: [32939, 430725, 732722, 856813, 942271], 13: [32939, 430725, 732722, 856813, 942271], 14: [32939, 430725, 732722, 856813, 942271]}, 2: {5: [32939, 430725, 732722, 856813, 942271], 6: [109695, 116634, 259106, 528000, 681821], 7: [25330, 98259, 626445, 668965, 776089], 8: [16370, 191494, 274330, 318795, 871478], 9: [1070, 63860, 70609, 109210, 718082], 10: [299699, 363544, 424717, 610295, 970638], 11: [982, 196150, 418120, 643513, 927945], 12: [8020, 426747, 493049, 521546, 751974], 13: [25598, 205657, 342462, 565424, 862572], 14: [102598, 266661, 276017, 448074, 753629]}, 3: {5: [32939, 430725, 732722, 856813, 942271], 6: [25330, 98259, 626445, 668965, 776089], 7: [1070, 63860, 70609, 109210, 718082], 8: [982, 196150, 418120, 643513, 927945], 9: [25598, 205657, 342462, 565424, 862572], 10: [55106, 65672, 104396, 323365, 568283], 11: [187926, 332692, 356351, 378817, 886672], 12: [454529, 583547, 836517, 912555, 963343], 13: [160216, 262543, 399184, 550467, 898867], 14: [150900, 326990, 566201, 640056, 664281]}, 4: {5: [32939, 430725, 732722, 856813, 942271], 6: [16370, 191494, 274330, 318795, 871478], 7: [982, 196150, 418120, 643513, 927945], 8: [102598, 266661, 276017, 448074, 753629], 9: [187926, 332692, 356351, 378817, 886672], 10: [343464, 548243, 646002, 652264, 952253], 11: [150900, 326990, 566201, 640056, 664281], 12: [371702, 562945, 597389, 928596, 972826], 13: [279278, 668537, 750657, 836126, 938073], 14: [289480, 420935, 584997, 766763, 893076]}, 5: {5: [32939, 430725, 732722, 856813, 942271], 6: [1070, 63860, 70609, 109210, 718082], 7: [25598, 205657, 342462, 565424, 862572], 8: [187926, 332692, 356351, 378817, 886672], 9: [160216, 262543, 399184, 550467, 898867], 10: [45209, 69756, 86741, 414017, 735424], 11: [279278, 668537, 750657, 836126, 938073], 12: [118494, 179970, 474289, 738916, 990150], 13: [103170, 189574, 196460, 395793, 781155], 14: [214067, 537867, 671089, 687567, 799756]}, 6: {5: [32939, 430725, 732722, 856813, 942271], 6: [299699, 363544, 424717, 610295, 970638], 7: [55106, 65672, 104396, 323365, 568283], 8: [343464, 548243, 646002, 652264, 952253], 9: [45209, 69756, 86741, 414017, 735424], 10: [133633, 600756, 605370, 945845, 975225], 11: [147352, 217998, 320223, 487523, 678893], 12: [71028, 372439, 527668, 575603, 662094], 13: [296638, 505026, 520105, 702934, 718037], 14: [99326, 228013, 229999, 532371, 657138]}, 7: {5: [32939, 430725, 732722, 856813, 942271], 6: [982, 196150, 418120, 643513, 927945], 7: [187926, 332692, 356351, 378817, 886672], 8: [150900, 326990, 566201, 640056, 664281], 9: [279278, 668537, 750657, 836126, 938073], 10: [147352, 217998, 320223, 487523, 678893], 11: [214067, 537867, 671089, 687567, 799756], 12: [7268, 33052, 57385, 456377, 833366], 13: [84418, 335073, 850369, 952290, 988872], 14: [281786, 675221, 725580, 853763, 881493]}, 8: {5: [32939, 430725, 732722, 856813, 942271], 6: [8020, 426747, 493049, 521546, 751974], 7: [454529, 583547, 836517, 912555, 963343], 8: [371702, 562945, 597389, 928596, 972826], 9: [118494, 179970, 474289, 738916, 990150], 10: [71028, 372439, 527668, 575603, 662094], 11: [7268, 33052, 57385, 456377, 833366], 12: [163260, 192203, 325317, 568030, 778765], 13: [214517, 307630, 422765, 423226, 932818], 14: [110062, 750489, 802640, 849190, 996444]}, 9: {5: [32939, 430725, 732722, 856813, 942271], 6: [25598, 205657, 342462, 565424, 862572], 7: [160216, 262543, 399184, 550467, 898867], 8: [279278, 668537, 750657, 836126, 938073], 9: [103170, 189574, 196460, 395793, 781155], 10: [296638, 505026, 520105, 702934, 718037], 11: [84418, 335073, 850369, 952290, 988872], 12: [214517, 307630, 422765, 423226, 932818], 13: [253664, 452389, 702632, 772899, 950440], 14: [386780, 408789, 590152, 741063, 907549]}, 10: {5: [32939, 430725, 732722, 856813, 942271], 6: [102598, 266661, 276017, 448074, 753629], 7: [150900, 326990, 566201, 640056, 664281], 8: [289480, 420935, 584997, 766763, 893076], 9: [214067, 537867, 671089, 687567, 799756], 10: [99326, 228013, 229999, 532371, 657138], 11: [281786, 675221, 725580, 853763, 881493], 12: [110062, 750489, 802640, 849190, 996444], 13: [386780, 408789, 590152, 741063, 907549], 14: [123553, 581448, 614931, 823154, 864669]}, 11: {5: [32939, 430725, 732722, 856813, 942271], 6: [55106, 65672, 104396, 323365, 568283], 7: [45209, 69756, 86741, 414017, 735424], 8: [147352, 217998, 320223, 487523, 678893], 9: [296638, 505026, 520105, 702934, 718037], 10: [27031, 29926, 114024, 569068, 742656], 11: [145214, 187129, 378606, 685855, 880347], 12: [387850, 428060, 531373, 680416, 680761], 13: [202791, 257061, 511887, 839931, 991294], 14: [13581, 508829, 588358, 824950, 949395]}, 12: {5: [32939, 430725, 732722, 856813, 942271], 6: [46898, 147219, 754270, 770670, 923625], 7: [99000, 187838, 373084, 746536, 845171], 8: [377935, 524772, 753217, 849730, 922556], 9: [26535, 28492, 175827, 332501, 825845], 10: [12744, 39056, 223480, 555975, 584128], 11: [32691, 71461, 576504, 993991, 995901], 12: [323665, 638989, 641165, 742330, 887882], 13: [869, 60881, 171532, 384651, 747170], 14: [166579, 560725, 626010, 908351, 968030]}, 13: {5: [32939, 430725, 732722, 856813, 942271], 6: [187926, 332692, 356351, 378817, 886672], 7: [279278, 668537, 750657, 836126, 938073], 8: [214067, 537867, 671089, 687567, 799756], 9: [84418, 335073, 850369, 952290, 988872], 10: [145214, 187129, 378606, 685855, 880347], 11: [386780, 408789, 590152, 741063, 907549], 12: [7914, 95693, 624777, 686522, 974425], 13: [336815, 368594, 468447, 826710, 849241], 14: [21650, 185603, 357072, 652217, 769436]}, 14: {5: [32939, 430725, 732722, 856813, 942271], 6: [162905, 411785, 493067, 599782, 750171], 7: [168920, 238814, 409928, 418704, 561244], 8: [526192, 586474, 779481, 863108, 902087], 9: [68593, 193215, 294187, 813543, 838201], 10: [151340, 163385, 301686, 335989, 473590], 11: [104306, 483493, 613804, 747656, 853595], 12: [89360, 163056, 441057, 529511, 974931], 13: [52392, 214491, 535195, 673344, 675302], 14: [318154, 592808, 702519, 857578, 878410]}, 15: {5: [32939, 430725, 732722, 856813, 942271], 6: [454529, 583547, 836517, 912555, 963343], 7: [118494, 179970, 474289, 738916, 990150], 8: [7268, 33052, 57385, 456377, 833366], 9: [214517, 307630, 422765, 423226, 932818], 10: [387850, 428060, 531373, 680416, 680761], 11: [7914, 95693, 624777, 686522, 974425], 12: [112781, 128230, 561951, 671309, 695144], 13: [328723, 620101, 835330, 875557, 978698], 14: [1268, 307915, 585286, 914867, 916067]}, 16: {5: [32939, 430725, 732722, 856813, 942271], 6: [343464, 548243, 646002, 652264, 952253], 7: [147352, 217998, 320223, 487523, 678893], 8: [99326, 228013, 229999, 532371, 657138], 9: [145214, 187129, 378606, 685855, 880347], 10: [80496, 267295, 335154, 871604, 914277], 11: [13581, 508829, 588358, 824950, 949395], 12: [172467, 564934, 608777, 817390, 935980], 13: [80266, 315002, 456408, 647495, 764233], 14: [33483, 388004, 564052, 705732, 956375]}, 17: {5: [32939, 430725, 732722, 856813, 942271], 6: [160216, 262543, 399184, 550467, 898867], 7: [103170, 189574, 196460, 395793, 781155], 8: [84418, 335073, 850369, 952290, 988872], 9: [253664, 452389, 702632, 772899, 950440], 10: [202791, 257061, 511887, 839931, 991294], 11: [336815, 368594, 468447, 826710, 849241], 12: [328723, 620101, 835330, 875557, 978698], 13: [49195, 325678, 515277, 752583, 884982], 14: [16396, 108809, 279776, 464297, 963084]}, 18: {5: [32939, 430725, 732722, 856813, 942271], 6: [98294, 172906, 236010, 273571, 906658], 7: [301153, 407585, 467327, 676094, 793742], 8: [390466, 547039, 771874, 929719, 963887], 9: [28175, 253299, 662699, 736087, 802302], 10: [168693, 378008, 680915, 780212, 936179], 11: [262825, 519064, 528993, 611816, 986572], 12: [163704, 356315, 580702, 616289, 887145], 13: [54342, 200826, 228525, 451414, 924367], 14: [169134, 452044, 583777, 730666, 969115]}, 19: {5: [32939, 430725, 732722, 856813, 942271], 6: [150900, 326990, 566201, 640056, 664281], 7: [214067, 537867, 671089, 687567, 799756], 8: [281786, 675221, 725580, 853763, 881493], 9: [386780, 408789, 590152, 741063, 907549], 10: [13581, 508829, 588358, 824950, 949395], 11: [21650, 185603, 357072, 652217, 769436], 12: [1268, 307915, 585286, 914867, 916067], 13: [16396, 108809, 279776, 464297, 963084], 14: [99406, 251384, 277286, 377193, 741927]}, 20: {5: [32939, 430725, 732722, 856813, 942271], 6: [97116, 559185, 624815, 738963, 881173], 7: [416011, 471342, 629382, 712034, 779818], 8: [212512, 471138, 762731, 870672, 916155], 9: [265301, 519090, 768467, 791659, 863822], 10: [152482, 226420, 582990, 596703, 942519], 11: [63400, 192877, 244130, 564957, 912519], 12: [22834, 111213, 853810, 862578, 967997], 13: [422335, 546786, 792529, 863510, 986723], 14: [42168, 312447, 698713, 797086, 913554]}}

#function to plot each confusion matrix
def plot_confusion_matrix(matrix, class_names, title, metrics_dict, save_path=None, probability=False, file_format="png", matrix_font_size=10, axis_font_size=10, title_font_size=12, dict_font_size=12):  
    fig = plt.figure(figsize=(8.55, 9.00))
    ax_m, ax_t = fig.subplots(nrows=2,ncols=1, height_ratios=[0.95, 0.05])
    plt.sca(ax_m)
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize = title_font_size)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=matrix_font_size)
    plt.yticks(tick_marks, class_names, fontsize=matrix_font_size)
    #calculate class probability
    if probability:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / row_sums
    # Display values in the matrix cells
    for j in range(len(class_names)):
        for k in range(len(class_names)):
            if probability:
                plt.text(k, j, format(matrix[j, k], '.2f'), horizontalalignment="center", color="white" if matrix[j, k] > matrix.max() / 2 else "black", fontsize=matrix_font_size)
            else:
                plt.text(k, j, format(matrix[j, k], 'd'), horizontalalignment="center", color="white" if matrix[j, k] > matrix.max() / 2 else "black", fontsize=matrix_font_size)
    plt.xlabel('Predicted Label', fontsize = axis_font_size)
    plt.ylabel('True Label', fontsize = axis_font_size)
    #add average response time
    plt.sca(ax_t)
    ax_t.set_axis_off()
    ax_t.margins(x=0, y=0)
    metrics_str = ""
    for k in metrics_dict.keys():
        metrics_str = metrics_str+k+" = "+str(metrics_dict[k])+"\n"
    plt.title(metrics_str, fontsize = dict_font_size)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format=file_format, bbox_inches='tight')
    #plt.show()

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
        right = left + image.size[0]*transform_params["cover"]
        top = image.size[1]*(1 - transform_params["cover"])/2
        bottom = top + image.size[1]*transform_params["cover"]
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
        img_np = np.array(image.resize((int(transform_params["scale"]*image.size[0]), int(transform_params["scale"]*image.size[1]))).rotate(transform_params["angle"]))
        image.close()
        img_np = img_np + np.random.normal(0.0, transform_params["noise"], img_np.shape)
        image = Image.fromarray((img_np * 255).astype(np.uint8))
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
        image = Image.fromarray((img_np * 255).astype(np.uint8))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image.close()
        return Image.open(buffer)
    elif apply_transform == "perlinnoise": #params: width in [4,6,8,10,12], noise in [8, 32]
        img_np = np.asarray(image)
        image.close()
        noise_np = perlin_numpy.generate_perlin_noise_2d(img_np.shape, (transform_params["width"], transform_params["width"]))*transform_params["noise"]
        img_np = img_np + np.stack((noise_np,noise_np,noise_np), axis=2)
        image = Image.fromarray((img_np * 255).astype(np.uint8))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image.close()
        return Image.open(buffer)
    return None
    
#function to compile the index list
def CompileIndex(people, num_classes):
    index_list = []
    for i in range(num_classes):
        for j in people:
            for k in range(num_classes):
                index_list.append([i,j,k])
    return index_list
     
#function to compile the index list
def CompileRefIndex(people, num_classes):
    index_list = []
    for i in range(num_classes):
        for j in people:
            index_list.append([i,j])
    return index_list          
    
#define custom dataset class
class BlipImageDataset(Dataset):
    def __init__(self, index_list, blip_transform, device):
        self.index_list = index_list
        self.blip_transform = blip_transform
        self.device = device

    def __len__(self):
        return len(self.index_list)
        
    def translate_idx(self, idx):
        i = self.index_list[idx][0]
        j = self.index_list[idx][1]
        k = self.index_list[idx][2]
        return i, j, k

    def __getitem__(self, idx):
        i,j,k = self.translate_idx(idx)
        img = None
        #build image path
        if i >= 10 or k>= 10:
            img_path = "Dataset/expansion_images/Regenerations/"+fs_generator_list[i]+"/"+fs_generator_list[k]+"/img_"+str(j)+".png"
        else:
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
            img = Image.open(img_path).convert('RGB')
        #apply blip transform
        img_tensor = self.blip_transform(img).to(self.device)
        img.close()
        return img_tensor

#define custom dataset class
class BlipReferenceImageDataset(Dataset):
    def __init__(self, index_list, blip_transform, device):
        self.index_list = index_list
        self.blip_transform = blip_transform
        self.device = device

    def __len__(self):
        return len(self.index_list)
        
    def translate_idx(self, idx):
        i = self.index_list[idx][0]
        j = self.index_list[idx][1]
        return i, j

    def __getitem__(self, idx):
        i,j = self.translate_idx(idx)
        img = None
        #build image path
        if i >= 10:
            img_path = "Dataset/expansion_images/New Tests/"+self.gen_list[i]+"/img_"+str(j)+".png"
        else:
            img_path = path_dataset+generators[i]+"/"
            if j+1<10: img_path = img_path + "0000" + str(j+1)
            elif j+1<100:  img_path = img_path + "000" + str(j+1)
            else: img_path = img_path + "00" + str(j+1)
            if type(path_termination[generators[i]]) is list:
                img_path = [img_path + pt for pt in path_termination[generators[i]]]
            else:
                img_path = img_path + path_termination[generators[i]]
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
            img = Image.open(img_path).convert('RGB')
        #apply model transform
        img_tensor = self.blip_transform(img).to(self.device)
        img.close()
        return img_tensor    

#define custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, index_list, model_transform, device, apply_transform=None, transform_params=None):
        self.index_list = index_list
        self.model_transform = model_transform
        self.device = device
        self.apply_transform = apply_transform
        self.transform_params = transform_params

    def __len__(self):
        return len(self.index_list)
        
    def translate_idx(self, idx):
        i = self.index_list[idx][0]
        j = self.index_list[idx][1]
        k = self.index_list[idx][2]
        return i, j, k

    def __getitem__(self, idx):
        i,j,k = self.translate_idx(idx)
        img = None
        label = torch.tensor(np.eye(len(generators))[k]).to(self.device)
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
            img = Image.open(img_path).convert('RGB')
        #apply transform if specified
        if self.apply_transform is not None:
            if self.transform_params is None:
                img = Transform(img, self.apply_transform, RandomParams(self.apply_transform))
            else:
                img = Transform(img, self.apply_transform, self.transform_params)
        #apply model transform
        img_tensor = self.model_transform(img).to(self.device)
        img.close()
        return img_tensor, label

#define custom dataset class
class CustomReferenceImageDataset(Dataset):
    def __init__(self, index_list, model_transform, device, apply_transform=None, transform_params=None):
        self.index_list = index_list
        self.model_transform = model_transform
        self.device = device
        self.apply_transform = apply_transform
        self.transform_params = transform_params

    def __len__(self):
        return len(self.index_list)
        
    def translate_idx(self, idx):
        i = self.index_list[idx][0]
        j = self.index_list[idx][1]
        return i, j

    def __getitem__(self, idx):
        i,j = self.translate_idx(idx)
        img = None
        label = torch.tensor(np.eye(len(generators))[i]).to(self.device)
        #build image path
        img_path = path_dataset+generators[i]+"/"
        if j+1<10: img_path = img_path + "0000" + str(j+1)
        elif j+1<100:  img_path = img_path + "000" + str(j+1)
        else: img_path = img_path + "00" + str(j+1)
        if type(path_termination[generators[i]]) is list:
            img_path = [img_path + pt for pt in path_termination[generators[i]]]
        else:
            img_path = img_path + path_termination[generators[i]]
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
            img = Image.open(img_path).convert('RGB')
        #apply transform if specified
        if self.apply_transform is not None:
            if self.transform_params is None:
                img = Transform(img, self.apply_transform, RandomParams(self.apply_transform))
            else:
                img = Transform(img, self.apply_transform, self.transform_params)
        #apply model transform
        img_tensor = self.model_transform(img).to(self.device)
        img.close()
        return img_tensor, label

#define custom dataset class
class CaptionImageDataset(Dataset):
    def __init__(self, index_list, captions, model_transform, device, apply_transform=None, transform_params=None):
        self.index_list = index_list
        self.captions = captions
        self.model_transform = model_transform
        self.device = device
        self.apply_transform = apply_transform
        self.transform_params = transform_params

    def __len__(self):
        return len(self.index_list)
        
    def translate_idx(self, idx):
        i = self.index_list[idx][0]
        j = self.index_list[idx][1]
        k = self.index_list[idx][2]
        return i, j, k

    def __getitem__(self, idx):
        i,j,k = self.translate_idx(idx)
        img = None
        label = torch.tensor(np.eye(len(generators))[k]).to(self.device)
        caption = self.captions[i,j,k]
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
            img = Image.open(img_path).convert('RGB')
        #apply transform if specified
        if self.apply_transform is not None:
            if self.transform_params is None:
                img = Transform(img, self.apply_transform, RandomParams(self.apply_transform))
            else:
                img = Transform(img, self.apply_transform, self.transform_params)
        #apply model transform
        img_tensor = self.model_transform(img).to(self.device)
        img.close()
        return img_tensor, caption, label

#define custom dataset class
class CaptionReferenceImageDataset(Dataset):
    def __init__(self, index_list, captions, model_transform, device, apply_transform=None, transform_params=None):
        self.index_list = index_list
        self.captions = captions
        self.model_transform = model_transform
        self.device = device
        self.apply_transform = apply_transform
        self.transform_params = transform_params

    def __len__(self):
        return len(self.index_list)
        
    def translate_idx(self, idx):
        i = self.index_list[idx][0]
        j = self.index_list[idx][1]
        return i, j

    def __getitem__(self, idx):
        i,j = self.translate_idx(idx)
        img = None
        label = torch.tensor(np.eye(len(generators))[i]).to(self.device)
        caption = self.captions[i,j]
        #build image path
        img_path = path_dataset+generators[i]+"/"
        if j+1<10: img_path = img_path + "0000" + str(j+1)
        elif j+1<100:  img_path = img_path + "000" + str(j+1)
        else: img_path = img_path + "00" + str(j+1)
        if type(path_termination[generators[i]]) is list:
            img_path = [img_path + pt for pt in path_termination[generators[i]]]
        else:
            img_path = img_path + path_termination[generators[i]]
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
            img = Image.open(img_path).convert('RGB')
        #apply transform if specified
        if self.apply_transform is not None:
            if self.transform_params is None:
                img = Transform(img, self.apply_transform, RandomParams(self.apply_transform))
            else:
                img = Transform(img, self.apply_transform, self.transform_params)
        #apply model transform
        img_tensor = self.model_transform(img).to(self.device)
        img.close()
        return img_tensor, caption, label

#define custom dataset class
class FewShotImageDataset(Dataset):
    def __init__(self, index_list, gen_list, model_transform, device, captions=None):
        self.index_list = index_list
        self.gen_list = gen_list
        self.model_transform = model_transform
        self.device = device
        self.captions = captions

    def __len__(self):
        return len(self.index_list)
        
    def translate_idx(self, idx):
        i = self.index_list[idx][0]
        j = self.index_list[idx][1]
        k = self.index_list[idx][2]
        return i, j, k

    def __getitem__(self, idx):
        i,j,k = self.translate_idx(idx)
        img = None
        label = torch.tensor(np.eye(len(self.gen_list))[k]).to(self.device)
        caption = None
        if self.captions is not None:
            caption = self.captions[i,j,k]
        #build image path
        if i >= 10 or k>= 10:
            img_path = "Dataset/expansion_images/Regenerations/"+fs_generator_list[i]+"/"+fs_generator_list[k]+"/img_"+str(j)+".png"
        else:
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
            img = Image.open(img_path).convert('RGB')
        #apply model transform
        img_tensor = self.model_transform(img).to(self.device)
        img.close()
        if caption is None:
            return img_tensor, label
        else:
            return img_tensor, caption, label

#define custom dataset class
class FewShotReferenceImageDataset(Dataset):
    def __init__(self, index_list, gen_list, model_transform, device, apply_transform=None, transform_params=None, captions=None):
        self.index_list = index_list
        self.gen_list = gen_list
        self.apply_transform = apply_transform
        self.transform_params = transform_params
        self.model_transform = model_transform
        self.device = device
        self.captions = captions

    def __len__(self):
        return len(self.index_list)
        
    def translate_idx(self, idx):
        i = self.index_list[idx][0]
        j = self.index_list[idx][1]
        return i, j

    def __getitem__(self, idx):
        i,j = self.translate_idx(idx)
        img = None
        label = torch.tensor(np.eye(len(self.gen_list))[i]).to(self.device)
        caption = None
        if self.captions is not None:
            caption = self.captions[i,j]
        #build image path
        if i >= 10:
            img_path = "Dataset/expansion_images/New Tests/"+fs_generator_list[i]+"/img_"+str(j)+".png"
        else:
            img_path = path_dataset+self.gen_list[i]+"/"
            if j+1<10: img_path = img_path + "0000" + str(j+1)
            elif j+1<100:  img_path = img_path + "000" + str(j+1)
            else: img_path = img_path + "00" + str(j+1)
            if type(path_termination[self.gen_list[i]]) is list:
                img_path = [img_path + pt for pt in path_termination[self.gen_list[i]]]
            else:
                img_path = img_path + path_termination[self.gen_list[i]]
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
            img = Image.open(img_path).convert('RGB')        
        #apply transform if specified
        if self.apply_transform is not None:
            if self.transform_params is None:
                img = Transform(img, self.apply_transform, RandomParams(self.apply_transform))
            else:
                img = Transform(img, self.apply_transform, self.transform_params)
        #apply model transform
        img_tensor = self.model_transform(img).to(self.device)
        img.close()
        if caption is None:
            return img_tensor, label
        else:
            return img_tensor, caption, label