import sys
import os
import pickle
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sklearn.neural_network
import sklearn.metrics
import sklearn.ensemble
import sklearn.svm
import sklearn.linear_model

#parameters
path_features_images = "Extracted_Features/clip_features.npy"
path_features_references_base = "Extracted_Features/clip_references_features"
generators = ["bing", "firefly", "flux-dev", "freepik", "imagen", "leonardoai", "midjourney", "nightcafe", "stabilityai", "starryai"] #generators for dataset 100 people
num_people = 100
subset = "all" #"test" -> evaluates only on the test set for comparison purposes, "validation" -> evaluates only on the test set for comparison purposes,  "all" -> evaluates on all the pairings
shares = {'tr': 0.8, 'va': 0.1, 'te': 0.1}
seed = 8375
distance = sys.argv[1] #choose between "euclidean", "manhattan", "cosine", "correlation"

#function to plot each confusion matrix
def plot_confusion_matrix(matrix, class_names, title, metrics_dict, save_path=None):  
    fig = plt.figure(figsize=(8.55, 9.00))
    ax_m, ax_t = fig.subplots(nrows=2,ncols=1, height_ratios=[0.95, 0.05])
    plt.sca(ax_m)
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    # Display values in the matrix cells
    for j in range(len(class_names)):
        for k in range(len(class_names)):
            plt.text(k, j, format(matrix[j, k], 'd'), horizontalalignment="center", color="white" if matrix[j, k] > matrix.max() / 2 else "black")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    #add average response time
    plt.sca(ax_t)
    ax_t.set_axis_off()
    ax_t.margins(x=0, y=0)
    metrics_str = ""
    for k in metrics_dict.keys():
        metrics_str = metrics_str+k+" = "+str(metrics_dict[k])+"\n"
    plt.title(metrics_str)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    #plt.show()
    
#function to calculate distance
def calculate_distance(img, ref, method):
    if method=="euclidean":
        return scipy.spatial.distance.euclidean(img, ref)
    elif method=="manhattan":
        return scipy.spatial.distance.cityblock(img, ref)
    elif method=="cosine":
        return scipy.spatial.distance.cosine(img, ref)
    elif method=="correlation":
        return scipy.spatial.distance.correlation(img, ref)

#check subset
if subset not in ['test', 'validation', 'all']:
    sys.exit("Error, unknown subset : "+subset)

#check distance
if distance not in ["euclidean", "manhattan", "cosine", "correlation"]:
    sys.exit("Error, unknown distance : "+distance)

#cycle over all possible values of apply_transform
for apply_transform in ["plain", "jpeg", "resize", "crop", "webp", "rotation", "contrast", "brightness", "blur", "social", "greyscale", "cmyk", "hsv", "median", "printscan", "twitter", "whitenoise"]:
    path_features_references = path_features_references_base + "_" + apply_transform + ".npy" if apply_transform != "plain" else path_features_references_base + ".npy"

    #extract feature vectors and supervisions
    print("Loading image features extracted from CLIP")
    in_file = open(path_features_images, 'rb')
    img_data = np.load(in_file)
    in_file.close()
    print("Loading reference features extracted from CLIP")
    in_file = open(path_features_references, 'rb')
    ref_data = np.load(in_file)
    in_file.close()

    #shuffle dataset and create sets
    print("Shuffling dataset and building sets")
    indices = list(range(img_data.shape[1]))
    np.random.seed(seed)
    np.random.shuffle(indices)
    ind = {}
    ind['tr'] = indices[:int(len(indices)*shares['tr'])]
    ind['te'] = indices[int(len(indices)*shares['tr']):int(len(indices)*shares['tr'])+int(len(indices)*shares['va'])]
    ind['va'] = indices[int(len(indices)*shares['tr'])+int(len(indices)*shares['va']):]

    #select subset if indicated
    if(subset == "validation"):
        X_img = img_data[:,ind['va'],:,:]
        X_ref = ref_data[:,ind['va'],:]
    if(subset == "test"):
        X_img = img_data[:,ind['te'],:,:]
        X_ref = ref_data[:,ind['te'],:]
    if(subset == "all"):
        X_img = img_data
        X_ref = ref_data
        
    #use first generators as supervisions
    Y = np.concatenate([np.tile(np.eye(len(generators))[i], (X_ref.shape[1],1)) for i in range(len(generators))])
    #remove first generators from features by concatenation
    X_img = np.concatenate([X_img[i,:,:,:] for i in range(len(generators))])
    X_ref = np.concatenate([X_ref[i,:,:] for i in range(len(generators))])

    #calculate distances
    distances = np.zeros((X_img.shape[0], len(generators)))
    print("Calculating distances")
    for i in range(X_img.shape[0]):
        for j in range(len(generators)):
            distances[i][j] = calculate_distance(X_img[i][j], X_ref[i], distance)

    #calculate image selection probabilities and predictions based on distance
    print("Calculating image selection probabilities and predictions based on distance")
    prob = scipy.special.softmax(-distances, axis=1)
    pred = np.argmin(distances, axis=1)
    truth = np.argmax(Y, axis=1)

    #calculate metrics
    print("Calculating Metrics")
    confusion_matrix = sklearn.metrics.confusion_matrix(truth, pred)
    accuracy = sklearn.metrics.accuracy_score(truth, pred)
    top3_acc = sklearn.metrics.top_k_accuracy_score(truth, prob, k=3)
    roc_auc = sklearn.metrics.roc_auc_score(Y, prob)

    #plotting results
    save_path = distance+"_"+subset+"_"+apply_transform+".png"
    plot_confusion_matrix(confusion_matrix, generators, "Confusion Matrix", {"Accuracy": accuracy, "Top3 Acc": top3_acc, "ROC_AUC": roc_auc}, save_path)

#terminate execution
print("Execution Terminated Succesfully!")