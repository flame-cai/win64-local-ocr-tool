import os
import argparse
import numpy as np
import cv2
import torch
import time
from scipy.signal import find_peaks
import torch.nn.functional as F
from skimage import io
import torch.nn as nn
import torch.nn.init as init
import torchvision
from torchvision import models

from scipy.ndimage import maximum_filter
from scipy.ndimage import label
import numpy.random as npr

from collections import namedtuple
from packaging import version
from collections import OrderedDict

import matplotlib.patches as patches
import matplotlib.pyplot as plt


# Function Definitions
def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

class vgg16_bn(torch.nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(vgg16_bn, self).__init__()
        if version.parse(torchvision.__version__) >= version.parse('0.13'):
            vgg_pretrained_features = models.vgg16_bn(
                weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None
            ).features
        else: #torchvision.__version__ < 0.13
            models.vgg.model_urls['vgg16_bn'] = models.vgg.model_urls['vgg16_bn'].replace('https://', 'http://')
            vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(12):         # conv2_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):         # conv3_3
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):         # conv4_3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):         # conv5_3
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
            
        # fc6, fc7 without atrous conv
        self.slice5 = torch.nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                nn.Conv2d(1024, 1024, kernel_size=1)
        )

        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())

        init_weights(self.slice5.modules())        # no pretrained model for fc6 and fc7

        if freeze:
            for param in self.slice1.parameters():      # only first conv
                param.requires_grad= False

    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        try: # multi gpu needs this
            self.rnn.flatten_parameters()
        except: # quantization doesn't work with this
            pass
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output

class VGG_FeatureExtractor(nn.Module):

    def __init__(self, input_channel, output_channel=256):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True))

    def forward(self, input):
        return self.ConvNet(input)



class Model(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        super(Model, self).__init__()
        """ FeatureExtraction """
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.SequenceModeling_output = hidden_size

        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)


    def forward(self, input, text):
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction

"""### CRAFT Model"""

#CRAFT

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CRAFT(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()

        """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)

        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0,2,3,1)

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def detect(img, detector, device):


        x = [np.transpose(normalizeMeanVariance(img), (2, 0, 1))]
        x = torch.from_numpy(np.array(x))
        x = x.to(device)
        with torch.no_grad():
            y = detector(x)
        region_score = y[0,:,:,0].cpu().data.numpy()
        affinity_score = y[0,:,:,1].cpu().data.numpy()
        return region_score,affinity_score

def load_images_from_folder(folder_path):
    inp_images = []
    file_names = []
    
    # Get all files in the directory
    files = sorted(os.listdir(folder_path))
    
    for file in files:
        # Check if the file is an image (PNG or JPG)
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Construct the full file path
                file_path = os.path.join(folder_path, file)
                
                # Open the image file
                image = loadImage(file_path)
                
                # Append the image and filename to our lists
                inp_images.append(image)
                file_names.append(file)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    
    return inp_images, file_names

def heatmap_to_pointcloud(heatmap, min_peak_value=0.3, min_distance=10):
    """
    Convert a 2D heatmap to a point cloud by identifying local maxima and generating
    points with density proportional to the heatmap intensity.
    
    Parameters:
    -----------
    heatmap : numpy.ndarray
        2D array representing the heatmap
    min_peak_value : float
        Minimum value for a peak to be considered (normalized between 0 and 1)
    min_distance : int
        Minimum distance between peaks in pixels
        
    Returns:
    --------
    points : numpy.ndarray
        Array of shape (N, 2) containing the generated points
    """
    # Normalize heatmap to [0, 1]
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    # Find local maxima
    local_max = maximum_filter(heatmap_norm, size=min_distance)
    peaks = (heatmap_norm == local_max) & (heatmap_norm > min_peak_value)
    
    # Label connected components
    labeled_peaks, num_peaks = label(peaks)
    
    points = []
    
    # For each peak, generate points
    height = heatmap.shape[0]  # Get the height of the heatmap
    for peak_idx in range(1, num_peaks + 1):
        # Get peak location
        peak_y, peak_x = np.where(labeled_peaks == peak_idx)[0][0], np.where(labeled_peaks == peak_idx)[1][0]
        #points.append([peak_x, peak_y])
        points.append([peak_x, height - 1 - peak_y])  # This line is modified

    return np.array(points)

def visualize_results(heatmap, points):
    """
    Visualize the original heatmap and overlay the resulting point cloud on it.
    """
    import matplotlib.pyplot as plt
    
    # Create a single figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    ax.imshow(heatmap, cmap='hot')
    
    # Overlay the point cloud on the heatmap
    ax.scatter(points[:, 0], points[:, 1], s=1, c='blue', alpha=0.5)
    
    # Set title and axis limits
    ax.set_title('Heatmap with Generated Point Cloud')
    ax.set_xlim(0, heatmap.shape[1])
    ax.set_ylim(heatmap.shape[0], 0)  # Invert y-axis to match image coordinates
    
    plt.tight_layout()
    return fig

def process_image(folder_path):
    m_name = os.path.basename(os.path.dirname(folder_path))
    if os.path.exists(f'/mnt/cai-data/layout-analysis/manuscripts/{m_name}/heatmaps') == False:
        os.makedirs(f'/mnt/cai-data/layout-analysis/manuscripts/{m_name}/heatmaps')

    device = torch.device('cuda') #change to cpu if no gpu

    #load images from folder
    inp_images, file_names = load_images_from_folder(folder_path)

    #load craft model
    _detector = CRAFT()
    _detector.load_state_dict(copyStateDict(torch.load("/mnt/cai-data/layout-analysis/models/craft_mlt_25k.pth",map_location=device)))
    detector = torch.nn.DataParallel(_detector).to(device)
    detector.eval()

    st = time.time()


    # for each image in the folder
    test_data_path = '/home/kartik/layout-analysis/data/test-data/'

    pg_counter=0
    for image,_filename in zip(inp_images, file_names):
        # get region score and affinity score
        region_score, affinity_score = detect(image,detector, device)
        assert region_score.shape == affinity_score.shape

        # resize image to match the shape of the heatmaps
        image = cv2.resize(image, dsize=region_score.shape[::-1], interpolation=cv2.INTER_CUBIC)
        # Array of shape (N, 2) - x,y locations of each peak
        points = heatmap_to_pointcloud(region_score, min_peak_value=0.3, min_distance=10)
        print(points.shape[0])
        # Save figure
        print(region_score.shape)
        print(points.shape)
        np.savetxt(test_data_path+f'pg_{pg_counter}_points.txt', points, fmt='%d')

        fig = visualize_results(region_score, points)
        plt.show()
        plt.axis('off')  # Turn off the axis        
        #plt.savefig(f'/mnt/cai-data/layout-analysis/manuscripts/{m_name}/heatmaps/{_filename}',dpi=300, bbox_inches='tight', pad_inches=0)
        #plt.savefig(f'/home/kartik/layout-analysis/analysis_images/{_filename}',dpi=300, bbox_inches='tight', pad_inches=0)
        pg_counter +=1

    print(f"{time.time()-st:.2f} seconds elapsed.....")

        









# Create the arg parser
parser = argparse.ArgumentParser(description="A simple script to process a path")
parser.add_argument('--path', type=str, help='The path to folder which contains leaf images', default="/mnt/cai-data/layout-analysis/manuscripts/DMV/leaves")

args = parser.parse_args()
folder_path = args.path

process_image(folder_path)