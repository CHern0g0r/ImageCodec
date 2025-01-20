#This code is the simplest example of image compression based on neural networks
#Comparison with JPEG is provided as well
#It is a demonstation for Information Theory course
#Written by Evgeny Belyaev, July 2024.
import os
import math
import numpy
import torch
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import imghdr
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
#from keras import backend as K
import tensorflow.keras.backend as K

from skimage.metrics import structural_similarity as ssim

from src.model import ResModel

#from tensorflow.keras.callbacks import ModelCheckpoint

#import C-implementation of Witten&Neal&Cleary-1987 arithmetic coding as a external module
from EntropyCodec import *

#source folder with test images
testfolder = './test/'
#source folder with train images
trainfolder = './train/'
#size of test and train images
w=128
h=128
#If 0, then the training will be started, otherwise the model will be readed from a file
LoadModel = 1

#Number of bits for representation of the layers sample in the training process
bt = 2
#Training parameters 
# epochs = 2000
epochs = 500

#Model parameters
# batch_sizeM1 = 24
batch_sizeM1 = 48
n1M1=128
n2M1=32
n3M1=16

#Number of images to be compressed and shown from the test folder
NumImagesToShow = 21

#Number of bits for representation of the layers sample
b = 2

#Compute PSNR in RGB domain
def PSNR_RGB(image1,image2):
    width, height = image1.size
    I1 = numpy.array(image1.getdata()).reshape(image1.size[0], image1.size[1], 3)
    I2 = numpy.array(image2.getdata()).reshape(image2.size[0], image2.size[1], 3)
    I1 = numpy.reshape(I1, width * height * 3)
    I2 = numpy.reshape(I2, width * height * 3)
    I1 = I1.astype(float)
    I2 = I2.astype(float)
    mse = numpy.mean((I1 - I2) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        psnr=100.0
    else:
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    #print("PSNR = %5.2f dB" % psnr)
    return psnr

#Compute PSNR between two vectors
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))

#reads all images from folder and puts them into x array
def LoadImagesFromFolder (foldername):
    dir_list = os.listdir(foldername)
    N = 0
    Nmax = 0
    for name in dir_list:
        fullname = foldername + name
        filetype = imghdr.what(fullname)
        if filetype is None:
            print('')
        else:
            Nmax = Nmax + 1

    x = numpy.zeros([Nmax, w, h, 3])
    N = 0
    for name in dir_list:
        fullname = foldername + name
        filetype = imghdr.what(fullname)
        if filetype is None:
            print('Unknown image format for file: ', name)
        else:
            print('Progress: N = %i' % N)
            image = Image.open(fullname)
            I1 = numpy.array(image.getdata()).reshape(image.size[0], image.size[1], 3)
            x[N, :, :, :] = I1
            N = N + 1
    return x

#Model training function
def ImageCodecModel(trainfolder,trainwithnoise):
    input = layers.Input(shape=(w, h, 3))
    # Encoder
    e1 = layers.Conv2D(n1M1, (7, 7), activation="relu", padding="same")(input)
    e1 = layers.AveragePooling2D((2, 2), padding="same")(e1)
    e2 = layers.Conv2D(n2M1, (5, 5), activation="relu", padding="same")(e1)
    e2 = layers.AveragePooling2D((2, 2), padding="same")(e2)
    e3 = layers.Conv2D(n3M1, (3, 3), activation="relu", padding="same")(e2)
    e3 = layers.AveragePooling2D((2, 2), padding="same")(e3)
    layers.BatchNormalization()
    if trainwithnoise==1:
        maxt = tensorflow.keras.ops.max(e3)
        e3 = e3 + maxt*keras.random.uniform(shape=(16,16,16), minval=-1.0/pow(2, bt+1), maxval=1.0/pow(2, bt+1), dtype=None, seed=None)
    
    # Decoder
    x = layers.Conv2DTranspose(n3M1, (3, 3), strides=2, activation="relu", padding="same")(e3)
    x = layers.Conv2DTranspose(n2M1, (5, 5), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(n1M1, (7, 7), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    # Autoencoder
    encoder = Model(input, e3)
    decoder = Model(e3, x)
    autoencoder = Model(input, x)
    autoencoder.compile(optimizer="adam", loss='mean_squared_error')
    autoencoder.summary()

    if LoadModel == 0:
        print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))
        xtrain = LoadImagesFromFolder(trainfolder)
        xtrain = xtrain / 255

        with tensorflow.device('gpu'):    
            autoencoder.fit(xtrain, xtrain, epochs=epochs, batch_size=batch_sizeM1,shuffle=True)
        
        if trainwithnoise==1:
            encoder.save('encoder2.keras')
            decoder.save('decoder2.keras')
        else:
            encoder.save('encoder.keras')
            decoder.save('decoder.keras')
    else:
        if trainwithnoise==1:
            encoder = keras.models.load_model('encoder2.keras',safe_mode=False)
            decoder = keras.models.load_model('decoder2.keras',safe_mode=False)
        else:
            encoder = keras.models.load_model('encoder.keras',safe_mode=False)
            decoder = keras.models.load_model('decoder.keras',safe_mode=False)
    return encoder,decoder


#Compresses input layer by multi-alphabet arithmetic coding using memoryless source model
def EntropyEncoder (filename,enclayers,size_z,size_h,size_w):
    temp = numpy.zeros((size_z, size_h, size_w), numpy.uint8, 'C')
    for z in range(size_z):
        for h in range(size_h):
            for w in range(size_w):
                temp[z][h][w] = enclayers[z][h][w]
    maxbinsize = (size_h * size_w * size_z)
    bitstream = numpy.zeros(maxbinsize, numpy.uint8, 'C')
    StreamSize = numpy.zeros(1, numpy.int32, 'C')
    HiddenLayersEncoder(temp, size_w, size_h, size_z, bitstream, StreamSize)
    name = filename
    path = './'
    fp = open(os.path.join(path, name), 'wb')
    out = bitstream[0:StreamSize[0]]
    out.astype('uint8').tofile(fp)
    fp.close()

#Decompresses input layer by multi-alphabet arithmetic coding using memoryless source model
def EntropyDecoder (filename,size_z,size_h,size_w):
    fp = open(filename, 'rb')
    bitstream = fp.read()
    fp.close()
    bitstream = numpy.frombuffer(bitstream, dtype=numpy.uint8)
    declayers = numpy.zeros((size_z, size_h, size_w), numpy.uint8, 'C')
    FrameOffset = numpy.zeros(1, numpy.int32, 'C')
    FrameOffset[0] = 0
    HiddenLayersDecoder(declayers, size_w, size_h, size_z, bitstream, FrameOffset)
    return declayers

#This function is searching for the JPEG quality factor (QF)
#which provides neares compression to TargetBPP
def JPEGRDSingleImage(X,TargetBPP,i):
    X = X*255
    image = Image.fromarray(X.astype('uint8'), 'RGB')
    width, height = image. size
    realbpp = 0
    realpsnr = 0
    realQ = 0
    for Q in range(101):
        image.save('test.jpeg', "JPEG", quality=Q)
        image_dec = Image.open('test.jpeg')
        bytesize = os.path.getsize('test.jpeg')
        bpp = bytesize*8/(width*height)
        psnr = PSNR_RGB(image, image_dec)
        if abs(realbpp-TargetBPP)>abs(bpp-TargetBPP):
            realbpp=bpp
            realpsnr=psnr
            realQ = Q
    image.save('test.jpeg', "JPEG", quality=realQ)
    image_dec = Image.open('test.jpeg')
    I1 = numpy.array(image.getdata()).reshape(image.size[0], image.size[1], 3)
    I2 = numpy.array(image_dec.getdata()).reshape(image_dec.size[0], image_dec.size[1], 3)

    #print('\n\n Size = ',numpy.shape(I1))
    psnr = ssim(I1[ :, :, 0], I2[ :, :, 0],data_range=255.0)
    psnr = psnr + ssim(I1[ :, :, 1], I2[ :, :, 1],data_range=255.0)
    psnr = psnr + ssim(I1[ :, :, 2], I2[ :, :, 2],data_range=255.0)
    realpsnr=psnr/3.0
    JPEGfilename = 'image%i.jpeg' % i
    image.save(JPEGfilename, "JPEG", quality=realQ)
    return realQ, realbpp, realpsnr


def NeuralCompressor(enc,dec):
    #Run the model for first NumImagesToShow images from the test set
    encoded_layers = enc.predict(xtest, batch_size=NumImagesToShow)
    # print(encoded_layers.shape)
    # print(encoded_layers.min(), encoded_layers.max())
    # exit()
    max_encoded_layers = numpy.zeros(NumImagesToShow, numpy.float16, 'C')

    #normalization the layer to interval [0,1)
    for i in range(NumImagesToShow):
        max_encoded_layers[i] = numpy.max(encoded_layers[i])
        encoded_layers[i] = encoded_layers[i] / max_encoded_layers[i]

    #Quantization of layer to b bits
    encoded_layers1 = numpy.clip(encoded_layers, 0, 0.9999999)
    encoded_layers1 = K.cast(encoded_layers1*pow(2, b), "int32")

    #Encoding and decoding of each quantized layer by arithmetic coding
    bpp = numpy.zeros(NumImagesToShow, numpy.float16, 'C')
    declayers = numpy.zeros((NumImagesToShow,16, 16, 16), numpy.uint8, 'C')
    for i in range(NumImagesToShow):
        binfilename = 'image%i.bin' % i
        EntropyEncoder(binfilename, encoded_layers1[i], 16, 16, 16)
        bytesize = os.path.getsize(binfilename)
        bpp[i] = bytesize * 8 / (w * h)
        declayers[i] = EntropyDecoder(binfilename,  16, 16, 16)

    #Dequantization and denormalization of each layer
    print(bpp)
    shift = 1.0/pow(2, b+1)
    declayers = K.cast(declayers, "float32") / pow(2, b)
    declayers = declayers + shift
    encoded_layers_quantized = numpy.zeros((NumImagesToShow, 16, 16, 16), numpy.double, 'C')
    for i in range(NumImagesToShow):
        encoded_layers_quantized[i] = K.cast(declayers[i]*max_encoded_layers[i], "float32")
        encoded_layers[i] = K.cast(encoded_layers[i] * max_encoded_layers[i], "float32")
    decoded_imgs = dec.predict(encoded_layers, batch_size=NumImagesToShow)
    decoded_imgsQ = dec.predict(encoded_layers_quantized, batch_size=NumImagesToShow)
    return bpp, decoded_imgs, decoded_imgsQ


def load_torch(exp, ep):
    model = ResModel()
    pth = f'./exp/{exp}/models'
    model.enc.load_state_dict(
        torch.load(f'{pth}/enc{ep}.pt', weights_only=True, map_location=torch.device('cpu'))
    )
    model.dec.load_state_dict(
        torch.load(f'{pth}/dec{ep}.pt', weights_only=True, map_location=torch.device('cpu'))
    )
    return model


def NeuralCompressor_torch(model, data):
    #Run the model for first NumImagesToShow images from the test set
    el = model.enc(data).detach()
    encoded_layers = el.numpy()
    # print(encoded_layers.shape)
    # print(encoded_layers.min(), encoded_layers.max())
    # exit()
    max_encoded_layers = numpy.zeros(NumImagesToShow, numpy.float16, 'C')

    #normalization the layer to interval [0,1)
    for i in range(NumImagesToShow):
        max_encoded_layers[i] = numpy.max(encoded_layers[i])
        encoded_layers[i] = encoded_layers[i] / max_encoded_layers[i]

    # max_el = el.amax(dim=(1, 2, 3), keepdim=True)
    # nel = el / max_el

    #Quantization of layer to b bits
    encoded_layers1 = numpy.clip(encoded_layers, 0, 0.9999999)
    encoded_layers1 = K.cast(encoded_layers1*pow(2, b), "int32")

    # qel = (torch.clip(nel, 0, 0.9999999) * pow(2, b)).to(
    #     torch.int32
    # )
    # qel = qel.numpy()

    #Encoding and decoding of each quantized layer by arithmetic coding
    bpp = numpy.zeros(NumImagesToShow, numpy.float16, 'C')
    declayers = numpy.zeros((NumImagesToShow,16, 16, 16), numpy.uint8, 'C')
    for i in range(NumImagesToShow):
        binfilename = 'image%i.bin' % i
        EntropyEncoder(binfilename, encoded_layers1[i], 16, 16, 16)
        bytesize = os.path.getsize(binfilename)
        bpp[i] = bytesize * 8 / (w * h)
        declayers[i] = EntropyDecoder(binfilename,  16, 16, 16)

    #Dequantization and denormalization of each layer
    print(bpp)
    shift = 1.0/pow(2, b+1)
    declayers = K.cast(declayers, "float32") / pow(2, b)
    declayers = declayers + shift
    encoded_layers_quantized = numpy.zeros((NumImagesToShow, 16, 16, 16), numpy.double, 'C')
    for i in range(NumImagesToShow):
        encoded_layers_quantized[i] = K.cast(declayers[i]*max_encoded_layers[i], "float32")
        encoded_layers[i] = K.cast(encoded_layers[i] * max_encoded_layers[i], "float32")
    encoded_layers = torch.Tensor(encoded_layers)
    encoded_layers_quantized = torch.Tensor(encoded_layers_quantized)
    decoded_imgs = model.dec(encoded_layers)
    decoded_imgsQ = model.dec(encoded_layers_quantized)
    return bpp, decoded_imgs, decoded_imgsQ


exp, epoch = 'bs5bn', 200

# Main function
if __name__ == '__main__':
    #Load test images
    xtest = LoadImagesFromFolder(testfolder)
    xtest = xtest / 255
    torch_test = torch.Tensor(xtest).permute(0, 3, 1, 2)
    # testm = torch_test.mean(dim=(2, 3))[:, :, None, None]
    # tests = torch_test.std(dim=(2, 3))[:, :, None, None]
    # torch_test = (torch_test - testm) / tests


    #Train/load the model
    encoder, decoder = ImageCodecModel(trainfolder,0)
    encoder2, decoder2 = ImageCodecModel(trainfolder,1)
    model = load_torch(exp, epoch)
    model.eval()

    bpp, decoded_imgs, decoded_imgsQ = NeuralCompressor(encoder,decoder)
    bpp2, decoded_imgs2, decoded_imgsQ2 = NeuralCompressor(encoder2,decoder2)
    bpp3, decoded_imgs3, decoded_imgsQ3 = NeuralCompressor_torch(model, torch_test)

    decoded_imgs3 = decoded_imgs3.detach(
    ).permute(0, 2, 3, 1).numpy()
    decoded_imgsQ3 = decoded_imgsQ3.detach(
    ).permute(0, 2, 3, 1).numpy()

    

    for i in range(NumImagesToShow):
            title=''
            plt.subplot(5, NumImagesToShow, i + 1).set_title(title, fontsize=10)
            if i==0:
                plt.subplot(5, NumImagesToShow, i + 1).text(-50, 64, 'RAW')
            plt.imshow(xtest[i, :, :, :], interpolation='nearest')
            plt.axis(False)
    # for i in range(NumImagesToShow):
    #     #psnr = PSNR(xtest[i, :, :, :], decoded_imgsQ[i, :, :, :])
    #     psnr = ssim(xtest[i, :, :, 0], decoded_imgsQ[i, :, :, 0],data_range=1.0)
    #     psnr = psnr + ssim(xtest[i, :, :, 1], decoded_imgsQ[i, :, :, 1],data_range=1.0)
    #     psnr = psnr + ssim(xtest[i, :, :, 2], decoded_imgsQ[i, :, :, 2],data_range=1.0)
    #     psnr=psnr/3.0

    #     #title = '%2.2f %2.2f' % (psnr, bpp[i])
    #     title = 'Q=%2.2f bpp=%2.2f' % (psnr, bpp[i])
    #     plt.subplot(5, NumImagesToShow, NumImagesToShow + i + 1).set_title(title, fontsize=10)
    #     if i==0:
    #         plt.subplot(5, NumImagesToShow, NumImagesToShow + i + 1).text(-50, 64, 'AE1')        
    #     plt.imshow(decoded_imgsQ[i, :, :, :], interpolation='nearest')
    #     plt.axis(False)
    for i in range(NumImagesToShow):
        #psnr = PSNR(xtest[i, :, :, :], decoded_imgsQ2[i, :, :, :])
        psnr = ssim(xtest[i, :, :, 0], decoded_imgsQ2[i, :, :, 0],data_range=1.0)
        psnr = psnr + ssim(xtest[i, :, :, 1], decoded_imgsQ2[i, :, :, 1],data_range=1.0)
        psnr = psnr + ssim(xtest[i, :, :, 2], decoded_imgsQ2[i, :, :, 2],data_range=1.0)
        psnr=psnr/3.0
        #title = '%2.2f %2.2f' % (psnr, bpp2[i])
        title = 'Q=%2.2f bpp=%2.2f' % (psnr, bpp2[i])
        plt.subplot(5, NumImagesToShow, 2*NumImagesToShow + i + 1).set_title(title, fontsize=10)
        if i==0:
            plt.subplot(5, NumImagesToShow, 2*NumImagesToShow + i + 1).text(-50, 64, 'AE2')        
        plt.imshow(decoded_imgsQ2[i, :, :, :], interpolation='nearest')
        plt.axis(False)
    for i in range(NumImagesToShow):
        psnr = ssim(xtest[i, :, :, 0], decoded_imgsQ3[i, :, :, 0],data_range=1.0)
        psnr = psnr + ssim(xtest[i, :, :, 1], decoded_imgsQ3[i, :, :, 1],data_range=1.0)
        psnr = psnr + ssim(xtest[i, :, :, 2], decoded_imgsQ3[i, :, :, 2],data_range=1.0)
        psnr=psnr/3.0
        #title = '%2.2f %2.2f' % (psnr, bpp2[i])
        title = 'Q=%2.2f bpp=%2.2f' % (psnr, bpp3[i])
        plt.subplot(5, NumImagesToShow, NumImagesToShow + i + 1).set_title(title, fontsize=10)
        if i==0:
            plt.subplot(5, NumImagesToShow, NumImagesToShow + i + 1).text(-50, 64, 'res')
        plt.imshow(decoded_imgsQ3[i, :, :, :], interpolation='nearest')
        plt.axis(False)
    jbpp, jssim = [], []
    for i in range(NumImagesToShow):
        JPEGQP,JPEGrealbpp, JPEGrealpsnr = JPEGRDSingleImage(xtest[i, :, :, :], bpp[i],i)
        jbpp.append(JPEGrealbpp)
        jssim.append(JPEGrealpsnr)
        JPEGfilename = 'image%i.jpeg' % i
        JPEGimage = Image.open(JPEGfilename)
        #title = '%2.2f %2.2f' % (JPEGrealpsnr,JPEGrealbpp)
        title = 'Q=%2.2f bpp=%2.2f' % (JPEGrealpsnr,JPEGrealbpp)
        plt.subplot(5, NumImagesToShow, 4*NumImagesToShow + i + 1).set_title(title, fontsize=10)
        if i==0:
            plt.subplot(5, NumImagesToShow, 4*NumImagesToShow + i +  1).text(-50, 64, 'JPEG')        
        plt.imshow(JPEGimage, interpolation='nearest')
        plt.axis(False)

    # resdf = pd.DataFrame()
    # for r, ims, name in zip([bpp2, bpp3, jbpp],
    #                         [decoded_imgsQ2, decoded_imgsQ3, jssim],
    #                         ['base', 'solution', 'jpeg']):
    #     ssims = []
    #     if name == 'jpeg':
    #         ssims = ims
    #     else:
    #         for i in range(NumImagesToShow):
    #             # psnr.append(PSNR(xtest[i, :, :, :], im))
    #             psnr = ssim(xtest[i, :, :, 0], ims[i, :, :, 0],data_range=1.0)
    #             psnr = psnr + ssim(xtest[i, :, :, 1], ims[i, :, :, 1],data_range=1.0)
    #             psnr = psnr + ssim(xtest[i, :, :, 2], ims[i, :, :, 2],data_range=1.0)
    #             ssims.append(psnr/3.)
    #     ssims = numpy.array(ssims)
    #     r = numpy.array(r)
    #     resdf[f'bpp_{name}'] = r
    #     resdf[f'ssim_{name}'] = ssims

    # resdf.to_csv(f'res/result{b}.csv')

    plt.show()
