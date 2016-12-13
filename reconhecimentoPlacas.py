import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys

nomeDoArquivo = str(sys.argv[1])
#print "nomeDoArquivo"
cv2.waitKey(0)

img = cv2.imread(nomeDoArquivo)
img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


imgGauss = cv2.GaussianBlur(img_cinza, (3, 3), 0)

sobelx = cv2.Sobel(imgGauss, cv2.CV_8U, 1, 0, ksize = 3)

#Otsu thresholding
_,th2 = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#Morphological Closing
se = cv2.getStructuringElement(cv2.MORPH_RECT,(30,2))

closing = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, se)

_,contours,_ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt in contours:
    rect = cv2.minAreaRect(cnt)  
    box = cv2.boxPoints(rect) 
    box = np.int0(box)  
    cv2.drawContours(img, [box], 0, (0,255,0),2)

#validate a contour. We validate by estimating a rough area and aspect ratio check.
def validate(cnt):    
    rect = cv2.minAreaRect(cnt)  
    box = cv2.boxPoints(rect) 
    box = np.int0(box)  
    output = False
    width = rect[1][0]
    height = rect[1][1]
    if ((width != 0) & (height != 0)):
        if (((height/width > 3) & (height > width)) | ((width/height > 3) & (width > height))):
            if((height * width < 20000) & (height * width > 5000)): 
                output = True
    return output

#Lets draw validated contours with red.
for cnt in contours:
    if validate(cnt):
        rect = cv2.minAreaRect(cnt)  
        box = cv2.boxPoints(rect) 
        box = np.int0(box)  
        cv2.drawContours(img, [box], 0, (0,0,255),2)

# defining a function doing this will come 
def generate_seeds(centre, width, height):
    minsize = int(min(width, height))
    seed = [None] * 10
    for i in range(10):
        random_integer1 = np.random.randint(1000)
        random_integer2 = np.random.randint(1000)
        seed[i] = (centre[0] + random_integer1 % int(minsize / 2) - int(minsize / 2), centre[1] + random_integer2 % int(minsize / 2) - int(minsize / 2))
    return seed

def generate_mask(image, seed_point):
    h = img.shape[0]
    w = img.shape[1]
    #OpenCV wants its mask to be exactly two pixels greater than the source image.
    mask = np.zeros((h+2, w+2), np.uint8)
    #We choose a color difference of (50,50,50). Thats a guess from my side.
    lodiff = 100
    updiff = 100
    connectivity = 8
    newmaskval = 255
    flags = connectivity + (newmaskval << 8) + cv2.FLOODFILL_FIXED_RANGE + cv2.FLOODFILL_MASK_ONLY
    _ = cv2.floodFill(image, mask, seed_point, (255, 0, 0), (lodiff, lodiff, lodiff), (updiff, updiff, updiff), flags)
    return mask

# we will need a fresh copy of the image so as to draw masks.
img_mask = cv2.imread(nomeDoArquivo)

# for viewing the different masks later
mask_list = []

for cnt in contours:
    if validate(cnt):
        rect = cv2.minAreaRect(cnt) 
        centre = (int(rect[0][0]), int(rect[0][1]))
        width = rect[1][0]
        height = rect[1][1]
        seeds = generate_seeds(centre, width, height)
        
        #now for each seed, we generate a mask
        for seed in seeds:
            # plot a tiny circle at the present seed.
            cv2.circle(img, seed, 1, (0,0,255), -1)
            # generate mask corresponding to the current seed.
            mask = generate_mask(img_mask, seed)
            mask_list.append(mask)   

validated_masklist = []
for mask in mask_list:
    contour = np.argwhere(mask.transpose() == 255)
    if validate(contour):
        validated_masklist.append(mask)

try:
    assert (len(validated_masklist) != 0)
except AssertionError:
    print ("Nenhuma placa encontrada")

def rmsdiff(im1, im2):
    diff = im1-im2
    output = False
    if np.sum(abs(diff)) / float(min(np.sum(im1), np.sum(im2))) < 0.01:
        output = True
    return output

# final masklist will be the final list of masks we will be working on.
final_masklist = []
index = []
for i in range(len(validated_masklist) - 1):
    for j in range(i + 1, len(validated_masklist)):
        if rmsdiff(validated_masklist[i], validated_masklist[j]):
            index.append(j)
for mask_no in list(set(range(len(validated_masklist)))-set(index)):
    final_masklist.append(validated_masklist[mask_no])

cropped_images = []
for mask in final_masklist:
    contour = np.argwhere(mask.transpose() == 255)
    rect = cv2.minAreaRect(contour)
    width = int(rect[1][0])
    height = int(rect[1][1])
    centre = (int(rect[0][0]), int(rect[0][1]))
    box = cv2.boxPoints(rect) 
    box = np.int0(box)
    #check for 90 degrees rotation
    if ((width / float(height)) > 1):
        # crop a particular rectangle from the source image
        cropped_image = cv2.getRectSubPix(img_mask, (width, height), centre)
    else:
        # crop a particular rectangle from the source image
        cropped_image = cv2.getRectSubPix(img_mask, (height, width), centre)

    # convert into grayscale
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    # equalize the histogram
    cropped_image = cv2.equalizeHist(cropped_image)
    # resize to 260 cols and 63 rows. (Just something I have set as standard here)
    cropped_image = cv2.resize(cropped_image, (260, 63))
    cropped_images.append(cropped_image)


#_ = plt.subplots_adjust(hspace = 0.000)
number_of_subplots = len(cropped_images)


#porque gera quantidades diferentes de imagens?
for i in range (len(cropped_images)): 
    cv2.imwrite("Resultados\Placa"+str(i)+".jpg", cropped_images[i])
    cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()

