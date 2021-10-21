# _Generate an Abstract Painting from Signatures - AI Artathon 2.0_

## Project General Idea

<img width="900" alt="Screen Shot 2021-10-21 at 9 18 39 PM" src="https://user-images.githubusercontent.com/24601296/138340585-1afeb042-5b10-4f4b-891d-10d8ca614dbc.png">


## Development Requirment
- Anaconda Framework
- Python 3.0
- Colab

## Development Process
- Collecte a dataset of signatures
- Removing the signature’s background by using OpenCV library
- Training the GAN model to generate new signature
- Using human face as input to Pix2Pix Model which used the trained signature to generate the abstract art painting

## 1. Collecting Signatures' Dataset
- https://www.kaggle.com/divyanshrai/handwritten-signatures (contains 1020 signatures)
- https://cedar.buffalo.edu/NIJ/data (contains 1320 signatures)

## 2. Removing the signature’s background
```sh
#path for uncleand signature folder
uncleaned_folder_path = "//Users//raghadbajahlan//Desktop//uncleaned_signature//"
#path for cleand signature folder
cleaned_folder_path = "//Users//raghadbajahlan//Desktop//cleaned_signature//"

#access to images
onlyfiles = [ f for f in listdir(uncleaned_folder_path) if isfile(join(uncleaned_folder_path,f)) ]
#create numpy array of images
images = numpy.empty(len(onlyfiles), dtype=object)

#start cleaning
for n in range(0, len(onlyfiles)):
    #read image
    images[n] = cv2.imread( join(uncleaned_folder_path,onlyfiles[n]))
    #convert color system
    gray = cv2.cvtColor(images[n], cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Bitwise-and and color background white
    result = cv2.bitwise_and(images[n], images[n], mask=thresh)
    result[thresh==0] = [255,255,255]
    
    #save cleaned fold
    cv2.imwrite(join(cleaned_folder_path,onlyfiles[n]),result)
```

## 3. Training GAN Model

## 4. Generate the Abstract Art Painting from Human Face
 
 <img width="656" alt="Screen Shot 2021-10-21 at 9 47 29 PM" src="https://user-images.githubusercontent.com/24601296/138340643-cf288421-fce4-4c23-b14e-96d6ca1b2227.png">




## _Sewar Team Members_
✨Raghad Bajhlan | AI Devloper and Artist
✨Salwa Alhajjaji | AI Devloper 


