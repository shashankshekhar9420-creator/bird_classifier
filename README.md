# Common Indian Backyard Bird Classifier
In this project we'll train a model to tell apart 5 common Indian birds:
- Red-wattled Lapwing
- Red-vented Bulbul
- Oriental Magpie Robin
- Greater Coucal
- Indian Grey Hornbill

Same approach as the fast.ai `is-it-a-bird` notebook — search images → clean → train → predict.
<br>

In this we are project not building a bird classifier- we are building a general image classification system, and birds are simply our data set.

### Where does the neural network learn from?
- the neural network learns from looking at many examples
- hence out dataset contains lots and lots of images
- the folder name would act as the label and not the image name

### Where do these images come from?
- instead of collecting thousands of images ourselves, we use a search engine
- the notebook uses: `search_images_ddg()`
- let's understand through an example: You ask for "red-wattled lapwing" -> duckduckgo -> returns urls -> download images -> save locally
- so after downloading the images will be organized into folders, and the folder names act as the labels

### Why do we need multiple images?
- suppose we train only on one image: red-wattled lapwing standing on grass in the morning
- then the model would not be able to identify a lapwing flying or any other such case, because it never learnt about it
- hence we want variation in our images- this would lead the network to discover what is in common and hence it will be able to identify

### Why do we need to clean the data set?
- search engine makes mistakes
- if we train on incorrect lables, the model learns incorrect mappings
- for example when we search for a bird, we might get images of maybe a zoo sign, logo, drawing, incorrect images- these are the mistakes that the seach engine could make

### Some thing to do before training: Resizing
- every image has a different size, for example: 640 x 480, 512 x 512, 1920 x 1080, etc
- the neural network cannot process variable-sized tensors in a batch
- hence we resize: we resize all the images to have identical dimensions

### Pixel become numbers
- a color image is simply a three dimensional array
- for example: 224 x 224 x3
- hence images are converted into tensors
<table>
  <tr>
    <td><img width="400" alt="image" src="https://github.com/user-attachments/assets/5c7149e7-3fc3-471b-aefd-f9eae425a9fb" /> </td>
    <td><img width="400" alt="image" src="https://github.com/user-attachments/assets/c618e96d-1d57-4f11-b83b-706cd7c6fc54" /> </td>
  </tr>
</table>

### The Neural Network
- create training and validation datasets
- feed batches of images (which are now converted into tensors) into a pretrained CNN (Convolutional Neural Network)
- CNN extracts features and the classifier predicts the class
- CNN -> feature extractor -> classifier -> 5 outputs
- those five outputs correspond to Lapwing, Bulbul, Robin, Coucal and Hornbill
- compute prediction scores for the five bird classes
- compare predictions with the true labels
- compute loss and backpropagation updates model parameters
- this repeats for many epochs for better predictions
- initally these outputs are random
- training gradually adjusts millions of parameters so that the correct class recieves the highest score

###  Prediction
Then we will be able to use the trained model to classify new bird images.
Suppose the network outputs:
```
Lapwing      0.2
Bulbul       4.8
Robin       -1.0
Coucal       2.1
Hornbill    -0.8
```

The largest score wins. Hence the prediction for this example would be bulbul.
