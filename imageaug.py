from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
import os,cv2
#Data augmentation parameters
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

PATH = os.getcwd()
# Define data path
data_path = PATH + '/surprise'
img_list=os.listdir(data_path)
for img1 in img_list:
    img=cv2.imread(data_path + '/'+ img1 )
    #img = load_img('data/test/2.jpg')  #taking input image
    x = img_to_array(img)  # creating a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # converting to a Numpy array with shape (1, 3, 150, 150)

    i = 0 #creating the required images using a loop
    for batch in datagen.flow(x,save_to_dir='surprise', save_prefix='aug', save_format='jpeg'):
        i += 1
        if i > 5:
            break 


    