library(keras)
library(tensorflow)
library(tidyverse)
library(imager)
library(reticulate)

use_condaenv("py3.6", required = TRUE)

img_folder <- 'data/img_align_celeba/'
partition <- 'list_eval_partition.csv'
class_labels <- 'list_attr_celeba.csv' 

#load class labels
attribute_labels <- read.csv(class_labels, header=TRUE, stringsAsFactors = FALSE)


#replacing -1 with 0
attribute_labels[attribute_labels == -1] <- 0

#load recommended file split for train, test & validation
image_split <- read.csv(partition, header=TRUE)

#checking for null values
any(is.na(attribute_labels))

# ### Examples of Images in Dataset
example_img <- paste0("data/img_align_celeba/", attribute_labels[12, 1])


im<-load.image(example_img)
plot(im)

height(im)
width(im)


# calculating relative frequencies of attributes
attr = attribute_labels %>%
  pivot_longer(!image_id, names_to = "attribute")


attr <- attr %>%
        group_by(attribute) %>%
        summarise(freq = sum(value)/nrow(attribute_labels)*100) %>%
        arrange(-freq) 
  


p <- ggplot2::ggplot(data = attr,
                     mapping = aes(x = reorder(attribute, freq), y = freq)) +
     geom_bar(stat = 'identity') + 
     coord_flip()
p



### Data Preparation


#extract male flag for building as predictor for male vs female
cols <-  c('image_id', 'Male')
cols2 <- c('image_id', 'partition')
male_labels = attribute_labels[cols]
image_split = image_split[cols2]

#join image splits to attribute labels
#so we can partition the data
image_attributes = merge(image_split,
                         male_labels, 
                         by = 'image_id')

#split into training and test
#create balanced and more manageable set 
#by only select 20000 male records and 20000 non-male records for train
# nd for test  2000 male records and 2000 female records

male_train = image_attributes %>% filter (Male == 1, partition == 0) %>% head(10000)
female_train = image_attributes %>% filter (Male == 0, partition == 0) %>% head(10000)

dir.create("data/Train")
male_dir <- "data/Train/Male"
female_dir <-"data/Train/Female"
dir.create(male_dir)
dir.create(female_dir)

files = as.character(male_train$image_id)

for (file in files){

  path <- paste0(img_folder, file)
  file.copy(from = path, to = male_dir)

}

files = as.character(female_train$image_id)

for (file in files){

  path <- paste0(img_folder, file)
  file.copy(from = path, to = female_dir)

}


male_test = image_attributes %>% filter (Male == 1, partition == 2) %>% head(1000)
female_test = image_attributes %>% filter (Male == 0, partition == 2) %>% head(1000)

dir.create("data/Test")
male_dir <- "data/Test/Male"
female_dir <-"data/Test/Female"
dir.create(male_dir)
dir.create(female_dir)

files = as.character(male_test$image_id)

for (file in files){

  path <- paste0(img_folder, file)
  file.copy(from = path, to = male_dir)

}

files = as.character(female_test$image_id)

for (file in files){

  path <- paste0(img_folder, file)
  file.copy(from = path, to = female_dir)

}


attribute_list <- c('Male', 'Female')
# number of output classes (i.e. fruits)
output_n <- length(attribute_list)

# image size to scale down to (original images are 100 x 100 px)
img_width <- 218
img_height <- 178
target_size <- c(img_width, img_height)

# RGB = 3 channels
channels <- 3

# path to image folders
train_image_files_path <- "data/Train/"
valid_image_files_path <- "data/Test"

#Loading images
#The handy image_data_generator() and flow_images_from_directory() functions can be used to load images from a directory. If you want to use data augmentation, you can directly define how and in what way you want to augment your images with image_data_generator. Here I am not augmenting the data, I only scale the pixel values to fall between 0 and 1.

library(keras)

# optional data augmentation
train_data_gen = image_data_generator(
  rescale = 1/255 #,
  #rotation_range = 40,
  #width_shift_range = 0.2,
  #height_shift_range = 0.2,
  #shear_range = 0.2,
  #zoom_range = 0.2,
  #horizontal_flip = TRUE,
  #fill_mode = "nearest"
)


# Validation data shouldn't be augmented! But it should also be scaled.
valid_data_gen <- image_data_generator(
  rescale = 1/255
)  
#Now we load the images into memory and resize them.

# training images
train_image_array_gen <- flow_images_from_directory(train_image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = attribute_list,
                                                    seed = 42)
# validation images
valid_image_array_gen <- flow_images_from_directory(valid_image_files_path, 
                                                    valid_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = attribute_list,
                                                    seed = 42)
cat("Number of images per class:")
## Number of images per class:
table(factor(train_image_array_gen$classes))

train_image_array_gen$class_indices

classes_indices <- train_image_array_gen$class_indices
save(classes_indices, file = "classes_indices.RData")


# number of training samples
train_samples <- train_image_array_gen$n
# number of validation samples
valid_samples <- valid_image_array_gen$n

# define batch size and number of epochs
batch_size <- 32
epochs <- 10




# initialise model
model <- keras_model_sequential()

# add layers
model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", input_shape = c(img_width, img_height, channels)) %>%
  layer_activation("relu") %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same") %>%
  layer_activation_leaky_relu(0.5) %>%
  layer_batch_normalization() %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(100) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  
  # Outputs from dense layer are projected onto output layer
  layer_dense(output_n) %>% 
  layer_activation("softmax")

 # compile
 model %>% compile(
   loss = "categorical_crossentropy",
   optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
   metrics = "accuracy"
 )




# fit
hist <- model %>% fit_generator(
  # training data
  train_image_array_gen,
  
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  
  # validation data
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  
  # print progress
  verbose = 2,
  callbacks = list(
    # save best model after every epoch
    callback_model_checkpoint("checkpoints.h5", save_best_only = TRUE),
    # only needed for visualising with TensorBoard
    callback_tensorboard(log_dir = "/keras/logs")
  )
)


plot(hist)

tensorboard("/keras/logs")


tf$keras$models$save_model(model, './model1/')

model <- tf$keras$models$load_model('./model1')

test_image <- 


library(jpeg)
a <- jpeg::readJPEG("D:/FacialFeatureIdentification/data/img_align_celeba/IMG_7551.jpg")
dim(a)

a <- imager::load.image("D:/FacialFeatureIdentification/data/Pred/IMG_7553.jpg")
b <- resize(a, 218, 178, 1, 3)
imager::save.image(b, "D:/FacialFeatureIdentification/data/Pred/pred3.jpg")
a <- jpeg::readJPEG("D:/FacialFeatureIdentification/data/Pred/pred1.jpg")
dim(a)
print(a)

pred_data <- tf$keras$preprocessing$image$img_to_array(a)
pred_data <- array_reshape(pred_data, c(1, 218, 178, 3))



model %>% predict_classes(pred_data)











  
train_data = np.array(train_data)
print('train_data shape:', train_data.shape)


# 
# # ### Training & Testing Labels
# 
# # In[71]:
# 
# 
# train['Male'] = np.where(train['Male'] == 0, 0, 1)
# test['Male'] = np.where(test['Male'] == 0, 0, 1)
# 
# train_label = train['Male'].astype(int).tolist()
# test_label = test['Male'].astype(int).tolist()
# 
# fig, axs = plt.subplots(1,2)
# train['Male'].value_counts().plot.bar(ax=axs[0])
# test["Male"].value_counts().plot.bar(ax=axs[1])
# 
# 
# # ## Extracting Histogram of Oriented Gradients (HOG) Features
# 
# # In[72]:
# 
# 
# # extracting HOG features for 1 image from testing set
# fd, hog_image = hog(test_data[0,:,:], orientations=8, pixels_per_cell=(8, 8),
#                     cells_per_block=(1, 1), visualize=True, multichannel=True)
# print('HOG features:', fd.shape)
# 
# 
# # ### HOG Features for Training Images
# 
# # In[73]:
# 
# 
# train_hog = np.empty(shape=(train_data.shape[0],len(fd)))
# for i in range(0,train_data.shape[0]):
#   fd = hog(train_data[i,:,:], orientations=8, pixels_per_cell=(8, 8),
#            cells_per_block=(1, 1), visualize=False, multichannel=True)
# train_hog[i,:] = fd
# print(train_hog.shape)
# 
# 
# # ### HOG Features for Testing Images
# 
# # In[78]:
# 
# 
# test_hog = np.empty(shape=(test_data.shape[0],4752))
# for i in range(0,test_data.shape[0]):
#   fd = hog(test_data[i,:,:], orientations=8, pixels_per_cell=(8, 8),
#            cells_per_block=(1, 1), visualize=False, multichannel=True)
# test_hog[i,:] = fd
# print(test_hog.shape)
# 
# 
# # ### Displaying the Original Image and HOG Descriptors
# 
# # In[88]:
# 
# 
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
# 
# ax1.axis('off')
# ax1.imshow(test_data[0,:,:])
# ax1.set_title('Input image')
# 
# #rescale histogram for better display
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
# 
# ax2.axis('off')
# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
# ax2.set_title('HOG Features')
# plt.savefig('HOG.png')
# 
# # ### Feature Model

# In[80]:


feature_model = XGBClassifier()
feature_model.fit(train_hog, train_label)

pred = feature_model.predict(test_hog)

predictions = [round(value) for value in pred]
accuracy = accuracy_score(test_label, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100))

cm = confusion_matrix(test_label, predictions)
print(cm)


# ## Constructing CNN Model

# ### CNN Hyperparameters

# In[81]:


NB_EPOCH = 100
BATCH_SIZE = 50
VERBOSE = 1
NB_CLASSES = 2 # number of classes
OPTIMIZER = Adam()
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2
METRICS =['accuracy']
LOSS = 'binary_crossentropy'
IMAGE_WIDTH= img.shape[0]
IMAGE_HEIGHT= img.shape[1]
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS= img.shape[2]
INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)


# ### Building CNN Model

# In[82]:


cnn_project=Sequential()

#2D CNN layer with 32 kernels
NUM_KERNELS = 32
cnn_project.add(Conv2D(NUM_KERNELS,kernel_size=(3,3),padding="valid",kernel_initializer='glorot_uniform',input_shape = INPUT_SHAPE))
cnn_project.add(Activation("relu"))

#2D pooling layer, max pooling
cnn_project.add(MaxPooling2D(pool_size=(2,2)))
cnn_project.add(BatchNormalization())
cnn_project.add(Dropout(0.2))

#2D CNN layer with 64 kernels
NUM_KERNELS = 64
cnn_project.add(Conv2D(NUM_KERNELS,kernel_size=(3,3),padding="valid",kernel_initializer='glorot_uniform'))
cnn_project.add(Activation("relu"))

#2D pooling layer, max pooling
cnn_project.add(MaxPooling2D(pool_size=(2,2)))
cnn_project.add(BatchNormalization())
cnn_project.add(Dropout(0.3))

#dense layer with 100 nodes
cnn_project.add(Flatten())
cnn_project.add(Dense(100))
cnn_project.add(Activation('relu'))
cnn_project.add(BatchNormalization())
cnn_project.add(Dropout(0.5))

#outut layer with 1 nodes, sigmoid activation
cnn_project.add(Dense(1)) # NB_CLASSES = 2
cnn_project.add(Activation("sigmoid"))
cnn_project.compile(loss=LOSS, optimizer = Adam(lr=0.05), metrics =METRICS)

print(cnn_project.summary())


# In[83]:


#normalize between 0 and 1
train_data = train_data/255
test_data  = test_data/255


# ### Compiling & Training Model

# In[84]:


filepath='cnn_project.hdf5'

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)

cnn_project_fit = cnn_project.fit(train_data,train_label,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,
                                  validation_split = VALIDATION_SPLIT,callbacks=[checkpoint,early_stopping_monitor])


# ### Test Accuracy

# In[86]:


def GetAccuracy(model,predictors,response):
  from sklearn.metrics import accuracy_score

#convert categorical to integer class labels
y_classes = [np.argmax(y, axis=None, out=None) for y in response]
pred_class = model.predict_classes(predictors)
acc = accuracy_score(y_classes,pred_class)
return acc

cnn_project_accuracy = GetAccuracy(cnn_project, test_data, test_label)

pred_class = cnn_project.predict_classes(test_data)
print(accuracy_score(test_label, pred_class))
cm = confusion_matrix(test_label, pred_class)

print(cm)


# ### Plotting Loss & Accuracy

# In[87]:


def plotHistory(Tuning):
  fig, axs = plt.subplots(1,2,figsize=(15,5))
axs[0].plot(Tuning.history['loss'])
axs[0].plot(Tuning.history['val_loss'])
axs[0].set_title('loss vs epoch')
axs[0].set_ylabel('loss')
axs[0].set_xlabel('epoch')
axs[0].legend(['train', 'vali'], loc='upper left')

axs[1].plot(Tuning.history['accuracy'])
axs[1].plot(Tuning.history['val_accuracy'])
axs[1].set_title('accuracy vs epoch')
axs[1].set_ylabel('accuracy')
axs[1].set_xlabel('epoch')
axs[1].set_ylim([0.0,1.0])
axs[1].legend(['train', 'vali'], loc='upper left')
plt.show(block = False)
plt.savefig('accuracy.png')

plotHistory(cnn_project_fit)






attribute_list <- c('Male', 'Female')
# number of output classes (i.e. fruits)
output_n <- length(fruit_list)

# image size to scale down to (original images are 100 x 100 px)
img_width <- 20
img_height <- 20
target_size <- c(img_width, img_height)

# RGB = 3 channels
channels <- 3

# path to image folders
train_image_files_path <- "/Users/shiringlander/Documents/Github/DL_AI/Tutti_Frutti/fruits-360/Training/"
valid_image_files_path <- "/Users/shiringlander/Documents/Github/DL_AI/Tutti_Frutti/fruits-360/Validation/"
Loading images
The handy image_data_generator() and flow_images_from_directory() functions can be used to load images from a directory. If you want to use data augmentation, you can directly define how and in what way you want to augment your images with image_data_generator. Here I am not augmenting the data, I only scale the pixel values to fall between 0 and 1.

# optional data augmentation
train_data_gen = image_data_generator(
  rescale = 1/255 #,
  #rotation_range = 40,
  #width_shift_range = 0.2,
  #height_shift_range = 0.2,
  #shear_range = 0.2,
  #zoom_range = 0.2,
  #horizontal_flip = TRUE,
  #fill_mode = "nearest"
)

# Validation data shouldn't be augmented! But it should also be scaled.
valid_data_gen <- image_data_generator(
  rescale = 1/255
)  
Now we load the images into memory and resize them.

# training images
train_image_array_gen <- flow_images_from_directory(train_image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = fruit_list,
                                                    seed = 42)

# validation images
valid_image_array_gen <- flow_images_from_directory(valid_image_files_path, 
                                                    valid_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = fruit_list,
                                                    seed = 42)
cat("Number of images per class:")
## Number of images per class:
table(factor(train_image_array_gen$classes))