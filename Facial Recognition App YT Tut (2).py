#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install --upgrade pip


# In[2]:


pip install tensorflow==2.17.0


# In[3]:


get_ipython().system('pip install opencv-python matplotlib')


# In[4]:


pip install matplotlib


# In[1]:


# Import standard dependencies
import cv2
import os
import uuid
import random
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf


# In[3]:


#Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# In[4]:


# Setup paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')


# In[5]:


# Make the directories
os.makedirs(POS_PATH)
os.makedirs(NEG_PATH)
os.makedirs(ANC_PATH)


# In[6]:


# Move LFW Images to the following repository data/negative
for directory in os.listdir('lfw-deepfunneled'):
    for file in os.listdir(os.path.join('lfw-deepfunneled', directory)):
        EX_PATH = os.path.join('lfw-deepfunneled', directory, file)
        NEW_PATH = os.path.join(NEG_PATH, file)
        os.replace(EX_PATH, NEW_PATH)


# In[7]:


# Import uuid library to generate unique image names
import uuid


# In[ ]:


# Establish a connection to the webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    # Cut down frame to 250x250px
    frame = frame[120:120+250,200:200+250, :]

    # Collect anchors
    if cv2.waitKey(1) & 0XFF == ord('a'):
        # Create the unique file path
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Write out anchor image
        cv2.imwrite(imgname, frame)
        
    # Collect positives
    if cv2.waitKey(1) & 0XFF == ord('p'):
        # Create the unique file path
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Write out anchor image
        cv2.imwrite(imgname, frame)
        
    # Show image back to screen
    cv2.imshow('Image Collection', frame)

    # Breaking gracefully
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
# Release the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()


# In[6]:


def data_aug(img):
    data = []
    for i in range(12):
        img_aug = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1,2))
        img_aug = tf.image.stateless_random_contrast(img_aug, lower=0.6, upper=1, seed=(1,3))
        # img_aug = tf.image.stateless_random_crop(img_aug, size=(20,20,3), seed=(1,2))
        img_aug = tf.image.stateless_random_flip_left_right(img_aug, seed=(np.random.randint(100), np.random.randint(100)))
        img_aug = tf.image.stateless_random_jpeg_quality(img_aug, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100), np.random.randint(100)))
        img_aug = tf.image.stateless_random_saturation(img_aug, lower=0.9, upper=1, seed=(np.random.randint(100), np.random.randint(100)))

        data.append(img_aug)

    return data


# In[ ]:


import os
import uuid
import cv2

# Assumes ANC_PATH and POS_PATH are already defined
ANC_PATH = os.path.join('data', 'anchor')
POS_PATH = os.path.join('data', 'positive')

AUG_ANCHOR_PATH = os.path.join('data', 'augmented_anchor')
AUG_POS_PATH = os.path.join('data', 'augmented_positive')

# Create directories to save augmented images
os.makedirs(AUG_ANCHOR_PATH, exist_ok=True)
os.makedirs(AUG_POS_PATH, exist_ok=True)

# Augmentation function (make sure this is defined somewhere above)
def data_aug(img):
    data = []
    for i in range(16):
        img_aug = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1, 2))
        img_aug = tf.image.stateless_random_contrast(img_aug, lower=0.6, upper=1, seed=(1, 3))
        img_aug = tf.image.stateless_random_flip_left_right(img_aug, seed=(np.random.randint(100), np.random.randint(100)))
        img_aug = tf.image.stateless_random_jpeg_quality(img_aug, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100), np.random.randint(100)))
        img_aug = tf.image.stateless_random_saturation(img_aug, lower=0.9, upper=1, seed=(np.random.randint(100), np.random.randint(100)))
        data.append(img_aug)
    return data

# 🔹 Augment Anchor Images
for file_name in os.listdir(ANC_PATH):
    img_path = os.path.join(ANC_PATH, file_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
    augmented_images = data_aug(img)
    for image in augmented_images:
        cv2.imwrite(
            os.path.join(AUG_ANCHOR_PATH, '{}.jpg'.format(uuid.uuid1())),
            image.numpy()
        )

# 🔹 Augment Positive Images
for file_name in os.listdir(POS_PATH):
    img_path = os.path.join(POS_PATH, file_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
    augmented_images = data_aug(img)
    for image in augmented_images:
        cv2.imwrite(
            os.path.join(AUG_POS_PATH, '{}.jpg'.format(uuid.uuid1())),
            image.numpy()
        )


# In[58]:


anchor = tf.data.Dataset.list_files(ANC_PATH+'/*.jpg').take(3000)
positive = tf.data.Dataset.list_files(POS_PATH+'/*.jpg').take(3000)
negative = tf.data.Dataset.list_files(NEG_PATH+'/*.jpg').take(3000)


# In[59]:


def preprocess(file_path):

    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)

    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))
    # Scale image to be between 0 and 1
    img = img / 255.0

    #Return image
    return img


# In[60]:


positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)


# In[61]:


def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)


# In[62]:


# Build dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=10000)


# In[63]:


# Training partition
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)


# In[64]:


# Testing partition
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


# In[65]:


def make_embedding():
    inp = Input(shape=(100,100,3), name = 'input_image')

    # First block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)

    # Second block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)

    # Third block
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)

    # Final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')


# In[66]:


embedding = make_embedding()


# In[67]:


embedding.summary()


# In[68]:


# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(tf.math.subtract(input_embedding, validation_embedding))


# In[69]:


l1 = L1Dist()


# In[70]:


l1(anchor_embedding, validation_embedding)


# In[71]:


input_image = Input(name='input_img', shape=(100,100,3))
validation_image = Input(name='validation_img', shape=(100,100,3))


# In[72]:


inp_embedding = embedding(input_image)
val_embedding = embedding(validation_image)


# In[73]:


siamese_layer = L1Dist()


# In[74]:


distances = siamese_layer(inp_embedding, val_embedding)


# In[75]:


classifier = Dense(1, activation='sigmoid')(distances)


# In[76]:


classifier


# In[77]:


siamese_network = Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


# In[78]:


siamese_network.summary()


# In[79]:


def make_siamese_model():

    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100,100,3))

    # Validation image in the network
    validation_image = Input(name='validation_img', shape=(100,100,3))

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    # Classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


# In[80]:


siamese_model = make_siamese_model()


# In[81]:


siamese_model.summary()


# In[82]:


binary_cross_loss = tf.losses.BinaryCrossentropy()


# In[83]:


opt = tf.keras.optimizers.Adam(1e-4)


# In[84]:


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)


# In[85]:


@tf.function
def train_step(batch):

    # Record all of our operations
    with tf.GradientTape() as tape:
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]

        # Forward pass
        yhat = siamese_model(X, training=True)
        y = tf.reshape(y, tf.shape(yhat))
        # Calculate loss
        loss = binary_cross_loss(y, yhat)
    print(loss)

    #Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    #Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))


    # Return loss
    return loss


# In[86]:


from tensorflow.keras.metrics import Precision, Recall


# In[87]:


# Initialize lists to store metrics for each epoch
loss_history = []
precision_history = []
recall_history = []

def train(data, EPOCHS):
    # Loop through epochs
    for epoch in range(1, EPOCHS + 1):
        print(f'\n Epoch {epoch}/{EPOCHS}')
        progbar = tf.keras.utils.Progbar(len(data))

        # Creating a metric object
        r = Recall()
        p = Precision()

        # Loop through each batch
        for idx, batch in enumerate(data):
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat)
            progbar.update(idx + 1)

        # Save metrics after each epoch
        loss_history.append(loss.numpy())      # Collect loss
        precision_history.append(p.result().numpy())  # Collect precision
        recall_history.append(r.result().numpy())     # Collect recall

        print(f"Loss: {loss.numpy()}, Precision: {p.result().numpy()}, Recall: {r.result().numpy()}")

        # Save checkpoints
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


# In[88]:


EPOCHS = 50


# In[89]:


train(train_data, EPOCHS)


# In[90]:


import pandas as pd
import matplotlib.pyplot as plt

#Performance table and graph
metrics_df = pd.DataFrame({
    'Epoch': list(range(1, len(loss_history)+1)),
    'Loss': loss_history,
    'Precision': precision_history,
    'Recall': recall_history
})

# Show table
display(metrics_df)

# ====================
# 📈 Plot Performance
# ====================
plt.figure(figsize=(12, 6))
plt.plot(metrics_df['Epoch'], metrics_df['Loss'], label='Loss')
plt.plot(metrics_df['Epoch'], metrics_df['Precision'], label='Precision')
plt.plot(metrics_df['Epoch'], metrics_df['Recall'], label='Recall')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.title('Model Performance Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[91]:


# Import metric calculations
from tensorflow.keras.metrics import Precision, Recall


# In[92]:


# Get a batch of test data
test_input, test_val, y_true = test_data.as_numpy_iterator().next()


# In[93]:


y_hat = siamese_model.predict([test_input, test_val])


# In[94]:


# Post processing the results 
[1 if prediction > 0.5 else 0 for prediction in y_true ]


# In[95]:


y_true


# In[96]:


# Creating a metric object
m = Recall()

# Calculating the recall value
m.update_state(y_true, y_hat)

# Return Recall Result
m.result().numpy()


# In[ ]:


r = Recall()
p = Precision()

for test_input, test_val, y_true in test_data.as_numpy_iterator():
    yhat = siamese_model.predict([test_input, test_val])
    r.update_state(y_true, yhat)
    p.update_state(y_true,yhat) 

print(r.result().numpy(), p.result().numpy())


# In[ ]:


# Set plot size 
plt.figure(figsize=(10,8))

# Set first subplot
plt.subplot(1,2,1)
plt.imshow(test_input[0])

# Set second subplot
plt.subplot(1,2,2)
plt.imshow(test_val[0])

# Renders cleanly
plt.show()


# In[ ]:


# Save weights
siamese_model.save('siamesemodelv7.h5')


# In[ ]:


L1Dist


# In[ ]:


# Reload model 
siamese_model = tf.keras.models.load_model('siamesemodelv7.h5', 
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy}, compile = False)


# In[ ]:


# Make predictions with reloaded model
siamese_model.predict([test_input, test_val])


# In[ ]:


# View model summary
siamese_model.summary()


# In[ ]:


def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_images', image))
        
        # Make Predictions 
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    
    # Detection Threshold: Metric above which a prediciton is considered positive 
    detection = np.sum(np.array(results) > detection_threshold)
    
    # Verification Threshold: Proportion of positive predictions / total positive samples 
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
    verified = verification > verification_threshold
    
    return results, verified


# In[ ]:


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120:120+250,200:200+250, :]
    
    cv2.imshow('Verification', frame)
    
    # Verification trigger
    if cv2.waitKey(10) & 0xFF == ord('v'):
        # Save input image to application_data/input_image folder 
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         h, s, v = cv2.split(hsv)

#         lim = 255 - 10
#         v[v > lim] = 255
#         v[v <= lim] -= 10
        
#         final_hsv = cv2.merge((h, s, v))
#         img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
        # Run verification
        results, verified = verify(siamese_model, 0.5, 0.5)
        print(verified)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:


np.sum(np.squeeze(results) > 0.5)


# In[ ]:


results


# In[ ]:




