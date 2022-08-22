import torch
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision.models import detection
import cv2
import numpy as np

# set the device we will be using to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# saved model to local file

# model = detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True,
#                                             num_classes=91,pretrained_backbone=True)
PATH = 'resnet.pth'
# torch.save(model.state_dict(), PATH)

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

COLORS = np.random.uniform(0, 255, size=(len(COCO_INSTANCE_CATEGORY_NAMES), 3))

# load model
model = detection.fasterrcnn_resnet50_fpn()
model.load_state_dict(torch.load(PATH))
model.to(DEVICE)
model.eval()
print("Successfully loaded pretrained resnet!")

# load the image from disk
image = cv2.imread("data\dog.jpg")
orig = image.copy()
# convert the image from BGR to RGB channel ordering and change the
# image from channels last to channels first ordering
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.transpose((2, 0, 1))
# add the batch dimension, scale the raw pixel intensities to the
# range [0, 1], and convert the image to a floating point tensor
image = np.expand_dims(image, axis=0)
image = image / 255.0
image = torch.FloatTensor(image)
# send the input to the device and pass the it through the network to
# get the detections and predictions
image = image.to(DEVICE)

weights = detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
weights = weights.meta["categories"]

with torch.inference_mode():
    detections = model(image)[0]
    # loop over the detections
    for i in range(0, len(detections["boxes"])):
    	# extract the confidence (i.e., probability) associated with the
    	# prediction
    	confidence = detections["scores"][i]
    	# filter out weak detections by ensuring the confidence is
    	# greater than the minimum confidence
    	if confidence > 0.75:
    		# extract the index of the class label from the detections,
    		# then compute the (x, y)-coordinates of the bounding box
    		# for the object
    		idx = int(detections["labels"][i])
    		box = detections["boxes"][i].detach().cpu().numpy()
    		(startX, startY, endX, endY) = box.astype("int")
    		# display the prediction to our terminal
    		label = "{}: {:.2f}%".format(COCO_INSTANCE_CATEGORY_NAMES[idx], confidence * 100)
    		print("[INFO] {}".format(label))
    		# draw the bounding box and label on the image
    		cv2.rectangle(orig, (startX, startY), (endX, endY),
    			COLORS[idx], 2)
    		y = startY - 15 if startY - 15 > 15 else startY + 15
    		cv2.putText(orig, label, (startX, y),
    			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

# show the output image
cv2.imshow("Output", orig)
cv2.waitKey(0)
