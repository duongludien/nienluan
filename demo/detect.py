import os
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import sys
import xml.etree.ElementTree as ET

flags = tf.app.flags
flags.DEFINE_string('img_dir', '', '')
flags.DEFINE_string('model_pb_path', '', '')
flags.DEFINE_string('label_map_proto_path' ,'', '')
FLAGS = flags.FLAGS

num_classes = 32
dir = FLAGS.img_dir
model = FLAGS.model_pb_path
label_map_proto = FLAGS.label_map_proto_path

images = []
labels = []
for file in os.listdir(dir):
	file = os.path.join(dir, file)
	if file[-3:] == 'png':
		images.append(file)
	if file[-3:] == 'xml':
		labels.append(file)

label_map = label_map_util.load_labelmap(label_map_proto)
#print(label_map)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
#print(categories)
category_index = label_map_util.create_category_index(categories)
#print(category_index)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(model, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

for img in images:
	image = cv2.imread(img)
	xml_file = img[ : -3] + "xml"
	tree = ET.parse(xml_file)
	root = tree.getroot()
	true_classes = []
	for member in root.findall('object'):
		true_class = member[0].text
		true_classes.append(true_class)
	
	image_expanded = np.expand_dims(image, axis=0)
	
	(boxes, scores, classes, num) = sess.run(
		[detection_boxes, detection_scores, detection_classes, num_detections],
		feed_dict={image_tensor: image_expanded})
	print(scores[:, :5])
	print(classes[:, :5])
	print(true_classes)

	vis_util.visualize_boxes_and_labels_on_image_array(
		image,
		np.squeeze(boxes),
		np.squeeze(classes).astype(np.int32),
		np.squeeze(scores),
		category_index,
		use_normalized_coordinates=True,
		line_thickness=2,
		min_score_thresh=0.64)

	cv2.imshow('Object detector', image)

	if cv2.waitKey(1) == ord('q'):
		break

cv2.destroyAllWindows()
