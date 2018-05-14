from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import io
import pandas as pd
import tensorflow as tf
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('output_path', '', '')
flags.DEFINE_string('xml_path', '', '')
FLAGS = flags.FLAGS

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('path').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['path', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def class_text_to_int(row_label):
	if row_label == 'c_102':
		return 1
	elif row_label == 'c_103a':
		return 2
	elif row_label == 'c_106b':
		return 3
	elif row_label == 'c_107':
		return 4
	elif row_label == 'c_115':
		return 5
	elif row_label == 'c_130':
		return 6
	elif row_label == 'c_131a':
		return 7
	elif row_label == 'cd_423a':
		return 8
	elif row_label == 'cd_423b':
		return 9
	elif row_label == 'cd_425':
		return 10
	elif row_label == 'cd_428':
		return 11
	elif row_label == 'cd_434':
		return 12
	elif row_label == 'cd_443':
		return 13
	elif row_label == 'hl_303':
		return 14
	elif row_label == 'nh_201a':
		return 15
	elif row_label == 'nh_201b':
		return 16
	elif row_label == 'nh_202a':
		return 17
	elif row_label == 'nh_202b':
		return 18
	elif row_label == 'nh_205a':
		return 19
	elif row_label == 'nh_205b':
		return 20
	elif row_label == 'nh_205c':
		return 21
	elif row_label == 'nh_205d':
		return 22
	elif row_label == 'nh_207a':
		return 23
	elif row_label == 'nh_207b':
		return 24
	elif row_label == 'nh_207c':
		return 25
	elif row_label == 'nh_207d':
		return 26
	elif row_label == 'nh_208':
		return 27
	elif row_label == 'nh_209':
		return 28
	elif row_label == 'nh_221b':
		return 29
	elif row_label == 'nh_224':
		return 30
	elif row_label == 'nh_225':
		return 31
	elif row_label == 'nh_233':
		return 32
	else:
		None


def split(df, group):
	data = namedtuple('data', ['path', 'object'])
	gb = df.groupby(group)
	return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group):
	with tf.gfile.GFile(format(group.path), 'rb') as fid:
		encoded_png = fid.read()
	encoded_png_io = io.BytesIO(encoded_png)
	image = Image.open(encoded_png_io)
	width, height = image.size

	filename = group.path.split('/')[-1].encode('utf8')
	#print(filename)
	image_format = b'png'
	xmins = []
	xmaxs = []
	ymins = []
	ymaxs = []
	classes_text = []
	classes = []

	for index, row in group.object.iterrows():
		xmins.append(row['xmin'] / width)
		xmaxs.append(row['xmax'] / width)
		ymins.append(row['ymin'] / height)
		ymaxs.append(row['ymax'] / height)
		classes_text.append(row['class'].encode('utf8'))
		classes.append(class_text_to_int(row['class']))

	tf_example = tf.train.Example(features=tf.train.Features(feature={
		'image/height': dataset_util.int64_feature(height),
		'image/width': dataset_util.int64_feature(width),
		'image/filename': dataset_util.bytes_feature(filename),
		'image/source_id': dataset_util.bytes_feature(filename),
		'image/encoded': dataset_util.bytes_feature(encoded_png),
		'image/format': dataset_util.bytes_feature(image_format),
		'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
		'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
		'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
		'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
		'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
		'image/object/class/label': dataset_util.int64_list_feature(classes),
	}))
	return tf_example


def main(_):
	writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
	
	xml_path = FLAGS.xml_path
	examples = xml_to_csv(xml_path)
	print(examples)
	grouped = split(examples, 'path')
	#print(grouped)
	for group in grouped:
		#print(group)
		tf_example = create_tf_example(group)
		#writer.write(tf_example.SerializeToString())

	writer.close()
	
	print('Successfully created the TFRecords: {}'.format(FLAGS.output_path))


if __name__ == '__main__':
	tf.app.run()
