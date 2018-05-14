'''python train.py --train_dir=path/to/train_dir --pipeline_config_path=pipeline_config.pbtxt'''
import functools
import json
import os
from object_detection import trainer
from object_detection.builders import dataset_builder
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import dataset_util
from object_detection.builders import optimizer_builder
from object_detection.builders import preprocessor_builder
from object_detection.core import batcher
from object_detection.core import preprocessor
from object_detection.core import standard_fields as fields
from object_detection.utils import ops as util_ops
from object_detection.utils import variables_helper
from deployment import model_deploy
import tensorflow as tf
slim = tf.contrib.slim


tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('train_dir', '', '')
flags.DEFINE_string('pipeline_config_path', '', '')

FLAGS = flags.FLAGS

def get_next(config):
	return dataset_util.make_initializable_iterator(dataset_builder.build(config)).get_next()

assert FLAGS.train_dir, '`train_dir` is missing.'
if FLAGS.pipeline_config_path:
	configs = config_util.get_configs_from_pipeline_file(FLAGS.pipeline_config_path)
	tf.gfile.Copy(FLAGS.pipeline_config_path, os.path.join(FLAGS.train_dir, 'pipeline.config'), overwrite=True)
	model_config = configs['model']
	train_config = configs['train_config']
	input_config = configs['train_input_config']
	model = functools.partial(model_builder.build, model_config=model_config, is_training=True)
	input = functools.partial(get_next, input_config)
	#trainer.train(input, model, train_config, 'lonely_worker', FLAGS.train_dir)
	
	master=''
	task=0
	worker_replicas=1
	ps_tasks=0
	num_clones=1
	clone_on_cpu=False
	is_chief=True
	detection_model = model()
	data_augmentation_options = [preprocessor_builder.build(step) for step in train_config.data_augmentation_options]


	with tf.Graph().as_default():
		
		deploy_config = model_deploy.DeploymentConfig(num_clones=num_clones,
				clone_on_cpu=clone_on_cpu,
				replica_id=task,
				num_replicas=worker_replicas,
				num_ps_tasks=ps_tasks,
				worker_job_name='lonely_worker')
				
		#print(deploy_config.variables_device(), 'FTTTTTTG')
		with tf.device(deploy_config.variables_device()):
			global_step = slim.create_global_step()
			
		#print('In function create_input_queue: ', train_config.batch_queue_capacity, ', ', train_config.num_batch_queue_threads, ', ', train_config.prefetch_queue_capacity)
		#print('True' if data_augmentation_options else 'False')
		with tf.device(deploy_config.inputs_device()):
			input_queue = trainer.create_input_queue(train_config.batch_size // num_clones,
					input, train_config.batch_queue_capacity,
					train_config.num_batch_queue_threads,
					train_config.prefetch_queue_capacity, data_augmentation_options)

		#print(len(input_queue.dequeue()))
		#initial summaries.
		summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
		
		global_summaries = set([])

		model_fn = functools.partial(trainer._create_losses, create_model_fn=model, train_config=train_config)
		clones = model_deploy.create_clones(deploy_config, model_fn, [input_queue])
		print(len(clones))
		first_clone_scope = clones[0].scope

		update_ops = []
		with tf.device(deploy_config.optimizer_device()):
			#momentum optimizer va summaries
			training_optimizer, optimizer_summary_vars = optimizer_builder.build(train_config.optimizer)
			for var in optimizer_summary_vars:
				tf.summary.scalar(var.op.name, var)

		#restore checkpoint.
		init_fn = None
		if train_config.fine_tune_checkpoint:
			var_map = detection_model.restore_map(from_detection_checkpoint=train_config.from_detection_checkpoint)
			available_var_map = (variables_helper.get_variables_available_in_checkpoint(var_map, train_config.fine_tune_checkpoint))
			init_saver = tf.train.Saver(available_var_map)
			def initializer_fn(sess):
				init_saver.restore(sess, train_config.fine_tune_checkpoint)
			init_fn = initializer_fn
		
		with tf.device(deploy_config.optimizer_device()):
			regularization_losses = (None if train_config.add_regularization_loss else [])
			#print('regularization_losses', regularization_losses)
			#tf.add_n sumloss va gradient cho tung variable.
			total_loss, grads_and_vars = model_deploy.optimize_clones(clones, training_optimizer, regularization_losses=regularization_losses)
			total_loss = tf.check_numerics(total_loss, 'LossTensor is inf or nan.')

			#clip gradients
			if train_config.gradient_clipping_by_norm > 0:
				with tf.name_scope('clip_grads'):
					grads_and_vars = slim.learning.clip_gradient_norms(grads_and_vars, train_config.gradient_clipping_by_norm)

			#gradient updates.
			grad_updates = training_optimizer.apply_gradients(grads_and_vars, global_step=global_step)
			update_ops.append(grad_updates)
			update_op = tf.group(*update_ops, name='update_barrier')
			with tf.control_dependencies([update_op]):
				train_tensor = tf.identity(total_loss, name='train_op')

		
		#add summaries.
		for model_var in slim.get_model_variables():
			global_summaries.add(tf.summary.histogram(model_var.op.name, model_var))
		for loss_tensor in tf.losses.get_losses():
			global_summaries.add(tf.summary.scalar(loss_tensor.op.name, loss_tensor))
		global_summaries.add(tf.summary.scalar('TotalLoss', tf.losses.get_total_loss()))

		#union
		summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))
		summaries |= global_summaries
		#print('summaries: ', summaries) -> list
		summary_op = tf.summary.merge(list(summaries), name='summary_op')

		#Khong co GPU -> xai CPU
		session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

		# Save checkpoints.
		keep_checkpoint_every_n_hours = train_config.keep_checkpoint_every_n_hours
		saver = tf.train.Saver(keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

		slim.learning.train(
				train_tensor,
				logdir=FLAGS.train_dir,
				master=master,
				is_chief=is_chief,
				session_config=session_config,
				startup_delay_steps=train_config.startup_delay_steps,
				init_fn=init_fn,
				summary_op=summary_op,
				number_of_steps=(train_config.num_steps if train_config.num_steps else None),
				save_summaries_secs=120,
				sync_optimizer=None,
				saver=saver)

