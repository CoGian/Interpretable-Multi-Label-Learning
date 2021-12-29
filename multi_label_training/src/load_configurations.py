def load_configs(configs):
	arg_dict = {}
	dataset = configs.get('dataset')
	assert isinstance(dataset['train_file'], str)
	assert isinstance(dataset['val_file'], str)
	assert isinstance(dataset['topics'], str)
	arg_dict['train_file'] = dataset['train_file']
	arg_dict['val_file'] = dataset['val_file']
	arg_dict['topics'] = dataset['topics']

	parameters = configs.get('parameters')
	assert isinstance(parameters['n_labels'], int)
	assert isinstance(parameters['lr'], float)
	assert isinstance(parameters['epochs'], int)
	assert isinstance(parameters['batch_size'], int)
	assert isinstance(parameters['max_length'], int)
	assert isinstance(parameters['dropout'], float)
	assert isinstance(parameters['threshold'], float)
	assert isinstance(parameters['token_classification'], bool)

	arg_dict['n_labels'] = parameters['n_labels']
	arg_dict['lr'] = parameters['lr']
	arg_dict['epochs'] = parameters['epochs']
	arg_dict['batch_size'] = parameters['batch_size']
	arg_dict['max_length'] = parameters['max_length']
	arg_dict['dropout'] = parameters['dropout']
	arg_dict['threshold'] = parameters['threshold']
	arg_dict['token_classification'] = parameters['token_classification']

	assert isinstance(configs.get('pretrained_model'), str)
	arg_dict['pretrained_model'] = configs.get('pretrained_model')

	logging = configs.get('logging')
	assert isinstance(logging.get('project'), str)
	assert isinstance(logging.get('logger_file'), str)
	arg_dict['project'] = logging.get('project')
	arg_dict['logger_file'] = logging.get('logger_file')

	checkpoint = configs.get('checkpoint')
	assert isinstance(checkpoint.get('folder'), str)
	assert isinstance(checkpoint.get('subfolder'), str)
	arg_dict['checkpoint_folder'] = checkpoint.get('folder')
	arg_dict['checkpoint_subfolder'] = checkpoint.get('subfolder')


	return arg_dict
