import numpy as np

def apple_p(feature_vector,model):
	processed_vector = np.array(feature_vector).reshape(1, -1)
	output = model.predict(processed_vector)
	output = int(output)
	label_dict = {0 :'Apple___healthy', 1: 'Apple___Apple_scab', 2: 'Apple___Black_rot', 3: 'Apple___Cedar_apple_rust'}
	output = label_dict[output]
	return output

def corn_p(feature_vector,model):
	processed_vector = np.array(feature_vector).reshape(1, -1)
	output = model.predict(processed_vector)
	output = int(output)
	label_dict = {0: 'Corn_(maize)___healthy',
	1: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
	2: 'Corn_(maize)__Common_rust',
	3: 'Corn_(maize)___Northern_Leaf_Blight'}
	output = label_dict[output]
	return output

def grapes_p(feature_vector,model):
	processed_vector = np.array(feature_vector).reshape(1, -1)
	output = model.predict(processed_vector)
	output = int(output)
	label_dict = {0 : 'Grape___healthy',
	1 : 'Grape___Black_rot',
	2 : 'Grape___Esca_(Black_Measles)',
	3 : 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)'}
	output = label_dict[output]
	return output

def potato_p(feature_vector,model):
	processed_vector = np.array(feature_vector).reshape(1, -1)
	output = model.predict(processed_vector)
	output = int(output)
	label_dict = {0: 'Potato___healthy',
	1: 'Potato___Early_blight',
	2: 'Potato___Late_blight'}
	output = label_dict[output]
	return output

def tomato_p(feature_vector,model):
	processed_vector = np.array(feature_vector).reshape(1, -1)
	output = model.predict(processed_vector)
	output = int(output)
	label_dict = {0 : 'Tomato___healthy',
	1 : 'Tomato___Bacterial_spot',
	2 : 'Tomato___Early_blight',
	3 : 'Tomato___Late_blight',
	4 : 'Tomato___Leaf_Mold',
	5 : 'Tomato___Septoria_leaf_spot',
	6 : 'Tomato___Spider_mites Two-spotted_spider_mite',
	7 : 'Tomato___Target_Spot',
	8 : 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
	9 : 'Tomato___Tomato_mosaic_virus'}
	output = label_dict[output]
	return output

