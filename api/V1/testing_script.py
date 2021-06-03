###############################################################
# Plant disease detection 
# API V1
# Version 0.1
# Testing Script
###############################################################
# Importing libraries
import imageprocess
import predictor
import pickle
import os

###############################################################
# Loading the models with pickle

applemodelpath = 'models/Applemodel_V1.sav'
apple_model = pickle.load(open(applemodelpath, 'rb'))

cornmodelpath = 'models/cornmodel_V1.sav'
corn_model = pickle.load(open(cornmodelpath, 'rb'))

grapesmodelpath = 'models/grapesmodel_V1.sav'
grapes_model = pickle.load(open(grapesmodelpath, 'rb'))

potatomodelpath = 'models/potatomodel_V1.sav'
potato_model = pickle.load(open(potatomodelpath, 'rb'))

tomatomodelpath = 'models/Tomatomodel_V1.sav'
tomato_model = pickle.load(open(tomatomodelpath, 'rb'))

###############################################################

def main():
	f_vector = imageprocess.feature_extractor(img)

	if response=='a':
		p_vector = [f_vector['area'],f_vector['perimeter'],f_vector['red_mean'],f_vector['blue_mean'],f_vector['f2'],f_vector['green_std'],
		f_vector['f4'],f_vector['f6'],f_vector['f7']]

		res = predictor.apple_p(p_vector,apple_model)
		return res

	if response=='c':
		p_vector = [f_vector['red_mean'], f_vector['green_mean'], f_vector['blue_mean'], f_vector['f2'], f_vector['red_std'], f_vector['blue_std'],
		f_vector['f7'], f_vector['f8']]

		res = predictor.corn_p(p_vector,corn_model)
		return res

	if response=='g':
		p_vector = [f_vector['area'], f_vector['perimeter'], f_vector['red_mean'], f_vector['green_mean'], f_vector['blue_mean'], f_vector['f2'],
	       f_vector['red_std'], f_vector['green_std'], f_vector['blue_std'], f_vector['f4'], f_vector['f5'], f_vector['f6'], f_vector['f7'], f_vector['f8']]

		res = predictor.grapes_p(p_vector,grapes_model)
		return res

	if response=='p':
		p_vector = [f_vector['area'], f_vector['perimeter'], f_vector['green_mean'], f_vector['blue_mean'], f_vector['f2'], f_vector['red_std'],
	       f_vector['green_std'], f_vector['blue_std'], f_vector['f4'], f_vector['f5'], f_vector['f7'], f_vector['f8']]

		res = predictor.potato_p(p_vector,potato_model)
		return res

	if response=='t':
		del f_vector["f1"]
		p_vector = list(f_vector.values())

		res = predictor.tomato_p(p_vector,tomato_model)
		return res


print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("!!!!!!!!!!!!!!!  Initiating The Testing Lop  !!!!!!!!!!!!!!!!")
testimagelist = os.listdir('test')
testimagelist.sort()
predicted_labels = []
k = 0
for i in testimagelist:
	response = i[0]
	img = 'test/'+i
	el = main()
	predicted_labels.append(el)
	print("!!!!!  sample  ",k,"  processed  !!!!" )
	k=k+1

target_labels = ['Apple___Black_rot', 'Apple___Apple_scab', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
'Corn_(maize)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Grape___healthy', 'Grape___Esca_(Black_Measles)',
 'Potato___Late_blight', 'Potato___healthy', 'Tomato___healthy', 'Tomato___Leaf_Mold', 'Tomato___Tomato_mosaic_virus']

flag = True
for i in range(len(predicted_labels)):
 	if predicted_labels[i] == target_labels[i]:
 		print("!!!!   Test Successfully passed    !!!!")
 	else:
 		flag = False
 		print("!!!!   Test Failed   !!!")

print("!!!!!    Testing Completed   !!!!!!")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
if flag:
	print("All tests passed Successfully")
else:
	print("Might be some error !")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

a = input()



