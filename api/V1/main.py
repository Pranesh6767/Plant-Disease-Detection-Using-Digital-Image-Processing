import imageprocess
import predictor

import pickle

import os

response = 't'

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

img = 'test/13.JPG'

f_vector = imageprocess.feature_extractor(img)

if response=='a':
	p_vector = [f_vector['area'],f_vector['perimeter'],f_vector['red_mean'],f_vector['blue_mean'],f_vector['f2'],f_vector['green_std'],
	f_vector['f4'],f_vector['f6'],f_vector['f7']]

	res = predictor.apple_p(p_vector,apple_model)
	print(res)

if response=='c':
	p_vector = [f_vector['red_mean'], f_vector['green_mean'], f_vector['blue_mean'], f_vector['f2'], f_vector['red_std'], f_vector['blue_std'],
	f_vector['f7'], f_vector['f8']]

	res = predictor.corn_p(p_vector,corn_model)
	print(res)

if response=='g':
	p_vector = [f_vector['area'], f_vector['perimeter'], f_vector['red_mean'], f_vector['green_mean'], f_vector['blue_mean'], f_vector['f2'],
       f_vector['red_std'], f_vector['green_std'], f_vector['blue_std'], f_vector['f4'], f_vector['f5'], f_vector['f6'], f_vector['f7'], f_vector['f8']]

	res = predictor.grapes_p(p_vector,grapes_model)
	print(res)

if response=='p':
	p_vector = [f_vector['area'], f_vector['perimeter'], f_vector['green_mean'], f_vector['blue_mean'], f_vector['f2'], f_vector['red_std'],
       f_vector['green_std'], f_vector['blue_std'], f_vector['f4'], f_vector['f5'], f_vector['f7'], f_vector['f8']]

	res = predictor.potato_p(p_vector,potato_model)
	print(res)

if response=='t':
	del f_vector["f1"]
	p_vector = list(f_vector.values())

	res = predictor.tomato_p(p_vector,tomato_model)
	print(res)
