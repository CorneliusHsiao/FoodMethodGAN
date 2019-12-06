import os
import sys
import json
import numpy as np

def main(infile_dir):
	out_json = open(infile_dir + '/output.json', 'w')
	out_data = list()

	in_labels = open(infile_dir + '/food-101/meta/labels.txt', 'r')
	label_list = in_labels.readlines()
	for i in range(len(label_list)):
		label_list[i] = label_list[i][:-1]
	in_class = open(infile_dir + '/food-101/meta/classes.txt', 'r')
	class_list = in_class.readlines()
	for i in range(len(class_list)):
		class_list[i] = class_list[i][:-1]

	for i in range(43, len(label_list)):
		label_name = label_list[i]
		class_name = class_list[i]
		print('Name: ' + label_name)
		entry = dict()
		entry['name'] = label_name
		entry['class'] = class_name
		ingredients = list()
		ingredients_in = input("Ingredients: \n")
		while (ingredients_in != '...'):
			if (ingredients_in == ' ' or ingredients_in == '\n' or ingredients_in == ''):
				ingredients_in = input("\t")
				continue
			extracted_ingredient = ingredients_in.strip().split(' ')[-1]
			ingredients.append(extracted_ingredient.lower())
			ingredients_in = input("\t")
		entry['ingredients'] = ingredients
		entry['method'] = input('Method: ')
		out_data.append(entry)

	json.dump(out_data, out_json, indent=2)
	out_json.close()

if __name__ == '__main__':
	infile_dir = os.getcwd()
	main(infile_dir)
