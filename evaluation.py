import argparse
import os

def main():
	args = ArgumentParser()
	eval(args)

def eval(args):
	# load all dev ids
	ids = load_ids_dev(args.ids_file)
	# load all anns for dev
	anns = load_anns_dev(args.anns_file,ids)
	# load predictions
	preds = load_preds(args.dev_file,ids)
	# evaluate and print errors
	tps = 0
	fps = 0
	fns = 0
	out_file = open(os.path.join(args.out_file), "w")
	for id in ids:
#		print(id,preds[id])
		anns_id = anns[id]
		if id in preds:
			preds_id = preds[id]
		else:
			preds_id = []
#		print(id,anns_id,preds_id)
		for pred in preds_id:
			if pred in anns_id:
				tps += 1
				out_file.write('TP\t'+id+'\t'+pred+'\n')
			else:
				fps += 1
				out_file.write('FP\t'+id+'\t'+pred+'\n')
		for ann in anns_id:
			if ann not in preds_id:
				fns += 1
				out_file.write('FN\t'+id+'\t'+ann+'\n')
	out_file.write('TPs='+str(tps)+'\n')
	print('TPs='+str(tps))
	out_file.write('FPs='+str(fps)+'\n')
	print('FPs='+str(fps))
	out_file.write('FNs='+str(fns)+'\n')
	print('FNs='+str(fns))
	precision = tps/(tps+fps)
	out_file.write('precision='+str(precision)+'\n')
	print('precision='+str(precision))
	recall = tps/(tps+fns)
	out_file.write('recall='+str(recall)+'\n')
	print('recall='+str(recall))
	fscore = 2*precision*recall/(precision+recall)
	out_file.write('fscore='+str(fscore)+'\n')
	print('fscore='+str(fscore))
	
def load_preds(dev_file,ids):
	preds = {}
	f = open(dev_file, "r")
	for line in f:
#		print(line,'**')
		if '\t' in line.strip(): 
			id,str_preds = line.strip().split('\t')
			if id in ids:
				if str_preds is not None:
					preds[id] = str_preds.split('|')
				else:
					preds[id] = []
		else:
			id = line.strip()
			preds[id] = []
	print('Number of documents with predictions:',len(preds))
	return preds	

def load_anns_dev(anns_file,ids):
	anns = {}
	f = open(anns_file, "r")
	for line in f:
		id,str_anns = line.strip().split('\t')
		if id in ids:
			if str_anns is not None:
				anns[id] = str_anns.split('|')
			else:
				anns[id] = []
	# add dev doc without annotations
	for id in ids:
		if id not in anns.keys():
			anns[id] = []
#	print(len(anns))
	return anns

def load_ids_dev(ids_file):
	f = open(ids_file, "r")
	ids = [line.strip() for line in f ]
	print('Number of dev documents:',len(ids))
	return ids

def ArgumentParser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dev_file', type=str, default='dev.txt', help='predictions for the documents of the development set')
	parser.add_argument('--anns_file', type=str, default='anns_train_dev.txt', help='annotations from documents of the development set')
	parser.add_argument('--ids_file', type=str, default='ids_development.txt', help='list of ids of the development set')
	parser.add_argument('--out_file', type=str, default='output.txt', help='output file')
	return parser.parse_args()

if __name__ == '__main__':
	main()
