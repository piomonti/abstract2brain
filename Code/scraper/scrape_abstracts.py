### We scrape all abstracts for papers in neurosynth from PUBMED 
#
#
#

import numpy as np 
import pandas as pd 
import os 
from Bio.Entrez import efetch, read
from Bio import Entrez
Entrez.email = 'ricardo.monti08@gmail.com'


def get_abstract(pmid):
	"""
	function to collect abstract from pubmed

	shameless taken from: https://stackoverflow.com/questions/17409107/obtaining-data-from-pubmed-using-python
	"""
	handle = efetch(db='pubmed', id=pmid, retmode='text', rettype='abstract')
	return handle.read()

 
def clean_abstract( abstract ):
	"""
	what we get from pubmed is a bit messy, so we clean it and throw away bits we are not interested in!
	"""

	return abstract.split('\n\n')[np.argmax([len(x) for x in abstract.split('\n\n')])]



# first get a list of all DOIs which we will use to collect abstracts
os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data')

# load in database:
dat = pd.read_table('database.txt')
# throw away all studies that are not in MNI space
dat = dat[ dat['space']=='MNI' ]

# get pmids:
pm_ids = np.unique(dat['id'])

# lets store everything in a dict of dicts for now
res = {x:{'MNI':[], 'abstract':[]} for x in pm_ids}

# start scraping:
print("We are about to scrape abstracts for " + str(len(pm_ids)) + " neuroscientific papers")

for pid in res.keys():
	print 'collecting for pid: ' + str(pid)
	try:
		res[pid]['abstract'] = clean_abstract( get_abstract( pid ) )
		res[pid]['MNI'] = dat[ dat['id']==pid][['x','y','z']]
	except:
		pass

# save results 
import cPickle as pickle 
pickle.dump(res, open('Abstract_MNI_raw.p', 'wb'))



















