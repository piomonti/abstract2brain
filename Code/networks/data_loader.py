### class to load neurosynth data into pytorch
#
#


class neurosynthData(Dataset):
  """load neurosynth dataset """

  def __init__(self, pickle_file):
    """
    Args:
        pickle_file (string): Path to the pickle file with annotations.
    """
    self.dat = pickle.load(open(pickle_file, 'rb'))
    self.wordVector  = self.dat['wordVectors']
    self.imageVector = self.dat['imageVectors']

  def __len__(self):
    return len(self.wordVector)

  def __getitem__( self, idx ):
    """
    get a sample
    """
    sample = {'image': self.imageVector[idx,:], 'wordVector': self.wordVector[idx,:]}
    return sample