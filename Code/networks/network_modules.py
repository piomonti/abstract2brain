# implementation of network modules 

import torch
import torch.nn as nn 
from torch.utils.data import Dataset 


class MLP( nn.Module ):
  """
  implement basic module for MLP 

  note that this module keeps the dimensions fixed! will implement a mapping from a 
  vector of dimension input_size to another vector of dimension input_size

  """
  
  def __init__( self, input_size ):
    super( MLP, self ).__init__()
    self.activation_function = nn.functional.relu
    self.linear_layer = nn.Linear( input_size, input_size )
    self.bn_layer = nn.BatchNorm1d( input_size )
    
  def forward( self, x ):
    x = self.bn_layer( x )
    linear_act = self.linear_layer( x )
    H_x = self.activation_function( linear_act )
    return H_x



class View( nn.Module ):
  """
  reshape tensor size!
  taken shamelessly from: https://discuss.pytorch.org/t/equivalent-of-np-reshape-in-pytorch/144/5
  """
  def __init__(self, *shape):
    super(View, self).__init__()
    self.shape = shape
  def forward(self, input):
    return input.view(*shape)



