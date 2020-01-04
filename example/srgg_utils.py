"""

Functions used to generate and
plot Soft RGGs.

"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def Euclidean_Distance(x,y,Boundaries = 'S',Dom_Size=1.0):
	"""
	Euclidean distance between positions x and y. 
	"""
	
	d = len(x)
	dij = 0 

	#Loop over number of dimensions
	for k in range(d):
		# Compute the absolute distance
		dist = abs( x[k] - y[k] )
	
		#Extra condition for periodic BCs:
		if Boundaries == 'P' or Boundaries == 'p' :
			if dist>0.5*Dom_Size :
				dist = Dom_Size - dist
		# Add to the total distance
		dij = dij + dist**2

	dij = dij**0.5   
	
	return dij

def Make_Soft_RGG_Given_Positions(positions,Kernel,metric="Euclidean",return_connection_probs = False,kernel_params=[],distance_metric_function=None):


	"""

	Sample the adjacency matrix of a single Soft RGG
	given positions in some domain.

	Parameters
	---------------

	positions : list

	List of positions in the embedding domain.
	
	Kernel : kernel function (this will be a method
	of the kernel class object or some chosen function
	of a single variable).

	Returns
	---------------

	G : networkx graph. 


	Graph object. 

	"""

	N = len(positions)
	Connection_Probabilities = np.zeros( (N,N) ) 
	

	# Make a new networkx graph and add nodes:
	G = nx.Graph()
	for j in range(N) :
		G.add_node(j)

	for i in range(N) : 
		for j in range(i+1,N) : 

			#compute the pairwise distance:
			if metric == "Euclidean" : 
				dij = Euclidean_Distance(positions[i],positions[j])
			elif metric == "Custom" : 
				dij = distance_metric_function(positions[i],positions[j])
			
			#Compute the connection probaility:
			params = tuple( np.concatenate( ([dij],kernel_params) ) ) 
			probability = Kernel(*params)
			
			
			Connection_Probabilities[i][j] = probability
			Connection_Probabilities[j][i] = probability

			#Draw the edge with speicified probability:
			u = np.random.uniform(0,1.0)
			if u < probability :
				G.add_edge(i,j)
	if 	return_connection_probs == True :	
		return G , Connection_Probabilities
	else : 
		return G



class graph_plot :

	"""
	Class for plotting spatial graphs.
	"""
	
	def __init__( self , graph , positions = None ) :
	
		"""
	
		Parameters 
		---------------
	
		graph : networkx graph
	
		the graph for plotting
	
		positions : list 
	
		set of positions. Currently only supports plotting in 2D. 
	
		"""
		
	
		self.graph = graph
		if positions is not None and positions != 'spectral':
			
			#If we have more than 2 position coords then take only the first two:
			if len(positions[0]) != 2 : 
				print("More than 2d found. Taking first two dimensions")
				self.positions = [  [ i[0] , i[1] ]  for i in positions ] 
			else : 
				self.positions = positions
		
		#If positions are not specified use the spring layout:
		elif positions == None : 
			self.positions=nx.spring_layout(graph)
		
		elif positions == 'spectral' :
			self.positions=nx.spectral_layout(graph)
		

		#Set default parameters:
		self.node_size = 35
		self.node_line_width = 0.1
		self.color_scheme = 'jet'
		self.edge_width = 1.0
		self.node_color = 'b'
		self.node_shape = 'o'
		self.node_transparency = 1.0
		self.edge_color = 'k'
		self.edge_style = 'dashed' #solid|dashed|dotted,dashdot
		self.edge_transparency = 1.0

	def make_plot(self) : 
		nx.draw_networkx_nodes(self.graph, self.positions ,  node_size = self.node_size, node_color = self.node_color,node_shape = self.node_shape,alpha = self.node_transparency, cmap = self.color_scheme, linewidths = self.node_line_width)
		plt.axis('off')
		nx.draw_networkx_edges(self.graph, self.positions, width = self.edge_width,edge_color = self.edge_color,style = self.edge_style,alpha =self.edge_transparency)
	
	
	def Add_Edges(self,graph) :	
		plt.axis('off')
		nx.draw_networkx_edges(graph, self.positions, width = self.edge_width,edge_color = self.edge_color,style = self.edge_style,alpha =self.edge_transparency)
		
	def save_plot(self,file_path,format="pdf",fig_label=None) :

		plt.figure( figsize=(10, 10) )
		if fig_label is not None :
			ax = plt.gca()
			plt.text(0.0, 0.95, fig_label, horizontalalignment='center', fontsize=64, verticalalignment='center',transform=ax.transAxes)


		self.make_plot()
		plt.savefig(file_path + ".".format(format) , bbox_inches = 'tight',format=format)
		
	def show_plot(self) :
		plt.figure(figsize=(10,10))
		self.make_plot()
		plt.show()
		
	def change_node_size(self,size) :
		self.node_size = size


