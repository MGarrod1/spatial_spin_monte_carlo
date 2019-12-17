"""

Functions used for simulating spin
systems on networkx graphs for general
initial conditions and background fields.

"""

#Python functions:
import numpy as np
import pdb
import networkx as nx
import math
from scipy import stats
from scipy.sparse.linalg import eigs, eigsh
from tqdm import tqdm as tqdm

def crit_beta(graph):

	"""

	Estimates the critical value of the inverse
	temperature for a finit sized graph.

	Note: for larger graphs we will instead need to use functions
	from ARPACK to estimate the largest eigenvalue.

	Parameters
	-------------

	graph : networkx graph

	Returns
	-------------

	crit_beta : float

	inverse of the largest eigenvalue of the
	adjacency matrix associated with A.

	"""

	A = nx.to_numpy_matrix(graph)
	eigs, vectors = np.linalg.eig(A)
	spectral_radius = max([abs(k) for k in eigs])
	crit_beta = 1.0 / spectral_radius
	return crit_beta


def crit_beta_sparse(graph) :

	"""

	Estimate the critical temperature of
	a large graph using sparse matrices.

	Parameters
	--------------

	graph : networkx graph

	Returns
	------------

	crit_beta : float

	Inverse spectral radius of the graph


	"""

	print("Computing critical temperature")
	A_sparse = nx.to_scipy_sparse_matrix(graph)
	A_sparse = A_sparse.astype(float)
	evals_large, evecs_large = eigsh(A_sparse, 1, which='LM')
	crit_beta_sparse = 1.0 / evals_large[0]
	return crit_beta_sparse

class spins :

	def __init__(self,graph) : 
	
		"""
		
		Parameters
		---------------
		
		graph : networkx graph.

		Connectivity structure of the spin system to be simulated.
		
		"""
		
		self.Beta = 1.0
		self.graph = graph
		self.N = len(graph)
		self.node_states = np.random.choice([-1,1],size=self.N)
		
		#Set external field to zeor function as defulat:
		self.field = self.zero_field
		self.scalar_field = self.zero_field
		self.node_label_map = self.null_node_label_map

		self.sampling_method="Glauber" #other option is: "Metropolis"
	
	
	def zero_field(self,node_label_data) :
		return 0.0
	
	def applied_field_to_each_node(self,target_node) :

		return self.get_field_value(target_node)


	def null_node_label_map(self,target_node) :
		"""

		Maps a node to iself.
		
		Parameters
		------------
		
		target_node : int
		
		Returns
		-----------
		
		target_node : int

		"""
		return target_node
		
	def get_field_value(self,target_node) :

		"""

		Return the external field at a certain point in the embedding
		space. 
		
		Parameters
		------------
		
		target_node : int
		
		Returns
		------------
		
		h : float 
		
		field to be applied top the target node.

		"""

		"""
		If the scalr field is not a numpy
		array we assume that it is a function
		which takes node labels/positions as
		an input.
		"""
		if isinstance(self.scalar_field,np.ndarray)  :
			h = self.scalar_field[target_node]
		else :
			the_label = self.node_label_map(target_node)
			h = self.scalar_field( the_label )
		return h
		
	def Random_Initial_States(self) :
		"""Set the node states to a random state """
		self.node_states = np.random.choice([-1,1],size=self.N)
	
	def spin_flip_probability_glauber(self,neighbour_state_sum,h=0) : 

		"""

		Returns the probability to perform a single spin flip
		for Glauber dynamics.

		See Lynn and Lee 2016 equation (1)

		Parameters
		-------------

		neighbour_state_sum : float

		sum of the states of the nodes neighbours

		h : float (optional)

		value of the external field applied to the target node

		Returns
		-------------

		Probability : float

		Probability of performing a spin flip.

		"""

		Probability = math.exp( self.Beta*(neighbour_state_sum + h)  )/(( math.exp(self.Beta*(neighbour_state_sum+h) ) + math.exp(-1.0*self.Beta*(neighbour_state_sum+h))))

		return Probability
	

	def spin_flip_probability_metropolis(self,target_node,neighbour_state_sum,h=0) :

		target_node_spin = self.node_states[target_node]
		Energy_Difference = 2.0*target_node_spin*(neighbour_state_sum + h)

		if Energy_Difference > 0 :
			return math.exp( -1.0*self.Beta*Energy_Difference )
		else :
			return 1.0

	
	def pick_random_up_or_down(self,probability) : 
		#Choose +1 with probaility probability. -1 otherwise.
		return np.random.choice([1,-1],p=[probability,1-probability]) 
	
	
	def update_spin(self,target_node,new_spin) :
		"""Update spin at the specified site"""
		self.node_states[target_node] = new_spin
		
	
	def do_spin_flip(self) :
		
		#Pick a random node and get the sum of neighbour states:
		target_node = np.random.randint(self.N)
		neighbour_state_sum = self.neighbour_state_sum(target_node)
		h_target = self.get_field_value(target_node)

		if self.sampling_method == "Metropolis" :
			prob = self.spin_flip_probability_metropolis(target_node,neighbour_state_sum,h=h_target)
		elif self.sampling_method == "Glauber" :
			prob = self.spin_flip_probability_glauber(neighbour_state_sum, h=h_target)

		#Update stats:
		if self.sampling_method == "Glauber" :
			self.update_spin(target_node,self.pick_random_up_or_down(prob))

		elif self.sampling_method == "Metropolis" :
			u = np.random.uniform(0, 1)
			if prob > u :
				self.update_spin(target_node, -1.0*self.node_states[target_node])

	def neighbour_state_sum(self,target_node) :
		neighbour_nodes = self.graph.neighbors(target_node)
		neighbour_states = [ self.node_states[k] for k in neighbour_nodes ]
		return np.sum(neighbour_states)
	

class spatial_spins(spins) :

	"""
	Child of the spin class in which we
	associate node labels with positions
	(or in general any array label).
	"""

	def __init__(self,positions,graph) :
		self.positions = positions
		spatial_spins = spins.__init__(self,graph)
		self.node_label_map = self.spatial_node_label_map

	def spatial_node_label_map(self,target_node) :
		node_loc = self.positions[target_node]
		return tuple(node_loc)


def Run_MonteCarlo(graph,T,beta,T_Burn=0,positions=None,Initial_State=None,control_field=None,sampling_method="Metropolis") :

	"""

	Samples a sequence of spin states on an Ising system
	for the graph supplied.
	
	Parameters
	-------------
	
	graph : networkx graph
	
	network structure
	
	T : int
	
	number of time steps to run the simulation for
	
	beta : float
	
	Inverse temperature

	T_Burn : int (opt)

	Burn in time. We run the dynamics for T+T_Burn timesteps
	and only record samples after T_Burn.

	positions : numpy ndarray (optional)

	Positions of nodes. Needs to be in the same order as the
	node list. If this is specified we use the spatial spins
	class.
	
	Returns
	------------
	
	Spin_Series : ndarray (N x T) 
	
	Array continaing time series of spin values for each
	of the nodes. 
	
	"""
	
	#Initialize_spin class:
	if positions is not None :
		spin_system = spatial_spins(positions,graph)
	else :
		spin_system =  spins(graph)

	#Set params:
	spin_system.sampling_method=sampling_method
	spin_system.Beta = beta

	#Set the initial state:
	if Initial_State is not None :
		spin_system.node_states = np.copy(Initial_State)
	else :
		spin_system.node_states = [ np.random.choice([-1,+1]) for k in range(len(graph)) ]

	if control_field is not None :
		spin_system.field = spin_system.applied_field_to_each_node
		spin_system.scalar_field = control_field
	else :
		spin_system.scalar_field = np.zeros(len(graph))
	

	current_states = [ ] 
	for p in range(T_Burn+T) :
		spin_system.do_spin_flip()
		#Copy the array so it is not a pointer:
		current_state = np.copy(spin_system.node_states)
		if p > T_Burn-1 : 
			current_states.append(current_state)
	Spin_Series = np.asarray(current_states)

	return Spin_Series


def sample_magnetization_series(graph, T, beta, positions = None, T_Burn=0, Initial_State=None, control_field=None, sampling_method="Metropolis",take_positive=True) :

	"""

	Samples a time series of magnetization for an Ising
	system on the networkx graph provided.
	
	Parameters
	---------------
	
	graph : networkx graph
	
	The connectivity graph for the Ising system.

	T : int

	Number of timesteps to run simulations for.

	beta : float

	Inverse temperature of the Ising system.
	
	positions : numpy array (opt)
	
	positions associated with the nodes .

	T_Burn : int (optional)
	
	Select a burn in time for the Ising simulations.
	
	Initial_State : int (optional)
	
	Set the initial state for the Ising system.
	
	control_field : function or array
	
	control field acting on the spin system.

	sampling_method : str (opt)

	Choose between sampling using Metropolis or Glauber
	dynamics. Valid choices are: "Glauber" or "Metropolis".

	take_positive : bool (opt)

	Choose whether to take the absolute values of the magnetisation.
	This is useful if we want to reproduce the phase diagram in terms
	of the absolute magnetisation.

	Returns
	--------------
	
	magnetization_series :
	
	Time serires of the magnetization for a particular graph
	in the interval [T_Burn,T]. 

	"""

	spin_series = Run_MonteCarlo(graph, T, beta, T_Burn=T_Burn ,positions=positions, Initial_State=Initial_State, control_field=control_field,sampling_method=sampling_method)

	if take_positive == True :
		magnetization_series = [ abs(np.mean(k)) for k in spin_series ]
	elif take_positive == False :
		magnetization_series = [ np.mean(k) for k in spin_series]
	
	return magnetization_series

def sample_magnetization_average(graph, T, Glauber_Runs , beta,positions=None , T_Burn=0, Initial_State=None,control_field=None) :

	"""
	
	Average magnetization at the end of the chain
	for a specified number of Glauber runs.

	For each glauber run we take the average magnetization
	for the period after the burn in time.

	Returns
	------------

	M_mean : float

	Average magnetizaion of all the chains
	after T time steps

	M_sem : float

	Estimate of the standard error on the mean
	for the above.

	"""

	time_averaged_magnetizations = [ ]


	for q in tqdm( range(Glauber_Runs) ) :
		mean_mag = sample_magnetization_series(graph, T, beta, positions= positions , T_Burn=T_Burn, Initial_State=Initial_State,control_field=control_field)
		time_averaged_magnetizations.append(  np.mean( mean_mag   ) )


	M_mean = np.mean(time_averaged_magnetizations)
	M_sem = stats.sem(time_averaged_magnetizations)

	return M_mean , M_sem


def mf_high_beta_susceptibility(degrees, applied_fields, beta):
	"""

    Analytic formula for the susceptibility of nodes
    in a graph for the low temperature limit using
    the mean field approximation.


    Estimated from:
    Lynn, Christopher, and Daniel D. Lee. "Maximizing influence in an ising network:
    A mean-field optimal solution." Advances in Neural Information Processing Systems. 2016.

    by taking the derivative of equation (11)

    Parameters
    --------------

    Degrees : list


    applied_fields : list

    Returns
    -------------

    Susceptibilities : list

    Vector of susceptbilities for each node.

    """

	susceptibilities = [4.0 * beta * math.exp(-2.0 * beta * (k + abs(j))) for k, j in zip(degrees, applied_fields)]

	return susceptibilities


def sample_susceptibility(graph, T, T_Burn, beta, control=None):

	"""

    Sample the vector of susceptibilities for Ising
    model on the specified graph by running Glauber
    dynamics simulations.

    The susceptibility of the spin system can be
    estimated from the covariance matrix of the
    vector of spins.



    Parameters
    --------------

    graph : networkx graph

    Connectivity structure of the Ising system

    T : int

    Number of timesteps to run Glauber
    dynamics for.


    T_Burn : int

    Burn in time for the dynamics. The susceptibility
    is only estimated by computing the covariances
    in the time interval [T_Burn,T]

    beta : float

    inverse temperature for the Ising system

    control : numpy array (optional)

    vector of controls applied to each field.



    Returns
    -------------

    sus_vec : list

    Susceptibilities of each of the nodes in the graph.


    """

	N = len(graph)
	positions = np.random.uniform(0.0, 1.0, (N, 2))
	spin_series = Run_MonteCarlo(graph, positions, T, beta, positions=None, T_Burn=T_Burn, control_field=control)
	cov = beta * np.cov(np.transpose(spin_series))
	sus_vec = [np.sum(k) for k in cov]

	return sus_vec


def sample_average_susceptibility(graph, T, T_Burn, beta,Samples, control=None) :

	"""

	Use multiple simulations to obtain an estimate
	of the average susceptibility.

	Parameters
	------------


	Returns
	------------

	:return:
	"""

	all_sus = []
	for i in range(Samples):
		print("Susceptibility sample = {}".format(i))
		susceptibility_vals = sample_susceptibility(graph, T, T_Burn, beta, control=control)
		all_sus.append(susceptibility_vals)

	means_sus_vals = [np.mean(k) for k in np.transpose(all_sus)]

	#compute standard error on the mean:
	# convert to array:
	Sus_vals = np.asarray(all_sus)
	COV = np.cov(np.transpose(Sus_vals))

	variances = np.diag(COV)
	se_on_means = [(k / Samples) ** 0.5 for k in variances]

	return means_sus_vals , se_on_means
