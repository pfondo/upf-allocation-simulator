# upf-allocation-simulator


## Description

Simulator for testing the allocation of User Plane Function (UPF) entities in dynamic Multi-Access Edge Computing (MEC) scenarios. Users are connected to base stations and data connectivity is provided by the closest UPF.

The simulator allows the evaluation of algorithms for the dynamic allocation of UPFs and calculates the latency between each UE and its closest UPF.

## Execution

	usage: main.py [-h] --algorithm ALGORITHM [--minUPFs MINUPFS] [--maxUPFs MAXUPFS] --bsFile BSFILE
				   --ueFile UEFILE [--iterationDuration ITERATIONDURATION]

	  -h, --help            show this help message and exit
	  --algorithm ALGORITHM
							Specifies the UPF allocation algorithm [Supported: random/greedy_percentile/gr
							eedy_percentile_fast/greedy_average/greedy_max/kmeans/kmeans_greedy_average/mo
							dularity_greedy_average/girvan_newman_greedy_average].
	  --minUPFs MINUPFS     Specifies the minimum number of UPFs to be allocated [Default: 1].
	  --maxUPFs MAXUPFS     Specifies the maximum number of UPFs to be allocated [Default: 10].
	  --bsFile BSFILE       File containing the information about the base stations [Format: each line
							contains the id, x coordinate and y coordinate of a base station separated by
							spaces].
	  --ueFile UEFILE       File containing the information about the users throughout the simulation
							[Format: each line contains the timestamp, user id, x coordinate, y
							coordinate, speed and, optionally, the base station id to which the user is
							attached].
	  --iterationDuration ITERATIONDURATION
							Duration of each time-slot [Default: 5].

## Results

The results are printed through the standard output stream with the following format:

    ALGORITHM NUM_UPFS LATENCY_AVG LATENCY_CI95_LOW LATENCY_CI95_HIGH EXECUTION_TIME_AVG EXECUTION_TIME_CI95_LOW EXECUTION_TIME_CI95_HIGH

Status messages are printed through the standard error stream in order to provide information about the current status of the simulation.

## Adding an algorithm

An additional algorithm named "algX" can be added to the simulator by implementing a method with the following signature:

    def UPF_allocation_algX(G: nx.Graph, num_UPFs, BSs_with_UPF_previous, G_shortest_path_lengths, highest_bs_id)

The method must return a set of num_UPFs integers representing the IDs of the base stations where UPFs are going to be deployed in the next interval.

## Copyright

Copyright ⓒ 2021 Pablo Fondo Ferreiro <pfondo@gti.uvigo.es>, David Candal Ventureira <dcandal@gti.uvigo.es>

This simulator is licensed under the GNU General Public License, version 3 (GPL-3.0). For more information see LICENSE.txt
