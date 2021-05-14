from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

dims = MPI.Compute_dims(size, 2)
periods = [1, 1]
cartesian2d = comm.Create_cart(dims, periods, reorder=False)
coord = cartesian2d.Get_coords(rank)
print(f"In 2d topogy, processor {rank} has coord {coord}")
West, East = cartesian2d.Shift(direction=1, disp=1)
North, South = cartesian2d.Shift(direction=0, disp=1)
print(f"In 2d topology, processor {rank} has neighbors: {West}, {East}, {North}, {South}.")

# exchange its rank with the four east/west/north/south neighbor
# Numpy array are communicated with very little overhead 
# but only with upper case methods
neighranks = numpy.zeros((1, 4))
# requests= numpy.zeros((1,4))
req=comm.Send([rank, MPI.INT], dest=West, tag=0)
req=comm.Send([rank, MPI.INT], dest=East, tag=0)
req=comm.Send([rank, MPI.INT], dest=North, tag=0)
req=comm.Send([rank, MPI.INT], dest=South, tag=0)
req=comm.Recv([neighranks[0], MPI.INT],src=West, tag=0)
req=comm.Recv([neighranks[1],MPI.INT],src=East, tag=0)
req=comm.Recv([neighranks[2],MPI.INT], src=North, tag=0)
req=comm.Recv([neighranks[3],MPI.INT],src=South, tag=0)
MPI.Request.waitall(req)

print(f"Processor {rank} receives their neighbors rank {neighranks[0]}, {neighranks[1]}, {neighranks[2]}, {neighranks[3]}.")