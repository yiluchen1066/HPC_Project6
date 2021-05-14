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
West_rank = numpy.array(1)
East_rank = numpy.array(1)
North_rank = numpy.array(1)
South_rank = numpy.array(1)
# requests= numpy.zeros((1,4))

req = comm.Sendrecv(rank, dest=West, recvbuf=West_rank, source=West)
#req = comm.Sendrecv(rank, dest=East, recvbuf=East_rank, source=East)
#req = comm.Sendrecv(rank, dest=North, recvbuf=North_rank, source=North)
#req = comm.Sendrecv(rank, dest=South, recvbuf=South_rank, source = South)
#MPI.Request.waitall(req)

print(f"Processor {rank} receives their neighbors rank {West_rank,} {East_rank}, {North_rank}, {South_rank}.")