from mpi4py import MPI 
import numpy

# compuutes the sum of all ranks
# python's collective communication methods

# using the pickle-based communication of generic Python objetcs: all-lowercase methods 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# sum = comm.allreduce(rank, op=MPI.SUM)

# print(f"The proecssor {rank}  computes the sum of all ranks is {sum}")

# using the fast (near C-speed), direct array data communication of buffer-provider objects: the method names starting with an upercase letter. 

rankF = numpy.array(float(rank))
sum = numpy.zeros(1)
comm.Reduce(rankF, sum, op=MPI.SUM)
print(f"The proecssor {rank}  computes the sum of all ranks is {sum}")