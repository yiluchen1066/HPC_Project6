from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

dims = MPI.Compute_dims(size, 2)
periods = [1,1]
cartesian2d = comm.Create_cart(dims, periods, reorder= False)
coord = cartesian2d.Get_coords(rank)
print(f"In 2d topogy, processor {rank} has coord {coord}")
left, right = cartesian2d.Shift(direction =0, disp =1)
top, bottom = cartesian2d.Shift(direction =1, disp =1)
print(f"In 2d topology, processor {rank} has neighbors: {left}, {right}, {top}, {bottom}. ")