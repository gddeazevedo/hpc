OUTPUT=out/a.out

RUN_ARGS := $(filter-out run, $(MAKECMDGOALS))

%:
	@:

compile:
	gcc ./openmp/$(file) -fopenmp -O3 -o $(OUTPUT)

compile-mpi:
	mpicc ./mpi/$(file) -fopenmp -O3 -o $(OUTPUT)

run:
	@./$(OUTPUT) $(RUN_ARGS)

run-mpi:
	@mpirun -np 4 ./$(OUTPUT) $(RUN_ARGS)