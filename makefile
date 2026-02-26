OUTPUT=out/a.out

RUN_ARGS := $(filter-out run, $(MAKECMDGOALS))

%:
	@:

compile:
	gcc $(file) -fopenmp -O3 -o $(OUTPUT)

run:
	./$(OUTPUT) $(RUN_ARGS)