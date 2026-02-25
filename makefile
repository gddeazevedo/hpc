OUTPUT=out/a.out

compile:
	gcc $(file) -fopenmp -O3 -o $(OUTPUT)

run:
	./$(OUTPUT) $(args)