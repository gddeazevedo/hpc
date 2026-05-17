#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>


int main(int argc, char ** argv ) {
    MPI_Init( &argc, & argv );

    int rank;
    int size;
    
    MPI_Comm_rank(MPI_COMM_WORLD,  &rank );
    MPI_Comm_size(MPI_COMM_WORLD,  &size );   
    
    char send = 'A' + rank;
    
    char recv[size];
     
    MPI_Allgather(&send, 1, MPI_CHAR, recv, 1, MPI_CHAR, MPI_COMM_WORLD );
    
    for(int i = 0; i < size; i++) {
        printf("[%d] %c\n", rank, recv[i] );
    }
    
    MPI_Finalize();
    return 0;
}
