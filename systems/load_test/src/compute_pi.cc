

#include <stdio.h>

#include <random.h>
#include <stdlib.h>


void compute_pi() {}


int main(int argc, char *argv[]) {

  const size_t GB_PER_BYTE = 1000000000;
  const size_t BUFFER_SIZE = 32 * GB_PER_BYTE;
  char *memoryblock = (char *)calloc(BUFFER_SIZE, sizeof(char));

  srandom(10);  
  while (1) {
    for (size_t ii = 0; ii < BUFFER_SIZE; ++ii) {
       memoryblock[ii] = 'A' + ((memoryblock[2 * ii % BUFFER_SIZE] - 'A') % 26);
    }      
  }    
   return 0;
}  
