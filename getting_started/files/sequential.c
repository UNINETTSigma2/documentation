// assumed to be sequential.c
#include <stdio.h>
#include <time.h>

double approximate_pi(void) {
    const int n = 2000000;
    int n_in = 0;
    for (int i = 0; i < n; i++) {
        double x = (double)rand() / RAND_MAX * 2.0 - 1.0;
        double y = (double)rand() / RAND_MAX * 2.0 - 1.0;
        if (x * x + y * y < 1.0)
            n_in++;
    }
    return 4.0 * n_in / n;
}

int main(void) {
    srand(time(NULL));
    for (int i = 0; i < 100; i++)
        approximate_pi();
    return 0;
}
