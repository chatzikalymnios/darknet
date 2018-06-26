#include "darknet.h"

#include "data.h"
#include "utils.h"
#include "image.h"
#include "cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <errno.h>

#define QUEUE_SIZE 64

// Added here for reference. Defined in jetson.c
extern ssize_t writen(int fd, const void *vptr, size_t n);

typedef struct {
    image im;
    image sized;
} loaded_image;

extern void free_loaded_image(void *item);

typedef struct {
    void * data[QUEUE_SIZE];
    int backlog;
    int next_in;
    int next_out;
    void (*free_item) (void *);
    pthread_mutex_t lock;
    pthread_cond_t item_avail;
    pthread_cond_t free_space;
} Queue;

extern Queue * create_queue(void (*free_item) (void *));

extern void destroy_queue(Queue *queue);

extern void append_to_queue(void *item, Queue *queue);

extern void read_from_queue(void **item, Queue *queue);

typedef struct {
    list *paths;
    int resize_h;
    int resize_w;
    Queue *queue;
} ImageLoaderArgs;

extern void *image_loader(void *args_ptr);

void run_client(char *imgfile, char *host, char *port, int resize, double fps) {
    int fd, err;
    struct addrinfo hints;
    struct addrinfo *servinfo, *p;

    // Get paths of the images to be sent to the server
    list *paths = get_paths(imgfile);

    // Image loader
    Queue *image_queue = create_queue(free_loaded_image);
    pthread_t loader_thread;
    ImageLoaderArgs loader_args = { .resize_h = resize, .resize_w = resize, .paths = paths, .queue = image_queue };

    err = pthread_create(&loader_thread, NULL, image_loader, (void *) &loader_args);
    if (err < 0) {
        perror("Error creating loader thread");
        exit(EXIT_FAILURE);
    }

    // Connect to server
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    err = getaddrinfo(host, port, &hints, &servinfo);
    if (err < 0) {
        perror("Error resolving host");
        exit(EXIT_FAILURE);
    }

    for (p = servinfo; p != NULL; p = p->ai_next) {
        fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) {
            perror("Error opening socket");
            exit(EXIT_FAILURE);
        }

        err = connect(fd, p->ai_addr, p->ai_addrlen);
        if (err == 0) {
            break;
        }
    }

    freeaddrinfo(servinfo);

    double delay = (1 / fps) * 1000000; // usec

    loaded_image *loaded_im = NULL;
    int mem_size = 3 * resize * resize * sizeof(float);
    int total_images = 0;

    if (p) {
        // Send all images and close socket

        double start_time = what_time_is_it_now();

        while (1) {
            read_from_queue((void **) &loaded_im, image_queue);

            // Done.
            if (loaded_im->im.c == 0) break;

            err = writen(fd, loaded_im->sized.data, mem_size);
            if (err < 0) {
                perror("Error sending image data");
                exit(EXIT_FAILURE);
            }

            free_loaded_image(loaded_im);

            total_images++;

            usleep(delay);
        }

        shutdown(fd, SHUT_RDWR);
        close(fd);

        double elapsed = what_time_is_it_now() - start_time;
        printf("Sending images took %f seconds\t(%5.3f FPS)\n", elapsed, total_images / elapsed);
    } else {
        fprintf(stderr, "Could not connect to host");
    }

    pthread_join(loader_thread, NULL);
    destroy_queue(image_queue);
}