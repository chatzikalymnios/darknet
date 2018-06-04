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

#define RESIZE_W 416
#define RESIZE_H 416

ssize_t writen(int fd, const void *vptr, size_t n) {
    size_t nleft;
    ssize_t nwritten;
    const char *ptr;

    ptr = vptr;
    nleft = n;
    while (nleft > 0) {
        if ((nwritten = write(fd, ptr, nleft)) <= 0) {
            if (errno == EINTR) nwritten = 0;
            else return -1;
        }
        nleft -= nwritten;
        ptr += nwritten;
    }

    return n;
}

void run_detector_client(char *imgfile, char *host, char *port) {
    int fd, err;
    struct addrinfo hints;
    struct addrinfo *servinfo, *p;
    char *responses;
    size_t len;

    // Get paths of the images to be sent to the server
    list *paths = get_paths(imgfile);

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

    if (p) {
        // Send all images and close socket

        while (paths->size > 0) {
            char *path = list_pop(paths);
            image im = load_image_color(path,0,0);
            image sized = letterbox_image(im, RESIZE_W, RESIZE_H);

            int mem_size = sized.c * sized.h * sized.w * sizeof(float);

            err = writen(fd, sized.data, mem_size);
            if (err < 0) {
                perror("Error sending image data");
                exit(EXIT_FAILURE);
            }

            printf("%s (%d bytes)\n", path, mem_size);
        }

        shutdown(fd, SHUT_RDWR);
        close(fd);
    } else {
        fprintf(stderr, "Could not connect to host");
    }

    return 0;
}