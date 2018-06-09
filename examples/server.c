#include "darknet.h"

#include "image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <errno.h>
#include <signal.h>

#define QUEUE_SIZE 64

#define INPUT_C 3

// Circular queue for images awaiting processing

typedef struct {
    int client_id;
    int image_id;
    image im;
} ClientImage;

typedef struct {
    ClientImage data[QUEUE_SIZE];
    int backlog;
    int next_in;
    int next_out;
    pthread_mutex_t lock;
    pthread_cond_t images_avail;
    pthread_cond_t free_space;
} ImageQueue;

ImageQueue * create_image_queue() {
    ImageQueue *queue = (ImageQueue *) malloc(sizeof(ImageQueue));
    if (!queue) {
        perror("Error allocating image queue");
        exit(EXIT_FAILURE);
    }

    queue->backlog= 0;
    queue->next_in = 0;
    queue->next_out = 0;

    pthread_mutex_init(&queue->lock, NULL);
    pthread_cond_init(&queue->images_avail, NULL);
    pthread_cond_init(&queue->free_space, NULL);

    return queue;
}

void destroy_image_queue(ImageQueue *queue) {
    pthread_mutex_destroy(&queue->lock);
    pthread_cond_destroy(&queue->images_avail);
    pthread_cond_destroy(&queue->free_space);

    while (queue->backlog > 0) {
        free(queue->data[queue->next_out].im.data);
        queue->next_out = (queue->next_out + 1) % QUEUE_SIZE;
        queue->backlog -= 1;
    }

    free(queue);
}

void append_to_image_queue(ClientImage image, ImageQueue *queue) {
    pthread_mutex_lock(&queue->lock);

    while (!(queue->backlog < QUEUE_SIZE)) {
        pthread_cond_wait(&queue->free_space, &queue->lock);
    }

    queue->data[queue->next_in] = image;
    queue->next_in = (queue->next_in + 1) % QUEUE_SIZE;
    queue->backlog += 1;

    pthread_cond_signal(&queue->images_avail);
    pthread_mutex_unlock(&queue->lock);
}

void read_from_image_queue(ClientImage *image, ImageQueue *queue) {
    pthread_mutex_lock(&queue->lock);

    while (!(queue->backlog > 0)) {
        pthread_cond_wait(&queue->images_avail, &queue->lock);
    }

    *image = queue->data[queue->next_out];
    queue->next_out = (queue->next_out + 1) % QUEUE_SIZE;
    queue->backlog -= 1;

    pthread_cond_signal(&queue->free_space);
    pthread_mutex_unlock(&queue->lock);
}

int read_image_data(int fd, float **X, size_t X_mem_size) {
    size_t total_bytes_read = 0;
    ssize_t bytes_read = 0;

    *X = (float *) malloc(X_mem_size);
    void *ptr;

    while (total_bytes_read < X_mem_size) {
        ptr = (void *) *X + total_bytes_read;

        bytes_read = read(fd, ptr, X_mem_size - total_bytes_read);

        if (bytes_read < 0) return -1;
        if (bytes_read == 0) break;

        total_bytes_read += bytes_read;
    }

//    printf("Wrote to %p\n", *X);

    return total_bytes_read;
}

int handle_connection(int fd, int tid, int input_h, int input_w, ImageQueue *queue) {
    int bytes;
    size_t X_mem_size = INPUT_C * input_h * input_w * sizeof(float);
    float *X = NULL;

    int img_id = 0;

    while (1) {
        X = NULL;

        bytes = read_image_data(fd, &X, X_mem_size);
        if (bytes < 0) {
            perror("Error reading image");
            exit(EXIT_FAILURE);
        }

        // This client is done
        if (bytes == 0) {
            // append sentinel image to notify the main thread
            ClientImage im = {
                    tid,
                    -1,
                    {0, 0, 0, NULL}
            };

            append_to_image_queue(im, queue);

            break;
        }

        if (X) {
            img_id += 1;

            ClientImage im = {
                    tid,
                    img_id,
                    { .c = INPUT_C, .h = input_h, .w = input_w, .data = X }
            };

            append_to_image_queue(im, queue);
        }
    }

    return 0;
}

typedef struct {
    int fd;
    int tid;
    int input_h;
    int input_w;
    pthread_mutex_t *accept_lock;
    ImageQueue *queue;
} WorkerArgs;

void *listen_for_requests(void *args_ptr) {
    WorkerArgs *args = (WorkerArgs *) args_ptr;
    int fd = args->fd;
    int new_fd = 0;
    int optval = 1;
    int err;

    // No while loop. For now, each worker handles exactly one client ("camera") and shuts down.
//    while (1) {
        pthread_mutex_lock(args->accept_lock);
        new_fd = accept(fd, NULL, NULL);
        pthread_mutex_unlock(args->accept_lock);
        if (new_fd < 0) {
            perror("Error accepting");
//            continue;
            pthread_exit(NULL);
        }

        err = setsockopt(new_fd, SOL_SOCKET, SO_KEEPALIVE, &optval, sizeof(int));
        if (err < 0) {
            perror("Error setting new socket option");
        }

        handle_connection(new_fd, args->tid, args->input_h, args->input_w, args->queue);
        close(new_fd);
//    }

    pthread_exit(NULL);
}

int socket_setup(int port, int backlog) {
    int fd, err, optval;
    struct sockaddr_in addr;

    fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    optval = 1;
    err = setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(int));
    if (err < 0) return -1;

    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    err = bind(fd, (struct sockaddr *) &addr, sizeof(struct sockaddr_in));
    if (err < 0) return -1;

    err = listen(fd, backlog);
    if (err < 0) return -1;

    return fd;
}

void run_detector_server(char *cfgfile, char *weightfile, float thresh, float hier_thresh, int port, int num_workers) {
    int err = 0;
    int i = 0;
    int b = 0;

    // Set up yolo network for detection
    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    int batch_size = net->batch;
    srand(2222222);
    double time;
    float nms = .45;

    // Create queue for client images
    printf("Creating image queue...\n");
    ImageQueue *queue = create_image_queue();

    // Setup threads to accept connections from clients
    printf("Setting up server...\n");
    int fd = 0;
    pthread_t workers[num_workers];
    WorkerArgs wargs[num_workers];

    fd = socket_setup(port, num_workers);
    if (fd < 0) {
        perror("Error setting up socket");
        exit(EXIT_FAILURE);
    }

    // Ignoring SIGPIPE to avoid server crashes
    signal(SIGPIPE, SIG_IGN);

    // Lock synchronize threads accepting new connections
    pthread_mutex_t accept_lock;
    pthread_mutex_init(&accept_lock, NULL);

    for (i = 0; i < num_workers; i++) {
        wargs[i].fd = fd;
        wargs[i].tid = i;
        wargs[i].input_h = net->h;
        wargs[i].input_w = net->w;
        wargs[i].accept_lock = &accept_lock;
        wargs[i].queue = queue;
        err = pthread_create(&workers[i], NULL, listen_for_requests, (void *) &wargs[i]);
        if (err < 0) {
            perror("Error creating new thread");
            exit(EXIT_FAILURE);
        }
    }

    printf("%d workers awaiting connections on port %d...\n", num_workers, port);

    image batch_im = make_image(net->w, net->h, net->c * batch_size);
    ClientImage batch[batch_size];
    int im_size = net->w * net->h * net->c;

    int sentinel_images = 0;
    int total_images = 0;

    int done = 0;

    layer l = net->layers[net->n-1];

    char **names = get_labels("data/coco.names");

    if (batch_size == 1) {
        // avoid copy
        for (i = 0; i < batch_size; i++) {
            read_from_image_queue(&batch[i], queue);

            if (batch[i].image_id == -1) {
                // sentinel image
                sentinel_images++;
                if (sentinel_images == num_workers) {
                    // we are done
                    done = 1;
                    break;
                }

                // decrease i to try to get another image
                i--;
                continue;
            }
        }

        batch_im.data = batch[0].im.data;

    } else {
        for (i = 0; i < batch_size; i++) {
            read_from_image_queue(&batch[i], queue);

            if (batch[i].image_id == -1) {
                // sentinel image
                sentinel_images++;
                if (sentinel_images == num_workers) {
                    // we are done
                    done = 1;
                    break;
                }

                // decrease i to try to get another image
                i--;
                continue;
            }

            copy_cpu(im_size, batch[i].im.data, 1, batch_im.data + i * im_size, 1);
        }
    }

#ifdef OPENCV
    char windows[batch_size][5];

    for (b = 0; b < batch_size; b++) {
        sprintf(windows[b], "%d", b);
        cvNamedWindow(windows[b], CV_WINDOW_NORMAL);
        cvMoveWindow(windows[b], b * net->w + 20, 100);
    }
#endif

    double start_time = what_time_is_it_now();
    double batch_time = 0;
    double bps = 0;

    time=what_time_is_it_now();

    // Read images from queue and process
    while (!done) {
        total_images += batch_size;

        float *X = batch_im.data;

        network_predict(net, X);

        for (b = 0; b < batch_size; b++) {
            int nboxes = 0;
            detection *dets = get_network_boxes(net, batch[b].im.w, batch[b].im.h, thresh, hier_thresh, 0, 1, b, &nboxes);
//            printf("nboxes: %d\n", nboxes);
            if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
            draw_detections(batch[b].im, dets, nboxes, thresh, names, alphabet, l.classes);
//            for (int k = 0; k < nboxes; k++) {
//                printf("%d: %f %f %f %f\n", k, dets[k].bbox.x, dets[k].bbox.y, dets[k].bbox.w, dets[k].bbox.h);
//            }
//            printf("\n");
            free_detections(dets, nboxes);
        }

        batch_time = what_time_is_it_now()-time;
        bps = 1 / batch_time;
        printf("\rBatch size: %d\tBPS: %5.3f\t FPS: %5.3f (per client)", batch_size, bps, bps / batch_size);
        fflush(stdout);

#ifdef OPENCV
        for (b = 0; b < batch_size; b++) {
            show_image(batch[b].im, windows[b]);
        }
        cvWaitKey(1);
#endif

        // Free input images
        for (i = 0; i < batch_size; i++) {
            free(batch[i].im.data);
        }

        time=what_time_is_it_now();

        if (batch_size == 1) {
            // avoid copy
            for (i = 0; i < batch_size; i++) {
                read_from_image_queue(&batch[i], queue);

                if (batch[i].image_id == -1) {
                    // sentinel image
                    sentinel_images++;
                    if (sentinel_images == num_workers) {
                        // we are done
                        done = 1;
                        break;
                    }

                    // decrease i to try to get another image
                    i--;
                    continue;
                }
            }

            batch_im.data = batch[0].im.data;

        } else {
            for (i = 0; i < batch_size; i++) {
                read_from_image_queue(&batch[i], queue);

                if (batch[i].image_id == -1) {
                    // sentinel image
                    sentinel_images++;
                    if (sentinel_images == num_workers) {
                        // we are done
                        done = 1;
                        break;
                    }

                    // decrease i to try to get another image
                    i--;
                    continue;
                }

                copy_cpu(im_size, batch[i].im.data, 1, batch_im.data + i * im_size, 1);
            }
        }
    }

    double end_time = what_time_is_it_now();
    printf("\nDetection for %d workers and %d total images with batch size %d took %f seconds.\n", num_workers, total_images, batch_size, end_time - start_time);

#ifdef OPENCV
    cvWaitKey(0);
    cvDestroyAllWindows();
#endif

    for (i = 0; i < num_workers; i++) {
        pthread_join(workers[i], NULL);
    }

    free_image(batch_im);
    destroy_image_queue(queue);
    pthread_mutex_destroy(&accept_lock);
}