#include "darknet.h"

#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <signal.h>

#define QUEUE_SIZE 64
#define INPUT_C 3

// Circular queue for images awaiting processing
typedef struct {
    int client_id;
    int image_id;
    image im;
    float *preprocessed_data;
} ClientImage;

typedef struct {
    ClientImage data[QUEUE_SIZE];
    int backlog;
    int next_in;
    int next_out;
    pthread_mutex_t lock;
    pthread_cond_t image_avail;
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
    pthread_cond_init(&queue->image_avail, NULL);
    pthread_cond_init(&queue->free_space, NULL);

    return queue;
}

void destroy_image_queue(ImageQueue *queue) {
    pthread_mutex_destroy(&queue->lock);
    pthread_cond_destroy(&queue->image_avail);
    pthread_cond_destroy(&queue->free_space);

    while (queue->backlog > 0) {
        free_image(queue->data[queue->next_out].im);
        free(queue->data[queue->next_out].preprocessed_data);
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

    pthread_cond_signal(&queue->image_avail);
    pthread_mutex_unlock(&queue->lock);
}

void read_from_image_queue(ClientImage *image, ImageQueue *queue) {
    pthread_mutex_lock(&queue->lock);

    while (!(queue->backlog > 0)) {
        pthread_cond_wait(&queue->image_avail, &queue->lock);
    }

    *image = queue->data[queue->next_out];
    queue->next_out = (queue->next_out + 1) % QUEUE_SIZE;
    queue->backlog -= 1;

    pthread_cond_signal(&queue->free_space);
    pthread_mutex_unlock(&queue->lock);
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

    return total_bytes_read;
}

int handle_connection(int fd, int tid, int input_h, int input_w, int prep_size, ImageQueue *queue) {
    int bytes = 0;
    int img_id = 0;

    int input_size = input_h * input_w * INPUT_C * sizeof(float);

    float *input_X, *prep_X;

    while (1) {
        input_X = NULL;
        prep_X = NULL;

        bytes = read_image_data(fd, &input_X, input_size);
        if (bytes < 0) {
            perror("Error reading image");
            exit(EXIT_FAILURE);
        }

        // This client is done
        if (bytes == 0) break;

        // Client has preprocessed the image data.
        if (prep_size > 0) {
            bytes = read_image_data(fd, &prep_X, prep_size);
            if (bytes < 0) {
                perror("Error reading prep");
                exit(EXIT_FAILURE);
            }
        }

        img_id++;

        ClientImage cim = {
                .client_id = tid, .image_id = img_id,
                .im = { .c = INPUT_C, .h = input_h, .w = input_w, .data = input_X },
                .preprocessed_data = prep_X
        };

        append_to_image_queue(cim, queue);
    }

    // Signal end
    ClientImage cim = { tid, -1 };
    append_to_image_queue(cim, queue);

    return 0;
}

typedef struct {
    int fd;
    int tid;
    int input_h;
    int input_w;
    int prep_size;
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

    handle_connection(new_fd, args->tid, args->input_h, args->input_w, args->prep_size, args->queue);
    close(new_fd);
//    }

    pthread_exit(NULL);
}

void run_server(char *datacfg, char *cfgfile, char *weightfile, int port, int size, int num_clients, float thresh, float hier_thresh, int partial, int display) {
    int err = 0;
    int i = 0;
    int b = 0;
    int num_workers = num_clients;

    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/coco.names");
    char **names = get_labels(name_list);

    // Set up yolo network for detection
    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    int batch_size = net->batch;
    srand(2222222);
    float nms = .45;

    // Last layer
    layer l = net->layers[net->n-1];

    int resize_h = size;
    int resize_w = size;

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

    int preprocessed_size = partial ? net->layers[0].inputs : 0;
    int im_size = net->c * net->h * net->w;

    for (i = 0; i < num_workers; i++) {
        wargs[i].fd = fd;
        wargs[i].tid = i;
        wargs[i].input_h = resize_h;
        wargs[i].input_w = resize_w;
        wargs[i].prep_size = preprocessed_size * sizeof(float);
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

    int sentinel_images = 0;
    int total_images = 0;

    int done = 0;
    double batch_start_time = 0;
    double bps = 0;
    double start_time = 0;

    int net_w = net->w;
    int net_h = net->h;

#ifdef OPENCV
    // Create windows for displaying detetcions
    char windows[batch_size][5];

    if (display) {
        for (b = 0; b < batch_size; ++b) {
            sprintf(windows[b], "%d", b);
            cvNamedWindow(windows[b], CV_WINDOW_NORMAL);
            cvMoveWindow(windows[b], b * net->w + 40, 100);
        }
    }
#endif

    while (1) {
        if (batch_size == 1) { // avoid copy if batch_size is 1
            for (b = 0; b < batch_size; b++) {
                read_from_image_queue(&batch[b], queue);

                if (batch[b].image_id == -1) { // sentinel image
                    sentinel_images++;
                    if (sentinel_images == num_workers) { // we are done
                        done = 1;
                        break;
                    }

                    // decrease b to try to get another image
                    b--;
                }
            }

            if (partial)
                batch_im.data = batch[0].preprocessed_data;
            else
                batch_im.data = batch[0].im.data;

        } else {
            for (b = 0; b < batch_size; b++) {
                read_from_image_queue(&batch[b], queue);

                if (batch[b].image_id == -1) { // sentinel image
                    sentinel_images++;
                    if (sentinel_images == num_workers) { // we are done
                        done = 1;
                        break;
                    }

                    // decrease b to try to get another image
                    b--;
                    continue;
                }

                if (partial)
                    copy_cpu(preprocessed_size, batch[b].preprocessed_data, 1, batch_im.data + b * preprocessed_size, 1);
                else
                    copy_cpu(im_size, batch[b].im.data, 1, batch_im.data + b * im_size, 1);
            }
        }

        // Check for end
        if (done) break;

        // Start timing
        if (total_images == 0) start_time = what_time_is_it_now();

        batch_start_time = what_time_is_it_now();

        total_images += batch_size;

        float *X = batch_im.data;
        network_predict(net, X);

        // Temporary workaround for input w and h to get detections
        net->w = resize_w;
        net->h = resize_h;

        for (b = 0; b < batch_size; b++) {
            int nboxes = 0;
            detection *dets = get_network_boxes(net, batch[b].im.w, batch[b].im.h, thresh, hier_thresh, 0, 1, b, &nboxes);
            if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
            draw_detections(batch[b].im, dets, nboxes, thresh, names, alphabet, l.classes);
            free_detections(dets, nboxes);
        }

        // Restore w and h to run next batch
        net->w = net_w;
        net->h = net_h;

        bps = 1 / (what_time_is_it_now() - batch_start_time);
        printf("\rBatch size: %d\tBPS: %5.3f", batch_size, bps);
        fflush(stdout);

        // Show and free input images
        for (i = 0; i < batch_size; i++) {
            #ifdef OPENCV
            if (display) {
                show_image(batch[i].im, windows[i]);
                cvWaitKey(1);
            }
            #endif

            free_image(batch[i].im);
            free(batch[i].preprocessed_data);
        }
    }

    double end_time = what_time_is_it_now();
    printf("\rDetection for %d workers and %d total images with batch size %d took %f seconds (%5.3f BPS).\n", num_workers, total_images, batch_size, end_time - start_time, (total_images / batch_size) / (end_time - start_time));

    for (i = 0; i < num_workers; i++) {
        pthread_join(workers[i], NULL);
    }

#ifdef OPENCV
    if (display) {
        cvWaitKey(0);
        cvDestroyAllWindows();
    }
#endif

    free_image(batch_im);
    destroy_image_queue(queue);
    pthread_mutex_destroy(&accept_lock);
}
