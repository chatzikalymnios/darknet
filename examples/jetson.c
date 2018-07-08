#include "darknet.h"

#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <errno.h>

#define QUEUE_SIZE 32

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

typedef struct {
    image im;
    image sized;
} loaded_image;

void free_loaded_image(void *item) {
    loaded_image *im = (loaded_image *) item;

    free_image(im->im);
    free_image(im->sized);

    free(im);
}

typedef struct {
    image im;
    int nboxes;
    detection *dets;
} processed_image;

void free_processed_image(void *item) {
    processed_image *im = (processed_image *) item;

    free_image(im->im);
    free_detections(im->dets, im->nboxes);

    free(im);
}

typedef struct {
    image im;
    long preprocessed_data_size;
    float *preprocessed_data;
} preprocessed_image;

void free_preprocessed_image(void *item) {
    preprocessed_image *im = (preprocessed_image *) item;

    free_image(im->im);
    free(im->preprocessed_data);

    free(im);
}

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

Queue * create_queue(void (*free_item) (void *)) {
    Queue *queue = (Queue *) malloc(sizeof(Queue));
    if (!queue) {
        perror("Error allocating queue");
        exit(EXIT_FAILURE);
    }

    queue->backlog= 0;
    queue->next_in = 0;
    queue->next_out = 0;
    queue->free_item = free_item;

    pthread_mutex_init(&queue->lock, NULL);
    pthread_cond_init(&queue->item_avail, NULL);
    pthread_cond_init(&queue->free_space, NULL);

    return queue;
}

void destroy_queue(Queue *queue) {
    pthread_mutex_destroy(&queue->lock);
    pthread_cond_destroy(&queue->item_avail);
    pthread_cond_destroy(&queue->free_space);

    while (queue->backlog > 0) {
        queue->free_item(queue->data[queue->next_out]);
        queue->next_out = (queue->next_out + 1) % QUEUE_SIZE;
        queue->backlog -= 1;
    }

    free(queue);
}

void append_to_queue(void *item, Queue *queue) {
    pthread_mutex_lock(&queue->lock);

    while (!(queue->backlog < QUEUE_SIZE)) {
        pthread_cond_wait(&queue->free_space, &queue->lock);
    }

    queue->data[queue->next_in] = item;
    queue->next_in = (queue->next_in + 1) % QUEUE_SIZE;
    queue->backlog += 1;

    pthread_cond_signal(&queue->item_avail);
    pthread_mutex_unlock(&queue->lock);
}

void read_from_queue(void **item, Queue *queue) {
    pthread_mutex_lock(&queue->lock);

    while (!(queue->backlog > 0)) {
        pthread_cond_wait(&queue->item_avail, &queue->lock);
    }

    *item = queue->data[queue->next_out];
    queue->next_out = (queue->next_out + 1) % QUEUE_SIZE;
    queue->backlog -= 1;

    pthread_cond_signal(&queue->free_space);
    pthread_mutex_unlock(&queue->lock);
}

typedef struct {
    list *paths;
    int resize_h;
    int resize_w;
    Queue *queue;
} ImageLoaderArgs;

void *image_loader(void *args_ptr) {
    ImageLoaderArgs *args = (ImageLoaderArgs *) args_ptr;

    int curr_idx = 0;
    char **image_paths = (char **) list_to_array(args->paths);

    char *path = 0;

    for (curr_idx = 0; curr_idx < args->paths->size; curr_idx++) {
        path = image_paths[curr_idx];
        image im = load_image_color(path, 0, 0);
        image sized = letterbox_image(im, args->resize_h, args->resize_w);

        loaded_image *loaded_im = (loaded_image *) malloc(sizeof(loaded_image));
        loaded_im->im = im;
        loaded_im->sized = sized;

        append_to_queue(loaded_im, args->queue);

        free(path);
    }

    loaded_image *end_im = (loaded_image *) malloc(sizeof(loaded_image));
    end_im->im.c = 0; // signals end

    append_to_queue(end_im, args->queue);

    pthread_exit(NULL);
}

typedef struct {
    network *net;
    float thresh;
    float nms;
    float hier_thresh;
    Queue *image_queue;
    Queue *out_queue;
} DetectorArgs;

void *detector(void *args_ptr) {
    DetectorArgs *args = (DetectorArgs *) args_ptr;

    loaded_image *input = NULL;
    int b = 0;

    layer l = args->net->layers[args->net->n - 1];

    while (1) {
        read_from_queue((void **) &input, args->image_queue);

        // Check for end of input data
        if (!input->im.c) break;

        network_predict(args->net, input->sized.data);

        // Retrieve detections
        int nboxes = 0;
        detection *dets = get_network_boxes(args->net, input->im.w, input->im.h, args->thresh, args->hier_thresh, 0, 1,
                                            b, &nboxes);
        if (args->nms) do_nms_sort(dets, nboxes, l.classes, args->nms);

        if (args->out_queue) {
            // Forward to printer thread
            processed_image *processed_im = (processed_image *) malloc(sizeof(processed_image));
            processed_im->im = input->im;
            processed_im->nboxes = nboxes;
            processed_im->dets = dets;

            append_to_queue(processed_im, args->out_queue);
        } else {
            free_image(input->im);
            free_detections(dets, nboxes);
        }

        free_image(input->sized);
        free(input);
    }

    if (args->out_queue) {
        processed_image *end_im = (processed_image *) malloc(sizeof(processed_image));
        end_im->im.c = 0; // signals end

        append_to_queue(end_im, args->out_queue);
    }

    pthread_exit(NULL);
}

typedef struct {
    network *net;
    Queue *image_queue;
    int fd;
    Queue *out_queue;
} PartialDetectorArgs;

void *partial_detector(void *args_ptr) {
    PartialDetectorArgs *args = (PartialDetectorArgs *) args_ptr;

    loaded_image *input = NULL;

    layer l = args->net->layers[args->net->n - 1];

    int prep_size = l.outputs * sizeof(float);

    while (1) {
        read_from_queue((void **) &input, args->image_queue);

        // Check for end of input data
        if (!input->im.c) break;

        // Preprocess
        network_predict(args->net, input->sized.data);

        preprocessed_image *prep_im = (preprocessed_image *) malloc(sizeof(preprocessed_image));
        prep_im->im = input->sized;
        prep_im->preprocessed_data = (float *) malloc(prep_size);
        memcpy(prep_im->preprocessed_data, l.output, prep_size);
        prep_im->preprocessed_data_size = prep_size;

        append_to_queue(prep_im, args->out_queue);

        free_image(input->im);
        free(input);
    }

    preprocessed_image *end_im = (preprocessed_image *) malloc(sizeof(preprocessed_image));
    end_im->im.c = 0; // signals end

    append_to_queue(end_im, args->out_queue);

    pthread_exit(NULL);
}

typedef struct {
    char *name_list;
    float thresh;
    int classes;
    Queue *image_queue;
} PrinterArgs;

void *printer(void *args_ptr) {
    PrinterArgs *args = (PrinterArgs *) args_ptr;

    image **alphabet = load_alphabet();
    char **names = get_labels(args->name_list);

    processed_image *input = NULL;

#ifdef OPENCV
    // Create windows for displaying detetcions
    char *window = "detections";
    cvNamedWindow(window, CV_WINDOW_NORMAL);
#endif

    while (1) {
        read_from_queue((void **) &input, args->image_queue);

        // Check for end of input data
        if (!input->im.c) break;

        draw_detections(input->im, input->dets, input->nboxes, args->thresh, names, alphabet, args->classes);
        free_detections(input->dets, input->nboxes);

#ifdef OPENCV
        show_image(input->im, window);
        cvWaitKey(1);
#endif
        free_image(input->im);
        free(input);
    }

#ifdef OPENCV
    cvDestroyAllWindows();
#endif

    pthread_exit(NULL);
}

typedef struct {
    int fd;
    Queue *image_queue;
} ForwarderArgs;

void *forwarder(void *args_ptr) {
    ForwarderArgs *args = (ForwarderArgs *) args_ptr;

    preprocessed_image *input = NULL;
    int err = 0;

    while (1) {
        read_from_queue((void **) &input, args->image_queue);

        // Check for end of data
        if (!input->im.c) break;

        // Send input image (known size)
        err = writen(args->fd, input->im.data, input->im.c * input->im.h * input->im.w * sizeof(float));
        if (err < 0) {
            perror("Error sending image data");
            exit(EXIT_FAILURE);
        }

        // Send preprocessed data (knows size)
        err = writen(args->fd, input->preprocessed_data, input->preprocessed_data_size);
        if (err < 0) {
            perror("Error sending preprocessed data");
            exit(EXIT_FAILURE);
        }

        free_preprocessed_image(input);
    }

    pthread_exit(NULL);
}

int connect_to_server(char *server_hostname, char *server_port) {
    int fd = 0, err = 0;
    struct addrinfo hints;
    struct addrinfo *servinfo, *p;

    // Connect to server
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    err = getaddrinfo(server_hostname, server_port, &hints, &servinfo);
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

    if (p) return fd;

    return -1;
}

void run_remote_detection(network *net, list *paths, char *server_hostname, char *server_port) {
    int fd = connect_to_server(server_hostname, server_port);
    if (fd < 0) {
        printf("Could not connect to server\n");
        exit(EXIT_FAILURE);
    }

    int err = 0;

    double start_time = what_time_is_it_now();

    // Image loader
    Queue *image_queue = create_queue(free_loaded_image);
    pthread_t loader_thread;
    ImageLoaderArgs loader_args = { .resize_h = net->h, .resize_w = net->h, .paths = paths, .queue = image_queue };

    err = pthread_create(&loader_thread, NULL, image_loader, (void *) &loader_args);
    if (err < 0) {
        perror("Error creating loader thread");
        exit(EXIT_FAILURE);
    }

    // Partial Detector
    Queue *preprocessed_queue = create_queue(free_preprocessed_image);
    pthread_t partial_detector_thread;
    PartialDetectorArgs partial_detector_args = { .net = net, .image_queue = image_queue, .fd = fd, .out_queue = preprocessed_queue };

    err = pthread_create(&partial_detector_thread, NULL, partial_detector, (void *) &partial_detector_args);
    if (err < 0) {
        perror("Error creating detector thread");
        exit(EXIT_FAILURE);
    }

    // Forwarder
    pthread_t forwarder_thread;
    ForwarderArgs forwarder_args = { .fd = fd, .image_queue = preprocessed_queue };
    err = pthread_create(&forwarder_thread, NULL, forwarder, (void *) &forwarder_args);
    if (err < 0) {
        perror("Error creating forwarder thread");
        exit(EXIT_FAILURE);
    }

    pthread_join(loader_thread, NULL);
    pthread_join(partial_detector_thread, NULL);
    pthread_join(forwarder_thread, NULL);

    double end_time = what_time_is_it_now();
    printf("\nNote: timing includes thread creation overhead\n");
    printf("Preprocessing and sending of %d images took %f seconds\t(%5.3f FPS)\n", paths->size, end_time - start_time, paths->size / (end_time - start_time));

    destroy_queue(image_queue);
    destroy_queue(preprocessed_queue);
}

void run_local_detection(network *net, list *paths, char *name_list, float thresh, float nms, float hier_thresh, int display) {
    int err = 0;

    double start_time = what_time_is_it_now();

    // Image loader
    Queue *image_queue = create_queue(free_loaded_image);
    pthread_t loader_thread;
    ImageLoaderArgs loader_args = { .resize_h = net->h, .resize_w = net->w, .paths = paths, .queue = image_queue };

    err = pthread_create(&loader_thread, NULL, image_loader, (void *) &loader_args);
    if (err < 0) {
        perror("Error creating loader thread");
        exit(EXIT_FAILURE);
    }

    // Detector
    Queue *processed_queue = NULL;
    if (display) {
        processed_queue = create_queue(free_processed_image);
    }

    pthread_t detector_thread;
    DetectorArgs detector_args = { .net = net, .thresh = thresh, .nms = nms, .hier_thresh = hier_thresh, .image_queue = image_queue, .out_queue = processed_queue };

    err = pthread_create(&detector_thread, NULL, detector, (void *) &detector_args);
    if (err < 0) {
        perror("Error creating detector thread");
        exit(EXIT_FAILURE);
    }

    // Printer
    pthread_t printer_thread;
    PrinterArgs printer_args = {.name_list= name_list, .classes = net->layers[net->n -
                                                                              1].classes, .thresh = thresh, .image_queue = processed_queue};
    if (display) {
        err = pthread_create(&printer_thread, NULL, printer, (void *) &printer_args);
        if (err < 0) {
            perror("Error creating printer thread");
            exit(EXIT_FAILURE);
        }
    }

    pthread_join(loader_thread, NULL);
    pthread_join(detector_thread, NULL);

    if (display) {
        pthread_join(printer_thread, NULL);
    }

    double end_time = what_time_is_it_now();
    printf("\nNote: timing includes thread creation overhead\n");
    printf("Detection of %d images took %f seconds\t(%5.3f FPS)\n", paths->size, end_time - start_time, paths->size / (end_time - start_time));

    destroy_queue(image_queue);
    if (display) {
        destroy_queue(processed_queue);
    }
}

void run_jetson(char *datacfg, char *cfgfile, char *weightfile, char *imgfile, char *server_hostname, char *server_port, float thresh, int display) {
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/coco.names");

    // Load YOLO network
    network *net = load_network(cfgfile, weightfile, 0);
    srand(2222222);
    float nms = .45;
    float hier_thresh = .5;

    // Are we working completely locally?
    int local = (server_hostname == 0 || server_port == 0) ? 1 : 0;

    // Get image paths
    list *paths = get_paths(imgfile);

    if (local && paths) {
        run_local_detection(net, paths, name_list, thresh, nms, hier_thresh, display);
    } else if (!local && paths) {
        run_remote_detection(net, paths, server_hostname, server_port);
    } else {
        printf("Invalid argument combination\n");
    }

    if (paths) {
        free_list(paths);
    }
}