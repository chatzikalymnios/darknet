#include "darknet.h"
#include "list.h"

#define QUEUE_SIZE 64

// Added here for reference. Defined in jetson.c

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


void run_batch_detector(char *datacfg, char *cfgfile, char *weightfile, char *imgfile, float thresh, float hier_thresh, int display) {
    int b;

    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/coco.names");
    char **names = get_labels(name_list);

    // Set up yolo network for detection
    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    srand(2222222);
    float nms = .45;

    int batch_size = net->batch;

    // Get image paths
    list *paths = get_paths(imgfile);

#ifdef OPENCV
    char windows[batch_size][5];

    if (display) {
        for (b = 0; b < batch_size; ++b) {
            sprintf(windows[b], "%d", b);
            cvNamedWindow(windows[b], CV_WINDOW_NORMAL);
            cvMoveWindow(windows[b], b * net->w + 40, 100);
        }
    }
#endif

    int err = 0;

    // Image loader
    Queue *image_queue = create_queue(free_loaded_image);
    pthread_t loader_thread;
    ImageLoaderArgs loader_args = { .resize_h = net->h, .resize_w = net->w, .paths = paths, .queue = image_queue };

    err = pthread_create(&loader_thread, NULL, image_loader, (void *) &loader_args);
    if (err < 0) {
        perror("Error creating loader thread");
        exit(EXIT_FAILURE);
    }

    layer l = net->layers[net->n-1];
    int im_size = net->w * net->h * net->c;

    // Image to contain batch data
    image batch_im = make_image(net->w, net->h, net->c * batch_size);

    // Array to contain unresized images of a batch
    loaded_image *batch[batch_size];

    double batch_start_time = 0;
    double bps = 0;
    int total_images = 0;
    double start_time = what_time_is_it_now();
    int done = 0;

    while (!done) {
        for (b = 0; b < batch_size; b++) {
            read_from_queue((void **) &batch[b], image_queue);

            if (!batch[b]->im.c) { // sentinel image
                done = 1;
                break;
            }

            if (batch_size == 1) // avoid copy
                batch_im.data = batch[b]->sized.data;
            else
                copy_cpu(im_size, batch[b]->sized.data, 1, batch_im.data + b * im_size, 1);
        }

        // Check for end
        if (done) break;

        // Start timing
        if (total_images == 0) start_time = what_time_is_it_now();

        batch_start_time = what_time_is_it_now();

        total_images += batch_size;

        float *X = batch_im.data;
        network_predict(net, X);

        bps = 1 / (what_time_is_it_now() - batch_start_time);
        printf("\rBatch size: %d\tBPS: %5.3f", batch_size, bps);
        fflush(stdout);

        for (b = 0; b < batch_size; b++) {
            int nboxes = 0;
            detection *dets = get_network_boxes(net, batch[b]->im.w, batch[b]->im.h, thresh, hier_thresh, 0, 1, b, &nboxes);
            if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
            draw_detections(batch[b]->im, dets, nboxes, thresh, names, alphabet, l.classes);
            free_detections(dets, nboxes);
        }

        // Show and free input images
        for (b = 0; b < batch_size; b++) {
#ifdef OPENCV
            if (display) {
                show_image(batch[b]->im, windows[b]);
                cvWaitKey(1);
            }
#endif

            free_loaded_image(batch[b]);
        }
    }

    double end_time = what_time_is_it_now();
    printf("\rDetection for %d total images with batch size %d took %f seconds (%5.3f BPS).\n", total_images, batch_size, end_time - start_time, total_images / (end_time - start_time));

#ifdef OPENCV
    if (display) {
        cvWaitKey(0);
        cvDestroyAllWindows();
    }
#endif

    pthread_join(loader_thread, NULL);

    destroy_queue(image_queue);
}