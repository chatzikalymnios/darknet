#include "darknet.h"
#include "list.h"

void run_batch_detector(char *cfgfile, char *weightfile, float thresh, float hier_thresh, int batch_size, char **imgfiles, int display) {
    int b;

    // Set up yolo network for detection
    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    srand(2222222);
    float nms = .45;
    char **names = get_labels("data/coco.names");

    // Get image paths
    // Assumption: all image lists have the same length
    list *paths[batch_size];

    for (b = 0; b < batch_size; ++b) {
        paths[b] = get_paths(imgfiles[b]);
    }

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

    layer l = net->layers[net->n-1];
    int im_size = net->w * net->h * net->c;

    // Image to contain batch data
    image batch_im = make_image(net->w, net->h, net->c * batch_size);

    // Array to contain unresized images of a batch
    image images[batch_size];

    double time = 0;
    double batch_time = 0;
    double bps = 0;
    int total_images = 0;
    double start_time = what_time_is_it_now();

    while (paths[0]->size > 0) {
        time = what_time_is_it_now();

        #pragma omp parallel for
        for (b = 0; b < batch_size; ++b) {
            char *path = list_pop(paths[b]);
            images[b] = load_image_color(path,0,0);
            image sized = letterbox_image(images[b], net->h, net->w);
            copy_cpu(im_size, sized.data, 1, batch_im.data + b * im_size, 1);
            free(path);
        }

        total_images += batch_size;

        network_predict(net, batch_im.data);

        #pragma omp parallel for
        for (b = 0; b < batch_size; ++b) {
            int nboxes = 0;
            detection *dets = get_network_boxes(net, images[b].w, images[b].h, thresh, hier_thresh, 0, 1, b, &nboxes);
            if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
            draw_detections(images[b], dets, nboxes, thresh, names, alphabet, l.classes);
            free_detections(dets, nboxes);
        }

        batch_time = what_time_is_it_now() - time;
        bps = 1 / batch_time;
        printf("\rBatch size: %d\tBPS: %5.3f\t FPS: %5.3f (per camera)", batch_size, bps, bps / batch_size);
        fflush(stdout);

#ifdef OPENCV
        if (display) {
            #pragma omp parallel for
            for (b = 0; b < batch_size; ++b) {
                show_image(images[b], windows[b]);
            }
            cvWaitKey(1);
        }
#endif

        // Free input images
        #pragma omp parallel for
        for (b = 0; b < batch_size; ++b) {
            free(images[b].data);
        }
    }

    double end_time = what_time_is_it_now();
    printf("\nDetection for %d total images with batch size %d took %f seconds.\n", total_images, batch_size, end_time - start_time);

#ifdef OPENCV
    if (display) {
        cvWaitKey(0);
        cvDestroyAllWindows();
    }
#endif

    free_image(batch_im);
}