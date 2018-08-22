## General

View `examples/darknet.c` to see all the modes and their flags in detail.

## Distributed Jetson TX2 - Server detection

First, get and split the weights file for YOLOv3.

```bash
$ sh ./weights/get_and_split_weights.sh
```

Now the weights directory will contain `yolov3.weights`, `yolov3-server.weights` and `yolov3-jetson.weights`.

Edit the `Makefile` as necessary and compile darknet. You need to compile with OpenCV if you need to display the detections on the screen.

```bash
$ make -j $(nproc)
```

First run the server process on the server

```bash
./darknet server cfg/yolov3-608-server.cfg weights/yolov3-server.weights -num_clients 1 -size 608 -partial
```

The `-partial` flag is important. If it is ommited, the server will assume no preprocessing is taking place and will expect to receive only the input images. You can also specify a port number or leave it to the default number `12345`. Adding the `-display` flag will also display the detections on the server screen (requires OpenCV).

Run the client process on the Jetson TX2 module

```bash
$ ./darknet jetson cfg/yolov3-608-jetson.cfg weights/yolov3-jetson.weights <image list file> -port <server port> -host <server hostname>
```

## Client - Server detection

In this mode, the client simply forwards the input images to the server without any preprocessing.

First run the server process on the server

```bash
./darknet server cfg/yolov3.cfg weights/yolov3.weights -num_clients 1
```

Note that the `-partial` flag is missing!

Run the client process.

```bash
./darknet client <image list file> <server hostname> <server port> <scale> <fps>
```

## Batch detection (local)

In this mode, images are processed locally just like the defualt version of YOLO, but they are processed in batches.

```bash
./darknet batch cfg/yolov3.cfg weights/yolov3.weights <image list file>
```

Note that the desired batch size needs to be set in `cfg/yolov3.cfg`.
