# Part 1
Please check the `barcode_detection.ipynb` notebook for some examples.

### Algorithm Overview

1. The algorithm first transforms the image to the frequency domain and finds the spatial angle at which the barcode is oriented. This works because the barcode is typically the most periodic structure in the image, so its orientation can be inferred from the dominant frequency responses.

2. The image is then rotated so that the barcode is aligned with the **x-axis**.

3. A Sobel filter is applied along that axis to maximize the response to the barcode’s edges.

4. Finally, the rotated image is transformed to the Fourier domain again, and only the periodic patterns corresponding to the barcode are preserved.

5. We then iFFT that masked frequency space, binarize it, and find the start/end points of the barcode.

For testing the algorithm type:

```
python barcode_detector.py /path/to/pngimage
```

I will save the cropped barcode in a file called `./output.png`. Also, it prints the barcode ROI coordinates in the rotated image.

### Potential improvements

Definitely moving to C++. Also, checking everything within the barcode white patch would increase the precision. Reducing the picture size would help with timing (it's commonly used in detection algorithms).



# Part 2

The edge device captures a frame, detects the barcode region of interest (ROI), crops it, and optionally compresses the crop to JPEG at quality 85–90.
It then builds a payload containing the JPEG bytes, timestamp, device ID, and ROI coordinates, and sends this data to the cloud through an HTTPS POST request using a client that waits for a success acknowledgment.
On the cloud side, the request first reaches an API Gateway, which immediately returns an acknowledgment and forwards the data into an ingestion queue.
A barcode processing service then reads the ROI from the queue, decodes the barcode, classifies any reflection or occlusion issues, and stores the cropped ROI in object storage while saving the decoded results and metadata like image ID, device ID, timestamp, barcode value, and occlusion type into a metadata database.
After processing is complete, the cloud sends a final acknowledgment back to the edge device.





                           ┌───────────────────────────────────────────┐
                           │                 EDGE DEVICE               │
                           └───────────────────────────────────────────┘
                                          │
                                          │ (1) Capture Frame
                                          ▼
                               ┌─────────────────────┐
                               │ Barcode ROI Detector│ 
                               └─────────────────────┘
                                          │ (ROI bbox)
                                          ▼
                               ┌────────────────────┐
                               │   ROI Cropper      │
                               └────────────────────┘
                                          │ (cropped Barcode)
                                          ▼
                               ┌─────────────────────────┐
                               │ Optional JPEG Compressor│
                               │   (Q=85–90)             │
                               └─────────────────────────┘
                                          │
                               ┌──────────────────────────┐
                               │ Payload Builder          │
                               │  - ROI bytes (JPEG)      │
                               │  - timestamp             │
                               │  - device id             │
                               │  - ROI coordinates       │
                               └──────────────────────────┘
                                          │
                                          ▼
                           ┌───────────────────────────────────────────┐ 
                           │            HTTPS Client                   │ 
                           │    - Send the data to the cloud server    │
                           │    - Wait to receive the success ack      │
                           └───────────────────────────────────────────┘
                                          │
                                          │  HTTPS POST
                                          ▼

                            ──────────────────────────────────────────

                       ┌───────────────────────────────────────────┐
                       │                CLOUD SIDE                 │
                       └───────────────────────────────────────────┘

                           ┌─────────────────────────┐
         (Immediate ACK) → │   API Gateway / HTTPS   │
                           └─────────────────────────┘
                                       │
                                       ▼
                           ┌─────────────────────────┐
                           │   Ingestion Queue       │
                           └─────────────────────────┘
                                       │
                                       ▼
                           ┌─────────────────────────┐
                           │  Barcode Processor      │
                           │  - Read ROI             │
                           │  - Decode barcode       │
                           │  - Classify reflection/ │
                           │    occlusion/strap      │
                           └─────────────────────────┘
                                   │              │
               (results + metadata)│              │
                                   ▼              │
         ┌──────────────────────────────────┐     │
         │   Object Storage                 │     │
         └──────────────────────────────────┘     │
                                   │              │
                                   │              │
                                   ▼              ▼
                     ┌──────────────────────────────────┐
                     │        Metadata DB               │
                     │        - image_id                │
                     │        - device_id               │
                     │        - timestamp               │
                     │        - barcode_value           │
                     │        - occlusion_type          │
                     └──────────────────────────────────┘
                                       │
                                       ▼
                                     (ACK to edge device)



# Part 3
Trained a ResNet50. Since the dataset was small, I applied some extensive data augmentations and kept a small batch size (4). Also performed validation at the end of each epoch, and had a scheduler to watch the ```accuracy``` (not loss) of
validation data. If it plateaus, then the learning rate is reduced. The accuracy I got was around 94% for the test data (100 completely different samples from the train/val to avoid data leakage).


To train the model, ensure that you place the reflection/strap folders next to the code. Then run

`
python classification.py . train
`

This would result in a model with the best validation accuracy saved as ```./best_model.pth```. 

For test:


`
python classification.py . test
`

To run the code, you'd need numpy, opencv, torchvision, torch, and PIL.
