import argparse
import utils
import cv2
import torch


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "engine", 
        type=str, 
        help="The file path of the TensorRT engine."
    )

    parser.add_argument(
        "image", 
        type=str, 
        help="The file path of the image provided as input for inference."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="The path to output the inference visualization."
    )

    parser.add_argument(
        "--inference-size", 
        type=str, 
        default="512x512", 
        help="The height and width that the image is resized to for inference."
             " Denoted as (height)x(width)."
    )

    parser.add_argument(
        "--peak-window",
        type=str,
        default="7x7",
        help="The size of the window used when finding local peaks. Denoted as "
             " (window_height)x(window_width)."
    )

    parser.add_argument(
        '--peak-threshold',
        type=float,
        default=0.5,
        help="The heatmap threshold to use when finding peaks.  Values must be "
             " larger than this value to be considered peaks."
    )

    parser.add_argument(
        '--line-thickness',
        type=int,
        default=1,
        help="The line thickness for drawn boxes"
    )

    args = parser.parse_args()

    # Parse inference height, width from arguments
    inference_size = tuple(int(x) for x in args.inference_size.split('x'))
    peak_window = tuple(int(x) for x in args.peak_window.split('x'))

    if args.output is None:
        output_path = '.'.join(args.image.split('.')[:-1]) + "_output.jpg"
    else:
        output_path = args.output

    # Create offset grid
    offset_grid = utils.make_offset_grid(inference_size).to("cuda")

    # Load model
    model = utils.load_trt_engine_wrapper(
        args.engine,
        input_names=["input"],
        output_names=["heatmap", "vectormap"]
    )

    # Load image
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Pad and resize image (aspect ratio preserving resize)
    image, _, _ = utils.pad_resize(image, inference_size)

    with torch.no_grad():

        # Format image for inference
        x = utils.format_bgr8_image(image)
        x = x.to("cuda")

        # Execute model
        heatmap, vectormap = model(x)

        # Scale and offset vectormap
        keypointmap = utils.vectormap_to_keypointmap(
            offset_grid,
            vectormap
        )

        # Find local peaks
        peak_mask = utils.find_heatmap_peak_mask(
            heatmap, 
            peak_window,
            args.peak_threshold
        )

        # Extract keypoints at local peak
        keypoints = keypointmap[0][peak_mask[0, 0]]
    
    # Draw
    vis_image = utils.draw_box(
        image, 
        keypoints,
        color=(118, 186, 0),
        thickness=args.line_thickness
    )

    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, vis_image)
