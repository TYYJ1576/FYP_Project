import cv2
import os
import glob

def create_video_from_images(image_folder, output_video_path, fps=24):
    # Get list of all images in the folder
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.png')))
    if not image_files:
        print("No images found in the specified folder.")
        return

    # Read the first image to get video dimensions
    frame = cv2.imread(image_files[0])
    if frame is None:
        print(f"Could not read image {image_files[0]}")
        return
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' for .avi files
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image_file in image_files:
        img = cv2.imread(image_file)
        if img is None:
            print(f"Could not read image {image_file}")
            continue
        video_writer.write(img)

    video_writer.release()
    print(f"Video saved at {output_video_path}")

if __name__ == "__main__":
    image_folder = "work_dirs/stuttgart_00"  # Replace with your folder path
    output_video_path = "output_video.mp4"      # Desired output video file
    fps = 24                                    # Frames per second

    create_video_from_images(image_folder, output_video_path, fps)
