from PIL import Image
import imageio
import cv2
from decord import VideoReader, cpu
import numpy as np
import os


def uniform_sample(lst, n):
    assert n <= len(lst)
    m = len(lst)
    step = m // n  # Calculate the step size
    return [lst[i * step] for i in range(n)]



def _get_rawvideo_dec(video_path, image_processor, max_frames=64, image_resolution=224, video_framerate=1, s=None, e=None, num_threads=0):

    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0. else 0.
        end_time = end_time if end_time >= 0. else 0.
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, num_threads=num_threads)
        # vreader = VideoReader(video_path, num_threads=1)
        # num_threads=1 makes decoding a bit slower. so just set num_threads for Action Antonym.
        # wpq: https://github.com/dmlc/decord/issues/145
    else:
        raise FileNotFoundError(video_path)

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1

    if num_frames <= 0:
        raise ValueError("video path: {} error.".format(video_path))

    # T x 3 x H x W
    sample_fps = int(video_framerate)
    t_stride = int(round(float(fps) / sample_fps))


    all_pos = list(range(f_start, f_end + 1, t_stride))
    if len(all_pos) > max_frames:
        sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
    else:
        sample_pos = all_pos
    
    # print({
    #     'fps': fps,
    #     'sample_fps': sample_fps,
    #     'len(vreader)': len(vreader),
    #     'f_start': f_start,
    #     'f_end': f_end,
    #     'num_frames': num_frames,
    #     't_stride': t_stride,
    #     'all_pos': all_pos,
    #     'sample_pos': sample_pos,
    # })

    patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
    if image_processor:
        patch_images = [image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in patch_images]
    slice_len = len(patch_images)

    return  patch_images, slice_len



def read_frame_mod(video_path, image_processor, max_frames=16, image_resolution=224, video_framerate=3,
                   s=None, e=None, sample_fps=1):

    # Check if video path exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path {video_path} not found.")

    # Determine frame range
    frame_files = sorted(os.listdir(video_path))
    num_frames = len(frame_files)

    # Calculate frame indices
    fps = video_framerate
    f_start = 0 if s is None else max(int(s * fps), 0)
    f_end = min(num_frames - 1, int(e * fps)) if e is not None else num_frames - 1

    t_stride = max(int(round(float(fps) / sample_fps)), 1)
    frame_indices = range(f_start, f_end + 1, t_stride)

    # Process frames
    all_frames = []
    for idx in frame_indices:
        img_path = os.path.join(video_path, frame_files[idx])
        img = np.array(Image.open(img_path))
        all_frames.append(img)

        if len(all_frames) >= 100: # max_frames:
            break

    num_video_frames_sampled = min(max_frames, len(all_frames))
    patch_images = uniform_sample(all_frames, num_video_frames_sampled)
    if image_processor:
        patch_images = [image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in patch_images]
    slice_len = len(patch_images)

    return patch_images, slice_len




def read_gif_mod(video_path, image_processor, max_frames=16, image_resolution=224, video_framerate=25,
                 s=None, e=None, sample_fps=1):
    # Initialize data structures
    video = np.zeros((max_frames, 3, image_resolution, image_resolution), dtype=np.float64)

    # Load GIF file
    video_bytes = client.get(video_path)
    gif_reader = imageio.get_reader(io.BytesIO(video_bytes))
    num_frames = len(gif_reader)

    # Calculate frame indices
    fps = video_framerate
    f_start = 0 if s is None else max(int(s * fps), 0)
    f_end = min(num_frames - 1, int(e * fps)) if e is not None else num_frames - 1

    t_stride = max(int(round(float(fps) / sample_fps)), 1)
    frame_indices = range(f_start, f_end + 1, t_stride)

    # Process frames
    processed_frames = []
    for i, frame in enumerate(gif_reader):
        if i in frame_indices:
            img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            img_pil = Image.fromarray(img).resize((image_resolution, image_resolution))
            processed_frames.append(img_pil)

            if len(processed_frames) >= max_frames:
                break
    # Transform images
    patch_images = processed_frames
    if image_processor:
        patch_images = [image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in patch_images]
    slice_len = len(patch_images)

    return  patch_images, slice_len

    # patch_images = image_processor.preprocess(patch_images)['pixel_values']
    # slice_len = patch_images.shape[0]

    # Store video data
    # video[:slice_len, ...] = patch_images

    # return video, slice_len
