# Last modified: 2025-10-17

from typing import List, Union

from moviepy.editor import ImageSequenceClip


def generate_video_from_image_paths(
    image_paths: List[str],
    output_path: str,
    fps: int = 30,
    bitrate: int = 1000,
    verbose=True,
):
    """Create a video from a list of image file paths using MoviePy.

    Args:
        image_paths: list of paths to image files
        output_path: path where the output video will be saved
        fps: frames per second for the output video
        bitrate: bitrate (in K) for the output video
    """
    if not image_paths:
        raise ValueError("The list of image paths is empty.")

    clip = ImageSequenceClip(image_paths, fps=fps)
    clip = clip.set_duration(len(image_paths) / fps)

    # Force even height to avoid yuv420p encoding errors
    clip = clip.crop(x1=0, y1=0, x2=clip.w, y2=clip.h - (clip.h % 2))

    clip.write_videofile(
        output_path,
        codec="libx264",
        bitrate=f"{bitrate}K",
        verbose=verbose,
        logger=None if not verbose else "bar",
    )
