# Usage instead of DebugRenderer but save pictures instead of displaying to the screen
import pygame
import subprocess

from .debug_renderer import DebugRenderer

from .env_renderer import EnvRenderer
from pathlib import Path
from ..simulator.standard_config import STANDARD_CONFIG


class PictureRendererDebug(DebugRenderer):
    def __init__(self, picture_dir: Path, max_frames=2000, cfg=STANDARD_CONFIG, video_path=None, **kwargs):
        self._picture_dir = picture_dir
        self._video_path = video_path
        self._max_frames = max_frames
        self._cnt = 0
        super().__init__(cfg, **kwargs)

    def _get_display(self, w, h):
        # No display needed
        return None

    def _send_to_display(self):
        # Save the current frame to a file
        filename = self._picture_dir / f"frame_{self._cnt:04d}.png"
        pygame.image.save(self.cur_scene, str(filename))
        self._cnt += 1
        if self._cnt % 100 == 0:
            print(f"Saved frame {self._cnt} to {filename}")

        if self._cnt >= self._max_frames:
            if self._video_path:
                print(f"Creating video at {self._video_path}")
                self.make_video(self._video_path)
            self._end()

    def make_video(self, video_path: Path):
        subprocess.run([
            "ffmpeg",
            "-framerate", str(self._fps),
            "-i", f"{self._picture_dir}/frame_%04d.png",
            "-c:v", "libx264",
            "-y", str(video_path)
        ], check=True)


class PictureRendererEnv(EnvRenderer):
    def __init__(self, picture_dir: Path, max_frames=2000, cfg=STANDARD_CONFIG, video_path=None, **kwargs):
        self._picture_dir = picture_dir
        self._video_path = video_path
        self._max_frames = max_frames
        self._cnt = 0
        super().__init__(cfg, **kwargs)

    def _get_display(self, w, h):
        return pygame.surface.Surface((w, h))

    def _send_to_display(self):
        # Save the current frame to a file
        filename = self._picture_dir / f"frame_{self._cnt:04d}.png"
        pygame.image.save(self.screen, str(filename))
        self._cnt += 1
        if self._cnt >= self._max_frames:
            if self._video_path:
                print(f"Creating video at {self._video_path}")
                self.make_video(self._video_path)
            self._end()

    def make_video(self, video_path: Path):
        subprocess.run([
            "ffmpeg",
            "-framerate", str(self.fps),
            "-i", f"{self._picture_dir}/frame_%04d.png",
            "-c:v", "libx264",
            "-y", str(video_path)
        ], check=True)

    def _end(self):
        self._running = False
        pygame.quit()
        exit(0)
