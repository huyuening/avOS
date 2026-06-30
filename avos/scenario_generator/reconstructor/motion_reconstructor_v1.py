"""Compatibility wrapper for the original trajectory reconstructor."""

try:
    from .motion_reconstructor import main, object_track_reconstruction, process_single_file
except ImportError:
    from reconstructor.motion_reconstructor import main, object_track_reconstruction, process_single_file


if __name__ == "__main__":
    main()
