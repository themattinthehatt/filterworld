"""Command-line interface for filterworld."""

import argparse
from pathlib import Path


def _derive_output_path(video_path: str) -> str:
    """Derive default output path from input video path.

    Appends '_filtered' before the file extension.

    Args:
        video_path: path to the input video file

    Returns:
        default output path string
    """
    p = Path(video_path)
    return str(p.with_stem(f'{p.stem}_filtered'))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list to parse, defaults to sys.argv[1:]

    Returns:
        parsed namespace with video, model, config, and output fields
    """
    parser = argparse.ArgumentParser(
        prog='filterworld',
        description='Apply visual-model overlays to video.',
    )
    parser.add_argument(
        'video',
        help='path to input mp4 file',
    )
    parser.add_argument(
        'model',
        help='model path/identifier or pre-computed output file',
    )
    parser.add_argument(
        '--config',
        default=None,
        help='path to YAML config file',
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='output video path (default: derived from input filename)',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the filterworld CLI.

    Args:
        argv: argument list to parse, defaults to sys.argv[1:]
    """
    from filterworld.pipeline import Pipeline

    args = parse_args(argv)
    output_path = args.output or _derive_output_path(args.video)

    pipeline = Pipeline(
        video_path=args.video,
        model_path=args.model,
        config_path=args.config,
        output_path=output_path,
    )
    pipeline.run()


if __name__ == '__main__':
    main()
