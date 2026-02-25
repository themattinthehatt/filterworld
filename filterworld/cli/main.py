"""Command-line interface for filterworld."""

import argparse
import logging
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
        parsed namespace with subcommand and its arguments
    """
    parser = argparse.ArgumentParser(
        prog='filterworld',
        description='Apply visual-model overlays to video.',
    )
    subparsers = parser.add_subparsers(dest='command')

    # --- run subcommand ---
    parser_run = subparsers.add_parser(
        'run',
        help='render video with visual-model overlays',
    )
    parser_run.add_argument(
        'video',
        help='path to input mp4 file',
    )
    parser_run.add_argument(
        'model',
        help='model path/identifier or pre-computed output file',
    )
    parser_run.add_argument(
        '--config',
        default=None,
        help='path to YAML config file',
    )
    parser_run.add_argument(
        '--output', '-o',
        default=None,
        help='output video path (default: derived from input filename)',
    )

    # --- precompute subcommand ---
    parser_precompute = subparsers.add_parser(
        'precompute',
        help='precompute PCA weights from video features',
    )
    parser_precompute.add_argument(
        'video',
        help='path to input mp4 file',
    )
    parser_precompute.add_argument(
        'model',
        help='model path/identifier (e.g. facebook/dino-vits16)',
    )
    parser_precompute.add_argument(
        '--output', '-o',
        required=True,
        help='output .npz path for PCA weights',
    )
    parser_precompute.add_argument(
        '--max-frames',
        type=int,
        default=200,
        help='max frames to use for PCA fitting (default: 200)',
    )

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        raise SystemExit(1)

    return args


def main(argv: list[str] | None = None) -> None:
    """Entry point for the filterworld CLI.

    Args:
        argv: argument list to parse, defaults to sys.argv[1:]
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s    %(message)s',
    )

    args = parse_args(argv)

    if args.command == 'run':
        from filterworld.pipeline import Pipeline

        output_path = args.output or _derive_output_path(args.video)
        pipeline = Pipeline(
            video_path=args.video,
            model_path=args.model,
            config_path=args.config,
            output_path=output_path,
        )
        pipeline.run()

    elif args.command == 'precompute':
        from filterworld.precompute import precompute_pca

        precompute_pca(
            video_path=args.video,
            model_path=args.model,
            output_path=args.output,
            max_frames=args.max_frames,
        )


if __name__ == '__main__':
    main()
