def preload(parser):
    parser.add_argument(
        "--deforum-api",
        action="store_true",
        help="Enable the Deforum API",
        default=None,
    )
    parser.add_argument(
        "--deforum-run-now",
        type=str,
        help="Comma-delimited list of deforum settings files to run immediately on startup",
        default=None,
    )
    parser.add_argument(
        "--deforum-terminate-after-run-now",
        action="store_true",
        help="Whether to shut down the a1111 process immediately after completing the generations passed in to '--deforum-run-now'.",
        default=None,
    )