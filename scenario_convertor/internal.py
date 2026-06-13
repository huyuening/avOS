import os


INTERNAL_RUN_ENV = "SCENARIO_CONVERTOR_INTERNAL"


def ensure_internal_converter_run():
    if os.environ.get(INTERNAL_RUN_ENV) != "1":
        raise SystemExit("Dataset converters are internal. Use the top-level convertor.py entrypoint.")


def run_converter_main(build_arg_parser, convert, argv=None):
    ensure_internal_converter_run()
    args = build_arg_parser().parse_args(argv)
    convert(args)
    return 0
