from amlb.utils import as_cmd_args, call_script_in_same_dir, dir_of


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(*args, **kwargs):
    from .exec import run
    return run(*args, **kwargs)


def docker_commands(*args, setup_cmd=None):
    return """
RUN {here}/setup.sh {args}
{cmd}
EXPOSE 54321
EXPOSE 54322
""".format(
        here=dir_of(__file__, True),
        args=' '.join(as_cmd_args(*args)),
        cmd="RUN {}".format(setup_cmd) if setup_cmd is not None else ""
    )


#There is no network isolation in Singularity,
#so there is no need to map any port.
#If the process inside the container binds to an IP:port,
#it will be immediately reachable on the host.
def singularity_commands(*args, setup_cmd=None):
    return """
{here}/setup.sh {args}
{cmd}
""".format(
        here=dir_of(__file__, True),
        args=' '.join(as_cmd_args(*args)),
        cmd="{}".format(setup_cmd) if setup_cmd is not None else ""
    )


__all__ = (setup, run, docker_commands)
