
import time
from pathlib import Path

START_TIME = time.perf_counter()


# ---------------------------------------------------------------------------
# console out, debug, etc:
# ---------------------------------------------------------------------------

ANSI = {
#foreground (bright):
#    "red": "\x1b[0;31m",
#    "green": "\x1b[0;32m",
#    "blue": "\x1b[0;34m",
#    "yellow": "\x1b[0;33m",
#    "cyan": "\x1b[0;36m",
#    "pink": "\x1b[0;35m",
    "red": "\x1b[0;91m",
    "green": "\x1b[0;92m",
    "blue": "\x1b[0;94m",
    "yellow": "\x1b[0;93m",
    "cyan": "\x1b[0;96m",
    "pink": "\x1b[0;95m",
#background:
    "RED": "\x1b[41m",
    "GREEN": "\x1b[42m",
    "BLUE": "\x1b[44m",
    "YELLOW": "\x1b[43m",
    "CYAN": "\x1b[46m",
    "PINK": "\x1b[45m",
#misc:
    "bold": "\x1b[1m",
    "italic": "\x1b[3m",
    "under": "\x1b[4m",
    "strike": "\x1b[9m",
    "reset": "\x1b[0m",
    "prefix": "\x1b[",
}
ANSI['good'] = ANSI['green']
ANSI['error'] = ANSI['red']
ANSI['warn'] = ANSI['yellow']
ANSI['info'] = ANSI['cyan']
ANSI['debug'] = ANSI['blue']
ANSI['memo'] = ANSI['pink']

    
def lineno(level = 0) -> int:
    import inspect
    caller_frame_record = inspect.stack()[1 + level]
    info = inspect.getframeinfo(caller_frame_record[0])
    return info.lineno

def out(*args, **kwargs):
    buf = ANSI[kwargs["color"]] #if "color" in kwargs else ""
    kwargs['depth'] = kwargs.get('depth', 0) + 1
#    for arg in args:
    for i, arg in enumerate(args):
#        which = f".{i + 1}" if len(args) > 1 else ""
#        buf += f"{arg}  @{lineno(+1)}{which}"
#        print(f"{arg}  @{lineno(+1)}{which}", file = sys.stderr)
        buf += arg + " "
#    buf += " @" + str(lineno(kwargs['depth'] if 'depth' in kwargs else +1))
    buf += kwargs['srcline'] if 'srcline' in kwargs else f" @{lineno(kwargs['depth'])}"
    print(buf + ANSI['reset'] if ANSI['prefix'] in buf else buf)  #, file = sys.stderr) #, end="")
    return args[-1]  #for inlining last arg

def good(*args, **kwargs):
    kwargs['depth'] = kwargs.get('depth', 0) + 1
    return out(*args, **kwargs, color="good")
def error(*args, **kwargs):
    kwargs['depth'] = kwargs.get('depth', 0) + 1
    return out(*args, **kwargs, color="error")
def warn(*args, **kwargs):
    kwargs['depth'] = kwargs.get('depth', 0) + 1
    return out(*args, **kwargs, color="warn")
def info(*args, **kwargs):
    kwargs['depth'] = kwargs.get('depth', 0) + 1
    return out(*args, **kwargs, color="info")
def debug(*args, **kwargs):
    kwargs['depth'] = kwargs.get('depth', 0) + 1
    return out(*args, **kwargs, color="debug")
def memo(*args, **kwargs):
    kwargs['depth'] = kwargs.get('depth', 0) + 1
    return out(*args, **kwargs, color="memo")

def relpath(file: Path, basedir: Path):
##    return "./" + Path(path).relative_to(Path(BASE_DIR))
#    return "./" + path.relative_to(BASE_DIR)
    try:
        return "./" + str(file.relative_to(basedir.parent))
    except Exception as e:
        debug(e)
        return file

def seconds(duration: float) -> str:
    scale = {
        360: "hr",
        60: "min",
        1: "sec",
        1e-3: "msec",
        1e-6: "usec",
    }
    for factor, units in scale.items():
        if duration >= factor:
            return f"{duration / factor:.1f} {units}".replace(".0", "")
    return "(no time)"


def elapsed(since: float = None) -> str:
    if not hasattr(elapsed, "previous"):
        global START_TIME
        elapsed.previous = START_TIME
    now = time.perf_counter()
    retval = seconds(now - (since if since is not None else elapsed.previous))
    elapsed.previous = now
    return retval

#eof