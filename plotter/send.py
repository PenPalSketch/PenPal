"""
Sends G-code to the plotter's Arduino (running GRBL) over USB.

    python plotter/send.py --list                  # show serial ports
    python plotter/send.py file.gcode --dry-run    # practice run with a pretend Arduino
    python plotter/send.py file.gcode --check      # the Arduino checks every line, motors don't move
    python plotter/send.py file.gcode              # plot it
    python plotter/send.py --test                  # self-tests, no hardware needed

Every file goes through the simulator first and isn't sent if it has problems (--force overrides).

GRBL doesn't know where the pen is when it starts: it calls that spot X0 Y0. The Arduino restarts
whenever a program connects, so put the pen at the home corner before connecting. Every job ends back there.
Only one program can use the port at a time, so close UGS / CNCjs / the Arduino IDE serial monitor first.
"""
import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plotter.gcode import BAUD, PORT
from plotter.simulate import simulate, summary

# Common GRBL error codes, to explain a rejected line
ERRORS = {
    "1": "the line doesn't start with a command letter",
    "2": "a number is badly formatted",
    "9": "GRBL is locked in alarm; run again with --unlock",
    "20": "the firmware doesn't support this command",
    "22": "G1 was sent before any feed rate (F)",
}


class PlotterError(Exception):
    pass


def list_ports():
    from serial.tools import list_ports
    return list(list_ports.comports())


# The single USB port that looks like an Arduino (Unos show up as "Arduino", clones as "CH340").
# Bluetooth ports are skipped: opening one can hang.
def find_port():
    ports = list_ports()
    usb = [p for p in ports if "bluetooth" not in p.description.lower() and
           any(k in f"{p.description} {p.manufacturer or ''}".lower() for k in ("arduino", "ch340", "ch341", "usb"))]
    if len(usb) == 1:
        return usb[0].device
    listing = "\n".join(f"  {p.device}: {p.description}" for p in ports) or "  (none)"
    if not usb:
        raise PlotterError(f"No Arduino found. Is it plugged in by USB? Ports seen:\n{listing}")
    raise PlotterError(f"Several USB ports found. Pick one with --port, or set PORT in plotter/gcode.py:\n{listing}")


# A pretend Arduino running GRBL, for practice runs and tests. Says ok to every line.
# fail_on: text that makes it reply with an error instead. state: what a "?" status query reports.
class FakeGrbl:
    def __init__(self, fail_on=None, state="Idle"):
        self.sent = []
        self.replies = ["Grbl 1.1h ['$' for help]"]
        self.fail_on, self.state = fail_on, state

    def write(self, data):
        if data == b"?":
            self.replies.append(f"<{self.state}|MPos:0.000,0.000,0.000|FS:0,0>")
            return
        text = data.decode("latin-1")
        if data in (b"!", b"\x18"):
            self.sent.append(text)
            return
        for line in text.splitlines():
            if line.strip():
                self.sent.append(line)
                if line == "$X":
                    self.state = "Idle"
                self.replies.append("error:20" if self.fail_on and self.fail_on in line else "ok")

    def readline(self):
        return (self.replies.pop(0) + "\r\n").encode() if self.replies else b""

    def reset_input_buffer(self):
        pass

    def close(self):
        pass


# Reads replies until one matches `want`, or gives up after `seconds`. Returns it, or None.
def _wait_for(ser, want, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        reply = ser.readline().decode(errors="replace").strip()
        if reply and want(reply):
            return reply
    return None


# Sends one line and waits for GRBL's "ok". GRBL says ok once the line is queued, not when the move
# finishes, so this only waits long when its queue (about 16 moves) is full.
def _command(ser, line, timeout=60):
    ser.write((line + "\n").encode())
    deadline = time.time() + timeout
    while time.time() < deadline:
        reply = ser.readline().decode(errors="replace").strip()
        if reply == "ok":
            return
        if reply.startswith(("error", "ALARM")):
            code = reply.split(":", 1)[-1].strip()
            hint = f" ({ERRORS[code]})" if code in ERRORS else ""
            raise PlotterError(f"GRBL rejected {line!r} with {reply}{hint}")
        # Anything else ([MSG:...], status reports) is just information
    raise PlotterError(f"No reply from GRBL to {line!r} after {timeout} s")


def connect(port=PORT, baud=BAUD, log=print):
    import serial

    port = port or find_port()
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except serial.SerialException as e:
        raise PlotterError(f"Couldn't open {port}: {e}\nIs another program (UGS, CNCjs, Arduino IDE) using it?")

    # Opening the port restarts the Arduino, and GRBL prints a banner when it starts.
    # If this board doesn't restart, a soft reset (Ctrl-X) makes GRBL print it.
    log(f"Connected to {port}, waiting for GRBL...")
    banner = _wait_for(ser, lambda r: r.lower().startswith("grbl"), 3)
    if banner is None:
        ser.write(b"\x18")
        banner = _wait_for(ser, lambda r: r.lower().startswith("grbl"), 3)
    if banner is None:
        ser.close()
        raise PlotterError(f"No reply from GRBL on {port}. Is GRBL flashed? "
                           "(The stepper test sketch in HW/ doesn't understand G-code.)")
    log(banner)
    time.sleep(0.2)
    ser.reset_input_buffer()
    return ser


# Stops right away: feed hold, then soft reset. Resetting turns the spindle output off, which lifts the pen.
def stop(ser):
    ser.write(b"!")
    time.sleep(0.2)
    ser.write(b"\x18")


def _check_state(ser, unlock, log):
    ser.write(b"?")
    status = _wait_for(ser, lambda r: r.startswith("<"), 2)
    state = re.split(r"[|,>]", status[1:])[0] if status else "unknown"
    if state.lower().startswith("alarm"):
        if not unlock:
            raise PlotterError("GRBL is in alarm (usually after a reset mid-move). Put the pen back at the "
                               "home corner, then run again with --unlock (sends $X).")
        log("Unlocking GRBL ($X)")
        _command(ser, "$X")


# Sends every line, then waits until the last move has finished
def stream(ser, lines, on_progress=None):
    lines = [re.sub(r"\(.*?\)", "", l).split(";")[0].strip() for l in lines]
    lines = [l for l in lines if l]
    try:
        for i, line in enumerate(lines, 1):
            _command(ser, line)
            if on_progress:
                on_progress(i, len(lines), line)
        _command(ser, "G4 P0", timeout=600)   # GRBL answers once every queued move is done
    except (PlotterError, KeyboardInterrupt):
        stop(ser)
        raise


# The whole job: simulate, connect, check GRBL is ready, send, disconnect.
# check: GRBL's check mode ($C) reads every line and reports errors without moving anything.
# ser: an already-open connection (or a FakeGrbl) instead of connecting.
def send_gcode(lines, port=PORT, dry_run=False, check=False, unlock=False, force=False,
               on_progress=None, log=print, ser=None):
    sim = simulate(lines)
    if sim["warnings"] and not force:
        raise PlotterError("The simulator found problems, so nothing was sent:\n  " + "\n  ".join(sim["warnings"]))
    log(summary(sim))

    if ser is None:
        ser = FakeGrbl() if dry_run else connect(port, log=log)
    try:
        _check_state(ser, unlock, log)
        if check:
            _command(ser, "$C")
        stream(ser, lines, on_progress)
        if check:
            ser.write(b"$C\n")      # turn check mode off; GRBL resets itself after this
    finally:
        ser.close()


def self_test():
    results = []
    def check(name, ok):
        print("PASS" if ok else "FAIL", name)
        results.append(ok)
    quiet = lambda *a: None

    job = ["G21", "G90", "M5", "G4 P0.15", "G0 X10 Y10 ; travel", "M3 S90", "G4 P0.15",
           "G1 X20 Y10 F1500", "M5", "G4 P0.15", "G0 X0 Y0"]

    fake = FakeGrbl()
    send_gcode(job, ser=fake, log=quiet)
    check("sends every line in order, comments removed",
          fake.sent == [l.split(";")[0].strip() for l in job] + ["G4 P0"])

    fake = FakeGrbl(fail_on="M3")
    try:
        send_gcode(job, ser=fake, log=quiet)
        check("a rejected line stops the job", False)
    except PlotterError as e:
        check("a rejected line stops the job", "M3 S90" in str(e) and "doesn't support" in str(e))
        check("nothing is sent after the rejected line", fake.sent[-3:] == ["M3 S90", "!", "\x18"])

    fake = FakeGrbl()
    try:
        send_gcode(["G21", "G90", "M3 S90", "G4 P0.15", "G0 X10 Y10", "M5"], ser=fake, log=quiet)
        check("the simulator blocks bad G-code", False)
    except PlotterError as e:
        check("the simulator blocks bad G-code", "nothing was sent" in str(e) and fake.sent == [])

    fake = FakeGrbl(state="Alarm")
    try:
        send_gcode(job, ser=fake, log=quiet)
        check("refuses to run while GRBL is in alarm", False)
    except PlotterError as e:
        check("refuses to run while GRBL is in alarm", "--unlock" in str(e) and fake.sent == [])
    fake = FakeGrbl(state="Alarm")
    send_gcode(job, ser=fake, log=quiet, unlock=True)
    check("--unlock sends $X, then the job", fake.sent[0] == "$X" and fake.sent[-1] == "G4 P0")

    fake = FakeGrbl()
    send_gcode(job, ser=fake, log=quiet, check=True)
    check("--check wraps the job in $C", fake.sent[0] == "$C" and fake.sent[-1] == "$C")

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


def main():
    p = argparse.ArgumentParser(description="Send G-code to the plotter's Arduino (GRBL).")
    p.add_argument("file", nargs="?", help=".gcode file to send")
    p.add_argument("--port", default=PORT, help="serial port, e.g. COM7 (default: find the Arduino)")
    p.add_argument("--dry-run", action="store_true", help="practice with a pretend Arduino")
    p.add_argument("--check", action="store_true", help="GRBL checks every line without moving")
    p.add_argument("--unlock", action="store_true", help="clear a GRBL alarm ($X) before sending")
    p.add_argument("--force", action="store_true", help="send even if the simulator found problems")
    p.add_argument("--list", action="store_true", help="list serial ports")
    p.add_argument("--test", action="store_true", help="run the self-tests")
    a = p.parse_args()

    if a.test:
        sys.exit(self_test())
    if a.list:
        for port in list_ports():
            print(f"{port.device}: {port.description}")
        return
    if not a.file:
        p.error("give a .gcode file, or --list / --test")

    def progress(i, n, line):
        print(f"[{i}/{n}] {line}")

    try:
        send_gcode(Path(a.file).read_text().splitlines(), port=a.port, dry_run=a.dry_run, check=a.check,
                   unlock=a.unlock, force=a.force, on_progress=progress)
        if a.check:
            print("GRBL accepted every line.")
        elif a.dry_run:
            print("Practice run done. Nothing was sent to a real Arduino.")
        else:
            print("Done. The pen should be back at the home corner.")
    except PlotterError as e:
        print(f"Stopped: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopped. GRBL was reset, so it has lost track of the pen: "
              "put the pen back at the home corner before the next job.")
        sys.exit(1)


if __name__ == "__main__":
    main()
