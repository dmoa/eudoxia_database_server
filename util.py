import signal

def set_signal_to_nothing():
    def handle_signal(signum, frame):
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$   Signal caught, exiting   $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        exit(0)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


def read_entire_file(file_path):
    f = open(file_path, "r", encoding="utf-8")
    data = f.read()
    f.close()
    return data

def read_entire_file_rb(file_path):
    f = open(file_path, "rb")
    data = f.read()
    f.close()
    return data