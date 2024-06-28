from abc import abstractmethod, ABC


class Logger(ABC):
    def __init__(self):
        pass

    def __call__(self, *msg):
        self.write(*msg)

    @abstractmethod
    def write(self, *msg):
        pass


class ConsoleLogger(Logger):
    def write(self, *msg):
        print(*msg)


class FileLogger(Logger):
    def __init__(self, fname):
        super().__init__()
        self.fname = fname

        with open(fname, "w") as f:
            f.write("\n")

    def write(self, *msg):
        with open(self.fname, "a") as f:
            print(*msg, file=f, end="", sep="")
