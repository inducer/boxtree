import time


class TimeRecorder:
    # functions in this class need to be called collectively
    def __init__(self, name, comm, logger):
        self.name = name
        self.comm = comm
        self.logger = logger
        self.start_time = None

        self.comm.Barrier()
        if self.comm.Get_rank() == 0:
            self.start_time = time.time()

    def record(self):
        self.comm.Barrier()
        if self.comm.Get_rank() == 0:
            self.logger.info("{0} time: {1} sec.".format(
                self.name,
                time.time() - self.start_time
            ))
