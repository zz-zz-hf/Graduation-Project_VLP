"""
实体类，保存线程执行PnP算法的数据
"""
class VLPThreadEntity:
    def __init__(self):
        self.detectedres_filename = []  # txt和tiff的文件名
        self.pnpres = {}   #{standard_index:[pnppos]……}
        self.pnpalg_times = []
        self.thread_time=-1

    def add_detectedres_filename(self,detectedres_filename):
        self.detectedres_filename.append(detectedres_filename)

    def add_pnpres(self,key,value):
        if key not in self.pnpres:
            self.pnpres[key] = []
        self.pnpres[key].append(value)

    def add_pnpalg_times(self, pnpalg_times):
        self.pnpalg_times=pnpalg_times

    def add_thread_time(self,thread_time):
        self.thread_time=thread_time