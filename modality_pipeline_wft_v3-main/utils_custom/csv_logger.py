import csv
import os

# class CSVLogger:
#     def __init__(self, log_dir, filename="log.csv", resume=False):
#         """
#         log_dir: directory to save the CSV file
#         filename: name of the CSV file
#         resume: if True, will append to existing CSV, else overwrite
#         """
#         os.makedirs(log_dir, exist_ok=True)
#         self.filepath = os.path.join(log_dir, filename)
#         self.resume = resume
#         self.header_written = False if not resume or not os.path.exists(self.filepath) else True
#         self.columns = None

#     def log(self, metrics: dict):
#         """
#         metrics: a dictionary of {column_name: value} pairs
#         Example:
#             metrics = {
#                 "epoch": 1,
#                 "train_loss": 0.123,
#                 "val_loss": 0.234,
#                 "psnr": 24.5,
#                 "ssim": 0.91
#             }
#         """
#         if not self.header_written:
#             self.columns = list(metrics.keys())
#             with open(self.filepath, mode='w', newline='') as f:
#                 writer = csv.DictWriter(f, fieldnames=self.columns)
#                 writer.writeheader()
#             self.header_written = True

#         # Write data row
#         with open(self.filepath, mode='a', newline='') as f:
#             writer = csv.DictWriter(f, fieldnames=self.columns)
#             writer.writerow(metrics)


import csv, os

class CSVLogger:
    def __init__(self, log_dir, filename="log.csv", resume=False):
        os.makedirs(log_dir, exist_ok=True)
        self.filepath = os.path.join(log_dir, filename)
        self.columns = None
        self.header_written = False

        if resume and os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0:
            with open(self.filepath, newline='') as f:
                header = next(csv.reader(f), None)
            if header:
                self.columns = header
                self.header_written = True

    def log(self, metrics: dict):
        if not self.header_written:
            self.columns = list(metrics.keys())
            with open(self.filepath, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=self.columns).writeheader()
            self.header_written = True

        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns, restval='')
            writer.writerow({k: metrics.get(k, '') for k in self.columns})