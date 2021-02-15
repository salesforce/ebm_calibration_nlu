import torch
import calibration as cal 

class TScalCalibrator:
    def __init__(self, num_bins):
        self._num_bins = num_bins
        self._temp = None

    def train_calibration(self, zs, ys):
        best_ece = 10000
        log_zs = torch.log(zs)
        for temp in [0.1, 0.5, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]:
            probs_now = torch.softmax(log_zs / temp, dim = -1)
            ece_now = cal.get_ece(probs_now.numpy(), ys.numpy(), num_bins = self._num_bins)
            #print(temp, ece_now)
            if ece_now < best_ece:
                best_ece = ece_now
                self._temp = temp
        print('tScal trained temp:', self._temp)

    def calibrate(self, zs):
        log_zs = torch.log(zs)
        new_probs = torch.softmax(log_zs / self._temp, dim = -1)
        return new_probs
