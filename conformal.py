import numpy as np


class ConformalPredictor:
    def __init__(self, alpha):
        self.alpha = float(alpha)
        self.q = None
        self.n_calib = 0

    @staticmethod
    def _adjusted_quantile(values, alpha):
        values = np.asarray(values)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("Calibration scores must be a non-empty 1-D array.")
        n = values.shape[0]
        q_level = np.ceil((n + 1) * (1 - float(alpha))) / n
        return np.quantile(values, q_level, method="higher")

    @property
    def quantile(self):
        if self.q is None:
            raise RuntimeError("Quantile not available before fit().")
        return self.q

    def fit(self, calibration_scores):
        calibration_scores = np.asarray(calibration_scores)
        if calibration_scores.ndim != 1 or calibration_scores.size == 0:
            raise ValueError("Calibration scores must be a non-empty 1-D array.")
        self.n_calib = calibration_scores.size
        self.q = self._adjusted_quantile(calibration_scores, self.alpha)
        self._sorted_calib = np.sort(calibration_scores)

        return self

    def predict(self, test_scores):
        if self.q is None:
            raise ValueError("Call fit(calibration_scores) before predict(test_scores)")

        test_scores = np.asarray(test_scores)

        return test_scores > self.q

    def p_values(self, test_scores, calibration_scores):
        test_scores = np.asarray(test_scores)

        if calibration_scores is None:
            if not hasattr(self, "_sorted_calib"):
                raise ValueError(
                    "No calibration scores provided and predictor not fitted with calibration data."
                )
            sorted_calib = self._sorted_calib
        else:
            sorted_calib = np.sort(np.asarray(calibration_scores))

        n = sorted_calib.size
        if n == 0:
            raise ValueError("Calibration scores must be non-empty")

        idx = np.searchsorted(sorted_calib, test_scores, side="left")
        counts_ge = n - idx
        return (counts_ge + 1) / (n + 1)


class AdaptiveConformalPredictor:
    def __init__(self, alpha, maxlen=50000, recompute_every=256):
        self.alpha = float(alpha)
        self.maxlen = int(maxlen)
        self.recompute_every = int(recompute_every)
        self.base = ConformalPredictor(alpha=self.alpha)

        self._calib_fifo = None
        self._calib_sorted = None
        self._since_last_recompute = 0

    def fit(self, calibration_scores):
        cs = np.asarray(calibration_scores, dtype=float).ravel()
        if cs.size == 0:
            raise ValueError("Calibration scores must be non-empty.")
        if cs.size > self.maxlen:
            cs = cs[-self.maxlen :]
        self._calib_fifo = cs.tolist()
        self._rebuild_sorted_and_q()
        return self

    @property
    def quantile(self):
        return self.base.quantile

    @property
    def calib_size(self):
        return 0 if self._calib_fifo is None else len(self._calib_fifo)

    def _rebuild_sorted_and_q(self):
        arr = np.asarray(self._calib_fifo, dtype=float)
        self._calib_sorted = np.sort(arr)
        self.base.fit(arr)
        self._since_last_recompute = 0

    def _maybe_recompute(self):
        self._since_last_recompute += 1
        if self._since_last_recompute >= self.recompute_every:
            self._rebuild_sorted_and_q()

    def p_values(self, test_scores):
        if self._calib_sorted is None:
            raise RuntimeError("Call fit() first.")
        test_scores = np.asarray(test_scores, dtype=float).ravel()
        n = self._calib_sorted.size
        idx = np.searchsorted(self._calib_sorted, test_scores, side="left")
        counts_ge = n - idx
        return (counts_ge + 1) / (n + 1)

    def decide(self, test_scores):
        test_scores = np.asarray(test_scores, dtype=float).ravel()
        return (test_scores > self.quantile).astype(int)

    def update_with_normal(self, score):
        self._calib_fifo.append(float(score))
        if len(self._calib_fifo) > self.maxlen:
            self._calib_fifo.pop(0)
        self._maybe_recompute()

    def stream_decide(
        self,
        scores,
        update_on="normal",
        return_q_history=True,
        verbose_every=0,
        callback=None,
        return_logs=False,
    ):

        scores = np.asarray(scores, dtype=float).ravel()
        pvals = np.empty_like(scores, dtype=float)
        yhat = np.empty_like(scores, dtype=int)
        q_hist = [] if return_q_history else None
        logs = [] if return_logs else None

        for i, s in enumerate(scores):
            q_now = self.quantile
            pv = self.p_values([s])[0]
            pred = int(pv <= self.alpha)

            if return_logs:
                logs.append(
                    {
                        "i": i,
                        "score": float(s),
                        "pval": float(pv),
                        "pred": int(pred),
                        "q": float(q_now),
                        "calib_size": int(self.calib_size),
                    }
                )

            if verbose_every and (i % verbose_every == 0):
                print(
                    f"[i={i:>7}] score={s:.6f} p={pv:.4f} pred={pred} "
                    f"q={q_now:.6f} calib={self.calib_size}"
                )

            if callback is not None:
                try:
                    callback(
                        i=i,
                        score=float(s),
                        pval=float(pv),
                        pred=int(pred),
                        q=float(q_now),
                        calib_size=int(self.calib_size),
                    )
                except Exception as e:
                    print(f"[callback error at i={i}] {e}")
            pvals[i] = pv
            yhat[i] = pred

            if update_on == "normal" and pred == 0:
                self.update_with_normal(s)

            if return_q_history:
                q_hist.append(self.quantile)

        if return_q_history and return_logs:
            return pvals, yhat, np.asarray(q_hist, dtype=float), logs
        if return_q_history:
            return pvals, yhat, np.asarray(q_hist, dtype=float)
        if return_logs:
            return pvals, yhat, logs
        return pvals, yhat
