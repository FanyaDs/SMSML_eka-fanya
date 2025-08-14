import time, requests
from prometheus_client import start_http_server, Summary, Counter, Gauge

PRED_LATENCY = Summary("iris_pred_latency_seconds", "Latency of /invocations")
PRED_COUNT   = Counter("iris_pred_total", "Total predictions")
PRED_ERRORS  = Counter("iris_pred_errors_total", "Total prediction errors")
HEALTH       = Gauge("iris_model_healthy", "Model health (1=up,0=down)")

MODEL_URL = "http://127.0.0.1:5001/invocations"
HEALTH_URL = "http://127.0.0.1:5001/ping"

def check_health():
    try:
        r = requests.get(HEALTH_URL, timeout=2)
        HEALTH.set(1 if r.status_code == 200 else 0)
    except Exception:
        HEALTH.set(0)

@PRED_LATENCY.time()
def predict_once():
    try:
        r = requests.post(MODEL_URL, json={"inputs": [[5.9,3.0,5.1,1.8]]}, timeout=3)
        if r.ok:
            PRED_COUNT.inc()
        else:
            PRED_ERRORS.inc()
    except Exception:
        PRED_ERRORS.inc()

if __name__ == "__main__":
    start_http_server(8000)  # expose /metrics
    while True:
        check_health()
        predict_once()
        time.sleep(5)
