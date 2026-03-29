"""
app.py — Flask REST API for the SLA-Aware Multi-Cloud Cost Optimizer.
"""
import sys
import os

# Fix module resolution when running via gunicorn from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
from flask import Flask, request, jsonify, Response
from env.cloud_env  import CloudEnvironment, PROVIDERS
from env.models     import Observation, Action, Reward, StepResponse
from tasks.tasks    import TASKS, list_tasks, get_task
from baseline.baseline import run_baseline, run_baseline_on_task

global_env = None

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


def _error(message: str, code: int = 400) -> Response:
    return jsonify({"error": message}), code


def _validate_task_id(task_id: str):
    try:
        return get_task(task_id), None
    except KeyError:
        return None, _error(
            f"Unknown task_id '{task_id}'. Available task IDs: {list(TASKS.keys())}", 404
        )


@app.get("/reset")
def reset_env():
    global global_env
    global_env = CloudEnvironment()
    state = global_env.reset()
    obs = Observation(**state)
    return jsonify(obs.model_dump())


@app.post("/step")
def step_env():
    global global_env
    data = request.json
    if not data or "action" not in data:
        return _error("Request body must include 'action' field.", 400)
    try:
        action_model = Action(action=data["action"])
    except Exception as e:
        return _error(f"Invalid action: {e}. Must be one of: aws, azure, gcp.", 400)
    if not global_env:
        return _error("Environment not initialized. Call /reset first.", 400)
    state, reward, done, info = global_env.step(action_model.action)
    obs = Observation(**state)
    reward_model = Reward(reward=reward, done=done, info=info)
    return jsonify({
        "state" : obs.model_dump(),
        "reward": reward_model.reward,
        "done"  : reward_model.done,
        "info"  : reward_model.info,
    })


@app.get("/state")
def get_state():
    global global_env
    if not global_env:
        return _error("Environment not initialized. Call /reset first.", 400)
    obs = Observation(**global_env.get_state())
    return jsonify(obs.model_dump())


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "SLA-Aware Multi-Cloud Cost Optimizer"})


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "SLA-Aware Multi-Cloud Cost Optimizer API",
        "version": "1.0",
        "openenv_endpoints": {
            "GET  /reset" : "Start a new episode — returns initial observation",
            "POST /step"  : "Take an action (aws/azure/gcp) — returns reward & state",
            "GET  /state" : "Get current environment state without stepping",
        },
        "task_endpoints": {
            "GET  /tasks"             : "List all 5 benchmark tasks",
            "GET  /tasks/<task_id>"   : "Full provider metrics for a specific task",
            "POST /grader"            : "Score a cloud provider selection (0.0-1.0)",
            "GET  /baseline"          : "Run greedy baseline agent on all tasks",
            "GET  /baseline/<task_id>": "Run baseline on a single task",
            "GET  /compare/<task_id>" : "Compare all 3 providers on a task",
        },
        "other": {
            "GET  /health"     : "Liveness check",
            "GET  /leaderboard": "Top performers by reward score",
        }
    })


@app.get("/tasks")
def get_tasks():
    tasks = list_tasks()
    return jsonify({"count": len(tasks), "tasks": tasks})


@app.get("/tasks/<task_id>")
def get_task_detail(task_id: str):
    task, err = _validate_task_id(task_id)
    if err:
        return err
    public = {k: v for k, v in task.items() if k != "optimal_cloud"}
    return jsonify(public)


@app.route("/grader", methods=["POST"])
def grader():
    data = request.json
    if not data or "task_id" not in data or "selected_cloud" not in data:
        return {"error": "Invalid input"}, 400
    task_id = data["task_id"]
    selected_cloud = data["selected_cloud"]
    if task_id not in TASKS:
        return {"error": "Invalid task_id"}, 400
    task = TASKS[task_id]
    env = CloudEnvironment(task=task, noise=0.0)
    state = env.reset()
    next_state, reward, done, info = env.step(selected_cloud)
    baseline_result = run_baseline_on_task(task)
    baseline_reward = baseline_result["reward"]
    return jsonify({
        "task_id"            : task_id,
        "selected_cloud"     : selected_cloud,
        "cost"               : info["cost"],
        "latency"            : info["latency"],
        "sla_max_latency"    : task["sla_max_latency"],
        "sla_met"            : info["latency"] <= task["sla_max_latency"],
        "reward"             : round(reward, 4),
        "baseline_reward"    : round(baseline_reward, 4),
        "better_than_baseline": reward > baseline_reward,
        "is_optimal"         : selected_cloud == task.get("optimal_cloud"),
        "grade"              : _grade(reward, info["latency"] <= task["sla_max_latency"]),
        "timestamp"          : datetime.datetime.utcnow().isoformat()
    })


@app.get("/baseline")
def baseline_all():
    summary = run_baseline(verbose=False)
    summary["strategy"] = "cheapest_sla_compliant"
    return jsonify(summary)


@app.get("/baseline/<task_id>")
def baseline_single(task_id: str):
    task, err = _validate_task_id(task_id)
    if err:
        return err
    info = run_baseline_on_task(task)
    info["task_id"] = task_id
    return jsonify(info)


@app.get("/compare/<task_id>")
def compare_all_providers(task_id: str):
    task, err = _validate_task_id(task_id)
    if err:
        return err
    scores = {}
    for provider in PROVIDERS:
        env = CloudEnvironment(task=task, noise=0.0)
        env.reset()
        _, _, _, info = env.step(provider)
        scores[provider] = {
            "cost"   : info["cost"],
            "latency": info["latency"],
            "sla_met": info["sla_met"],
            "reward" : info["reward"],
        }
    best = max(scores, key=lambda p: scores[p]["reward"])
    return jsonify({
        "task_id"        : task_id,
        "sla_max_latency": task["sla_max_latency"],
        "scores"         : scores,
        "best_provider"  : best,
    })


@app.get("/compare_all")
def compare_all():
    output = {}
    for task_id in TASKS:
        env = CloudEnvironment(task=TASKS[task_id], noise=0.0)
        env.reset()
        scores = {}
        for provider in PROVIDERS:
            _, _, _, info = env.step(provider)
            scores[provider] = info
        output[task_id] = scores
    return jsonify(output)


@app.get("/leaderboard")
def leaderboard():
    results = run_baseline(verbose=False)["results"]
    sorted_results = sorted(results, key=lambda x: x["reward"], reverse=True)
    return jsonify({"top_performers": sorted_results, "message": "Leaderboard based on reward score"})


def _grade(reward: float, sla_met: bool) -> str:
    if not sla_met:   return "failed (SLA violation)"
    if reward >= 0.90: return "excellent"
    if reward >= 0.75: return "good"
    if reward >= 0.55: return "fair"
    return "poor"


@app.errorhandler(404)
def not_found(_):
    return _error("Endpoint not found.", 404)


@app.errorhandler(405)
def method_not_allowed(_):
    return _error("HTTP method not allowed for this endpoint.", 405)


@app.errorhandler(500)
def internal_error(exc):
    return _error(f"Internal server error: {exc}", 500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
