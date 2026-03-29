# 🚀 AI-Powered SLA-Aware Multi-Cloud Optimization Environment

> 🚀 Trains AI agents to make real-world cloud routing decisions under practical SLA constraints — similar to production systems used in large-scale cloud infrastructure.

> An OpenEnv-compatible reinforcement learning environment where AI agents learn to optimize **cost vs latency vs SLA trade-offs** across AWS, Azure, and GCP.

Built for the **Meta × PyTorch Hackathon 2026** hosted by Scaler School of Technology.

---

## 🧠 Problem

Modern cloud systems face complex decision-making challenges.

Choosing the cheapest provider is not enough — real-world systems must balance:

- SLA compliance  
- Cost vs performance trade-offs  
- Dynamic and context-aware decision-making  

Traditional rule-based systems fail to handle these multi-objective constraints effectively.

---

## 💡 Solution

We built an OpenEnv-compatible environment where AI agents:

- Select optimal cloud providers (AWS, Azure, GCP)  
- Respect strict latency constraints (SLA)  
- Optimize cost under real-world trade-offs  

This transforms cloud routing into a **reinforcement learning problem**, enabling intelligent agents to learn optimal strategies over time.

---

## 🧠 AI Component

This project is designed as an **AI-first system**:

- Frames cloud routing as a **reinforcement learning problem**  
- Uses a **reward function** to guide optimal decision-making  
- Supports integration with **PyTorch-based RL agents (PPO, DQN)**  
- Includes an **LLM-based agent baseline** for intelligent reasoning  

👉 Enables training, evaluation, and benchmarking of AI agents in real-world optimization scenarios.

---

## ✨ Key Features

- ✅ Full **OpenEnv spec** compliance — `step()`, `reset()`, `state()`, `openenv.yaml`
- ✅ **Pydantic typed models** — `Observation`, `Action`, `Reward`, `StepResponse`
- ✅ **5 benchmark tasks** spanning easy → medium → hard difficulty
- ✅ **Continuous reward** in `[0, 1]` — not sparse, not binary
- ✅ **LLM baseline agent** using OpenAI-compatible client
- ✅ **Deterministic graders** — reproducible evaluation
- ✅ Deployed on **Hugging Face Spaces** with Docker
- ✅ Flask REST API with full endpoint coverage

---

## 🌐 Live Demo

👉 https://nityanama-multi_cloud_optimizer.hf.space  

---

## 🎥 Demo
### Example Flow:

**Input:**
- Cloud providers data  
- Latency constraints  
- Cost parameters  

**Output:**
- Optimal provider selection  
- Reward score (0–1)  
- SLA compliance result  

### Sample Output:

```json
{
  "baseline_reward": 0.9033,
  "better_than_baseline": true,
  "cost": 40,
  "grade": "excellent",
  "is_optimal": true,
  "latency": 58,
  "reward": 0.9033,
  "selected_cloud": "gcp",
  "sla_max_latency": 90
}
```

---

## 🔁 How It Works

```
observation = env.reset()
action = agent.act(observation)
obs, reward, done, info = env.step(action)
```
- Agent observes cloud conditions  
- Selects provider  
- Receives reward based on SLA + cost efficiency  

---

## 🏆 Reward Function

```
reward = 0.0 ← SLA violated
       = 0.75 × cost_score
       + 0.15 × latency_headroom_ratio
       + 0.10 × efficiency_bonus
```

- Penalizes SLA violations  
- Rewards cost efficiency  
- Encourages optimal decision-making  



## 📊 Impact

This project can be used for:

- Training AI agents for real-world infrastructure decisions  
- Cloud cost optimization systems  
- Reinforcement learning research  
- Intelligent DevOps and FinOps tools  

---

## 📁 Project Structure

```
multi_cloud_optimizer/
├── env/
├── api/
├── tasks/
├── baseline/
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🤖 AI Baseline

- Greedy rule-based agent  
- LLM-based agent (Qwen)  
- Comparable performance benchmarking  

---

## 🔮 Future Scope

- Full RL agent training (PPO/DQN with PyTorch)  
- Dynamic pricing simulation  
- Multi-region cloud optimization  
- Frontend dashboard for visualization  
- Multi-agent benchmarking  

---

## 👥 Team

- Nitya Phaneesh Chandra Nama  
- Vanditha Hamsa S B  
- Chandan N  

---

## 📄 License

MIT — free to use for research and hackathon purposes.