providers_text = "\n".join(
        f"  {name}: cost=${metrics['cost']}, latency={metrics['latency']}ms"
        for name, metrics in sorted_providers
    )