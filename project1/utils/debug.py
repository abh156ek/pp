# src/utils/debug.py

import json


def show_agent_reasoning(output, agent_name: str):
    """Pretty-print structured or JSON output from an agent."""
    print(f"\n{'=' * 10} {agent_name.center(28)} {'=' * 10}")

    def convert_to_serializable(obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        elif isinstance(obj, (int, float, bool, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return str(obj)

    if isinstance(output, (dict, list)):
        serializable_output = convert_to_serializable(output)
        print(json.dumps(serializable_output, indent=2))
    else:
        try:
            parsed_output = json.loads(output)
            print(json.dumps(parsed_output, indent=2))
        except json.JSONDecodeError:
            print(output)

    print("=" * 48)
