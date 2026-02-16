"""
Tool definitions for the Scientist Agent.

These are exposed to Nemotron via OpenAI-style function-calling schema.
The orchestrator executes the actual tool logic; these are just descriptors.
"""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "set_variable",
            "description": (
                "Set the value of a modifiable chunk (independent variable). "
                "This changes the stimulus generation parameters for the next test. "
                "Available variables: item_count (int, 1-30), spacing_px (int, 5-200), "
                "item_size_px (int, 10-100), occlusion_level (float, 0.0-1.0), "
                "distractor_presence (bool), distractor_count (int, 0-20), "
                "background_color (hex string), item_color (hex string), "
                "distractor_color (hex string), shape ('circle'|'square'|'triangle'), "
                "contrast_level (float, 0.0-1.0)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the modifiable variable to change.",
                        "enum": [
                            "item_count", "spacing_px", "item_size_px",
                            "occlusion_level", "distractor_presence",
                            "distractor_count", "background_color",
                            "item_color", "distractor_color", "shape",
                            "contrast_level",
                        ],
                    },
                    "value": {
                        "description": "New value for the variable. Type depends on the variable.",
                    },
                },
                "required": ["name", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_isomorphism_batch",
            "description": (
                "Generate N isomorphic variations of the current scene configuration "
                "and evaluate each with the VLM. Returns the success rate (pass count / N) "
                "and individual results. Use this to validate whether a boundary is stable "
                "across random layout variations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of isomorphic variations to generate and test.",
                        "default": 5,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_attention",
            "description": (
                "Run the current scene through the VLM with attention extraction enabled "
                "and return a summary of where the model is attending. Returns a textual "
                "description of the attention distribution (e.g., 'concentrated on center', "
                "'diffuse across background', 'focused on items')."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "binary_search",
            "description": (
                "Perform a binary search on a numeric variable to find the exact "
                "transition point where the VLM's behavior changes (Pass→Fail or Fail→Pass). "
                "Returns the boundary value and transition type."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "variable": {
                        "type": "string",
                        "description": "Name of the numeric variable to search.",
                    },
                    "lo": {
                        "type": "number",
                        "description": "Lower bound of the search range.",
                    },
                    "hi": {
                        "type": "number",
                        "description": "Upper bound of the search range.",
                    },
                },
                "required": ["variable", "lo", "hi"],
            },
        },
    },
]
